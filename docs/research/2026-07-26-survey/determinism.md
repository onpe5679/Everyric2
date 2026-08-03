# determinism — 정렬 비결정성·재현성 조사 (SendMessage 전문 1건)

## 결론 먼저

forced_align의 CUDA 커널 자체는 **결정적**입니다(GitHub 소스로 확인). 비결정성의 근원은 그 입력인 emission이 GPU에서 실행마다 미세하게(추정 1e-4~1e-3) 달라지는 데 있고, everyric2 코드 자체가 그 오차를 키우는 스위치를 켜고 있습니다: `everyric2\cli.py:22-26`이 모델 로드 전에 전역으로

```
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

를 켭니다(주석: "Enable TF32 for RTX 30/40/50 series"). TF32는 가수부를 10비트로 잘라 반올림 오차가 fp32 대비 수백~수천 배 큽니다. posterior가 평평한(quality_score 낮은) 곡일수록 이 미세 오차가 Viterbi 경로의 argmax를 뒤집을 확률이 커진다는 것이 실측(0.0068곡 3줄·0.74초 vs 0.0012곡 32줄)과 정확히 맞습니다.

## 원인 후보별 (근거 · 해당여부 · 확인방법 · 대응책/비용)

**1. forced_align의 DP/Viterbi 자체 — 원인 아님 (확인됨)**
- 근거: `pytorch/audio`의 `src/libtorchaudio/forced_align/gpu/compute.cu` — `BlockReduce(...).Reduce(threadMax, thrust::maximum)`로 최댓값을 구하고, 동점 시 `x2>x1 && x2>x0` → `x1>x0 && x1>x2` → else(x0) 순서의 **명시적** 우선순위로 역포인터 결정. atomicAdd나 스케줄러 의존 리덕션 없음.
- 해당 여부: 없음 — 같은 emission이면 forced_align은 항상 같은 경로를 냄.
- 확인 방법: 저장된 emission 하나 위에서 `F.forced_align`을 수십 번 반복 호출(모델 forward 없이 DP만 — 비용 거의 0). 완전히 같은 출력인지 확인.
- 대응책: 불필요.

**2. cuBLAS/cuDNN 알고리즘 선택 nondeterminism — 유력**
- 근거: PyTorch 공식 문서(`docs.pytorch.org/docs/2.8/notes/randomness`) — "cuDNN library...can be a source of nondeterminism across multiple executions"; CUDA≥10.2에서 cuBLAS mm/mv/bmm 계열은 `CUBLAS_WORKSPACE_CONFIG=:4096:8`(또는 `:16:8`) 미설정 시 비결정적이라고 명시.
- 해당 여부: 가능성 높음 — wav2vec2/MMS는 conv1d(cudnn) + 어텐션/FFN(cuBLAS GEMM) 조합이고, 저장소 전체를 grep해도 `CUBLAS_WORKSPACE_CONFIG`나 `torch.use_deterministic_algorithms` 호출이 없음.
- 확인 방법: 같은 파형으로 모델 forward만 2회 돌려 logits를 `torch.save` 후 `torch.equal` 비교.
- 대응책: `CUBLAS_WORKSPACE_CONFIG=:4096:8`(+24MiB) + `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True`. 비용: 문헌상 처리량 2~5배 저하가 흔하나(PyTorch 문서, `pytorch/pytorch#109856` "severe performance regression") 커널 shape 의존적 — 저희 42초 기준 정확한 수치는 실측 필요(실험 4).

**3. TF32(everyric2\cli.py:22-26) — 유력, 저희 코드가 직접 켬**
- 근거: NVIDIA cuBLAS 문서 — TF32는 가수부 10비트 절단(사실상 fp16급 정밀도). PyTorch 코드베이스 자체 확인.
- 해당 여부: 100% 해당.
- 확인 방법: 이 세 줄을 끈 채 같은 실험 반복(실험 2).
- 대응책: `torch.set_float32_matmul_precision("highest")`로 낮추면 반올림 오차가 줄어 near-tie 뒤집힘 빈도 감소 기대(완전한 비트 재현은 아님 — 오차 "크기"만 줄임, CUBLAS_WORKSPACE_CONFIG와는 별개 축). 비용: 텐서코어 가속(대개 2~3배) 포기.

**4. CTC loss backward의 atomicAdd 비결정성 — 해당 없음 (흔한 오해 주의)**
- 근거: `pytorch/pytorch#17798`, `#21680` — `ctc_loss()`의 **backward**(gradient) 커널이 atomicAdd를 써서 비결정적(`LossCTC.cu`).
- 해당 여부: 없음 — everyric2는 `forced_align`(추론 전용)만 쓰고 `CTCLoss`나 `.backward()`를 호출하지 않음(`torch.inference_mode()`로 감쌈, ctc_engine.py 전체 확인). "CTC는 원래 비결정적"이라는 소문의 근거가 대개 이 이슈인데, 저희 경우엔 무관합니다.

**5. Demucs 분리 단계 — 미확인, 검증 필요**
- 근거: 확실한 1차 근거를 못 찾음. htdemucs는 conv+양방향 LSTM 블록을 쓰는데 cudnn LSTM도 비결정성 보고 사례가 있음(`pytorch/pytorch#35661`, 다만 dropout+멀티레이어 학습 맥락이라 추론에 그대로 적용되는지는 별도 확인 필요).
- 확인 방법: 실험 3 — Demucs 출력을 두 실행에서 해시 비교해 정렬 이전 단계에서부터 갈리는지 먼저 확인. 갈리면 forced_align/TF32 대응만으로는 문제의 절반만 고치는 셈.

## 특히 물으신 것

- **추론만 하는데 비결정성이 생기는 경로**: forced_align 자체가 아니라(#1) 그 입력 emission이 매 실행 GPU에서 미세하게 달라지기 때문. 원인은 TF32로 정밀도를 낮춘 것(#3, 저희가 직접 켬)과 cuBLAS/cuDNN이 워크스페이스 설정 없이는 알고리즘을 실행마다 다르게 고를 수 있다는 것(#2)의 조합.
- **torch.use_deterministic_algorithms(True) 성능 대가**: 문헌상 2~5배 저하가 흔하나 형태 의존적. 곡당 42초 기준 정확한 증가분은 이번 조사로 찾지 못함 — 실험 4로 직접 재는 것을 권합니다.
- **A/B 측정 설계**: 완전한 결정성 확보 전이면 단일 실행 비교는 무효. (a) 설정당 N회(제안 N=10) 반복해 라인별 이동량의 median/IQR 보고, (b) 두 설정 비교는 같은 곡·같은 라인을 짝지어 Wilcoxon signed-rank test 등 paired 비검정 방법 사용, (c) 분산이 quality_score에 따라 이질적이므로(실측으로 이미 확인됨) 신뢰도 구간별로 층화해서 봐야 함 — 평균 하나로 뭉치면 저신뢰 곡의 큰 흔들림이 희석됩니다.
- **평평한 posterior를 탐지 지표로 쓸 수 있는가**: 가능하고 재료도 이미 있습니다. `ctc_engine.py:435-468`의 `_token_peak_support`(글자별 최고 로그확률 평균)가 사실상 "봉우리 위에 있는가"를 잽니다. 더 직접적으로는 emission 한 벌에 TF32/cuBLAS 오차 규모의 인공 잡음을 얹어 forced_align을 CPU에서 수십 회 재실행(모델 forward 재실행 없음, 비용 거의 0)해 이동량 분산 자체를 곡 단위 "재현성 위험 지표"로 쓸 수 있습니다(실험 5). 참고로 `arXiv:2406.02560`(Less Peaky and More Accurate CTC Forced Alignment by Label Priors)은 CTC posterior의 peakiness를 다루지만 "너무 뾰족한 것도 정확도엔 문제"라는 별개 논의라서, 재현성 지표의 직접 근거로 인용하지는 않았습니다.

## 원인 확정을 위한 실험 순서 (GPU 1대, 곡 2~3개)

1. **emission 자체가 갈리는지 확인.** 다음 정상 실행에 `torch.save(emission.cpu(), ...)` 한 줄만 추가해 같은 곡을 두 번(같은/다른 프로세스) 저장 → `torch.equal` + `(a-b).abs().max()` 비교. 완전히 같다면 이 조사의 전제가 바뀌므로 반드시 먼저 확인.
2. **TF32 스위치로 원인 분리.** `cli.py:22-26`을 끈 채 같은 곡 2회 → 1과 동일 비교. 갈림이 사라지거나 크게 줄면 #3(TF32)이 주범, 그래도 갈리면 #2(cuBLAS/cuDNN 알고리즘 선택)까지 관여.
3. **Demucs도 오염원인지 분리.** 분리된 보컬 파형을 두 실행에서 해시 비교.
4. **결정적 모드의 실제 비용 실측.** `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True`를 켠 채 곡 2~3개 정렬해 42초 기준과 처리시간 비교 — 문헌 수치 대신 저희 환경의 실측값.
5. **재현성 위험 지표 검증.** 1에서 저장한 emission에 인공 잡음(상대 1e-3 등)을 얹어 CPU에서 forced_align 30~50회 반복 → 그 이동량 분산이 실제 GPU 두 실행 간 이동량과 같은 자릿수인지 확인. 맞으면 "posterior 평평함→노이즈 민감도" 설명이 확정되고 그 분산을 곡 단위 신뢰도 지표로 채택할 근거가 생김.

제약 준수: 파일 수정·커밋 없음, 로컬 무거운 연산 실행 없음(코드 읽기와 웹 조사만 수행), 원격 서버(100.76.4.47) 접속 없음.

Sources:
- [torch.use_deterministic_algorithms — PyTorch 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html)
- [Reproducibility — PyTorch docs (randomness notes)](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
- [Severe performance regression on deterministic algorithm in torch 2.0 · Issue #109856 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/109856)
- [Feature Request: deterministic CUDA torch.nn.CTCLoss · Issue #17798 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/17798)
- [Disable nondeterministic CTCLoss from cuDNN · Issue #21680 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/21680)
- [pytorch/aten/src/ATen/native/cuda/LossCTC.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/LossCTC.cu)
- [torchaudio forced_align GPU kernel source](https://github.com/pytorch/audio/blob/main/src/libtorchaudio/forced_align/gpu/compute.cu)
- [torchaudio.functional.forced_align docs](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html)
- [Update on TorchAudio's future · Issue #3902 · pytorch/audio](https://github.com/pytorch/audio/issues/3902)
- [Defeating Nondeterminism in LLM Inference — Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference (arXiv:2506.09501)](https://arxiv.org/html/2506.09501v2)
- [nn.LSTM gives nondeterministic results · Issue #35661 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/35661)
- [Less Peaky and More Accurate CTC Forced Alignment by Label Priors (arXiv:2406.02560)](https://arxiv.org/html/2406.02560v3)
