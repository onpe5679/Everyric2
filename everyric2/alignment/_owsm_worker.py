"""OWSM-CTC 정렬의 실제 추론 — ``OwsmEngine``이 격리 venv에서 서브프로세스로 돌리는 워커.

**이 파일은 ``everyric2`` 패키지를 import하지 않는다** (한 줄 예외는 파일 경로로 직접 읽는
``everyric2/audio/chunking.py``뿐 — 아래 ``_load_chunking_module`` 참고). ESPnet은
``espnet2``/``sentencepiece`` 등 메인 .venv(torch>=2.0, transformers>=4.40)와 충돌하는
의존성 세트를 요구해서 별도 venv(``owsm_engine.py``의 ``_default_owsm_python`` 참고)에서만
돈다. ``everyric2.alignment.base``나 ``everyric2.config.settings``를 여기서 import하면
그 격리 venv에는 없는 pydantic-settings·sqlalchemy 등 전이 의존성이 딸려 들어와 죽는다 —
그래서 이 워커는 stdlib과 torch/torchaudio/numpy/sentencepiece/espnet2로만 산다.
``OwsmEngine``은 이 파일을 서브프로세스로 호출하기만 하고 자기 자신은 절대 실행하지 않는다.

핵심 실패 모드 (모델 형태가 강제하는 설계): 인코더(E-Branchformer 27블록,
``conv2d8`` 서브샘플링)가 ``[<lang>, <asr>]`` 2토큰 프리픽스에 조건화돼 있고 그 상태가
인코더 출력 **앞에 붙는다.** ``_verify_prefix_surplus``가 오디오만으로 기대되는 프레임
수를 프론트엔드/서브샘플링 산술로 직접 계산해 잉여가 정확히 프리픽스 길이와 같은지
검증하고, 어긋나면 즉시 죽는다 — 여기서 조용히 넘어가면 **모든 타임스탬프가 오류 없이
~160ms(프리픽스 2프레임 × 80ms) 밀린다**(값은 정상적으로 반환된다).

vocab: 5만 유니그램 SentencePiece라 토큰 하나가 가사 글자 여러 개를 덮을 수 있다
(``▁안녕하세요``가 토큰 하나). 그래서 이 워커는 모델의 **네이티브** 토큰화로 강제정렬하고
(그것이 CTC head가 실제로 학습한 것이고 정직한 posterior를 얻는 유일한 방법이다), 각
토큰의 측정된 스팬을 그 토큰이 덮는 가사 글자들에 **비례 분배**한다. 줄 경계는 항상 측정된
토큰 경계이고, 다중 글자 토큰 내부의 글자 경계만 보간이다.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# 학습된 채택 구성(벤치 ``owsm-ctc-v4-1b-bf16``) — fp32 대비 VRAM을 절반으로 줄이면서
# 정확도 손실이 측정되지 않았다(모듈 하위 문서 참고). float16은 이 인코더에서 오버플로해
# forced_align이 유한하지 않은 emission을 받아 죽는다(벤치 실측, dtype 선택 근거).
DEFAULT_DTYPE = "bfloat16"

EXP_DIR_NAME = "s2t_train_owsmctc_ebf27_conv2d8_size1024_mel128_bs320_raw_bpe50000"
MODEL_FILE_NAME = "valid.total_count.ave_5best.till70epoch.pth"


def _expected_audio_frames(n_samples: int, hop_length: int = 160) -> int:
    """오디오만 있을 때 인코더가 차지할 프레임 수: 로그멜 프레임 다음 ``conv2d8``.

    기본 ESPnet 프론트엔드는 중앙정렬 STFT(hop마다 프레임 1개 + 1개)이고,
    ``Conv2dSubsampling8``이 커널3·스트라이드2 컨볼루션을 시간축에 세 번 적용한다.
    """
    frames = n_samples // hop_length + 1
    for _ in range(3):
        frames = (frames - 3) // 2 + 1
    return frames


def _verify_prefix_surplus(total_frames: int, audio_frames: int, prefix_len: int) -> None:
    """인코더 출력의 잉여 프레임이 정확히 프리픽스 길이와 같은지 검증한다.

    모듈 docstring의 핵심 실패 모드 방어. 어긋나면 침묵 대신 즉시 크래시한다 — 프리픽스
    프레임이 오디오로 오인되면 이후 모든 시각 계산이 오류 없이 조용히 밀리기 때문이다.
    """
    surplus = total_frames - audio_frames
    if surplus != prefix_len:
        raise RuntimeError(
            f"unexpected OWSM encoder length: got {total_frames} frames, expected "
            f"{audio_frames} audio frames + {prefix_len} prefix frames"
        )


def _is_alignment_character(char: str) -> bool:
    """정렬 타깃이 될 수 있는 글자인가 (공백·구두점·기호 제외)."""
    if char.isspace():
        return False
    category = unicodedata.category(char)
    if category[0] in {"P", "S", "C", "Z"}:
        return False
    return category[0] in {"L", "M", "N"} or 0x4E00 <= ord(char) <= 0x9FFF


def _confidence(log_score: float) -> float:
    import math

    if not math.isfinite(log_score):
        return 0.0
    return round(math.exp(min(0.0, log_score)), 6)


def _load_chunking_module() -> Any:
    """``everyric2.audio.chunking``을 파일 경로로 직접 로드한다 (패키지 import 없이).

    ``everyric2.audio.__init__``이 다운로더/로더 스택을 끌어오는데 이 격리 venv에는
    설치돼 있지 않다. chunking 모듈 자체는 numpy만 있으면 되므로 파일로 로드하면 그
    충돌 없이 겹침-스티칭 구현을 서버와 하나만 공유할 수 있다.
    """
    import importlib.util

    path = REPO_ROOT / "everyric2" / "audio" / "chunking.py"
    spec = importlib.util.spec_from_file_location("_owsm_worker_chunking", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load chunking helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(request_path: Path, response_path: Path) -> int:
    """요청 JSON을 읽어 정렬하고 응답 JSON을 쓴다. 실패하면 예외를 던져 서브프로세스가
    0이 아닌 코드로 죽는다 — 부모(``OwsmEngine._run_worker``)가 stdout/stderr를 실어
    ``AlignmentError``로 감싼다."""

    import time

    import torch
    import torchaudio
    import torchaudio.functional as functional

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    snapshot = Path(payload["snapshot"])
    exp_dir = snapshot / "exp" / EXP_DIR_NAME
    dtype = str(payload.get("dtype") or DEFAULT_DTYPE)

    # config.yaml이 feats_stats.npz를 스냅샷 상대경로로 저장한다.
    import os

    os.chdir(snapshot)

    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    # CPU에서 먼저 만들고 나서 GPU로 옮긴다 — ESPnet이 fp32 가중치를 만든 뒤에 dtype으로
    # 캐스팅해서, cuda 위에서 바로 만들면 fp32 사본과 캐스팅 결과가 동시에 카드에 올라간다.
    s2t = Speech2TextGreedySearch(
        s2t_train_config=str(exp_dir / "config.yaml"),
        s2t_model_file=str(exp_dir / MODEL_FILE_NAME),
        bpemodel=str(snapshot / "data" / "token_list" / "bpe_unigram50000" / "bpe.model"),
        device="cpu",
        dtype=dtype,
        lang_sym=payload["lang_sym"],
        task_sym="<asr>",
        use_flash_attn=False,
    )
    if device != "cpu":
        s2t.s2t_model.to(device)
        s2t.device = device
    load_sec = round(time.perf_counter() - started, 2)

    model = s2t.s2t_model
    torch_dtype = getattr(torch, dtype)
    token_list = list(s2t.s2t_train_args.token_list)
    token_to_id = {token: index for index, token in enumerate(token_list)}
    blank_id = int(model.blank_id)
    # 패딩된 버퍼 길이 — 학습 길이(``preprocessor_conf.speech_length``)를 기본으로 쓴다.
    buffer_sec = float(s2t.s2t_train_args.preprocessor_conf["speech_length"])

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(
        model_file=str(snapshot / "data" / "token_list" / "bpe_unigram50000" / "bpe.model")
    )

    # ── 타깃: 모델의 네이티브 SentencePiece 조각을 원문 글자로 되짚는다 ──
    lines: list[str] = payload["lines"]
    target_ids: list[int] = []
    token_owners: list[tuple[int, list[int]]] = []  # (줄 인덱스, [이 토큰이 덮는 글자 인덱스들])
    line_totals: list[int] = []
    for line_index, line in enumerate(lines):
        alignable = [i for i, char in enumerate(line) if _is_alignment_character(char)]
        line_totals.append(len(alignable))
        alignable_set = set(alignable)
        for piece in sp.encode(line, out_type="immutable_proto").pieces:
            token_id = token_to_id.get(piece.piece)
            if token_id is None or token_id == blank_id:
                continue
            covered = [i for i in range(piece.begin, piece.end) if i in alignable_set]
            if not covered:
                continue
            target_ids.append(token_id)
            token_owners.append((line_index, covered))

    if not target_ids:
        raise RuntimeError("OWSM found no in-vocabulary lyric tokens")

    # ── 오디오 ──
    waveform, sample_rate = torchaudio.load(payload["vocals_path"])
    waveform = waveform.mean(dim=0)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    waveform = waveform.to(dtype=torch.float32, device="cpu").contiguous()
    n_samples = int(waveform.numel())
    audio_sec = n_samples / 16_000

    chunking = _load_chunking_module()
    buffer_samples = int(buffer_sec * 16_000)
    windows = chunking.plan_chunk_windows(
        n_samples, buffer_samples, int(float(payload["overlap_sec"]) * 16_000)
    )

    # 50,002폭 emission은 blank와 실제 타깃 열만 필요하므로 그 열들로만 gather한다.
    compact_tokens = [blank_id] + sorted({t for t in target_ids if t != blank_id})
    compact_index = {token: position for position, token in enumerate(compact_tokens)}
    column_index = torch.tensor(compact_tokens, dtype=torch.long, device=device)
    compact_targets = [compact_index[t] for t in target_ids]

    text_prev = torch.tensor([[model.na]], dtype=torch.long, device=device)
    text_prev_lengths = text_prev.new_full([1], dtype=torch.long, fill_value=1)
    prefix = torch.tensor(
        [[token_to_id[payload["lang_sym"]], token_to_id["<asr>"]]], dtype=torch.long, device=device
    )
    prefix_lengths = prefix.new_full([1], dtype=torch.long, fill_value=prefix.size(1))

    buffer_frames = _expected_audio_frames(buffer_samples)
    pieces: list[Any] = []
    for start, end in windows:
        segment = waveform[start:end]
        real = int(segment.numel())
        if real < buffer_samples:
            segment = torch.nn.functional.pad(segment, (0, buffer_samples - real))
        speech = segment.unsqueeze(0).to(device=device, dtype=torch_dtype)
        speech_lengths = torch.full([1], speech.size(1), dtype=torch.long, device=device)
        with torch.no_grad():
            enc, _ = model.encode(
                speech=speech,
                speech_lengths=speech_lengths,
                text_prev=text_prev,
                text_prev_lengths=text_prev_lengths,
                prefix=prefix,
                prefix_lengths=prefix_lengths,
            )
            if isinstance(enc, tuple):
                enc = enc[0]
            # ``model.ctc.log_softmax``와 동일하되 fp32에서 정규화한다 — 5만 폭 log_softmax를
            # 반정밀도로 하면 CTC DP가 필요로 하는 blank 근처 확률의 해상도가 사라진다.
            logp = torch.log_softmax(model.ctc.ctc_lo(enc).float(), dim=-1)

        # ── 핵심 실패 모드 방어 (모듈 docstring 참고) ──
        _verify_prefix_surplus(int(logp.shape[1]), buffer_frames, int(prefix.size(1)))
        sliced = logp[:, int(prefix.size(1)) :, :]
        compact = torch.index_select(sliced, 2, column_index).cpu()
        valid = max(1, min(buffer_frames, round(buffer_frames * real / buffer_samples)))
        pieces.append(compact[:, :valid, :])

    emission = (
        pieces[0]
        if len(pieces) == 1
        else chunking.stitch_chunk_outputs(pieces, windows, n_samples, frame_axis=1)
    )
    emission = emission.float().contiguous()

    tensor = torch.tensor([compact_targets], dtype=torch.int32)
    tokens_path, scores_path = functional.forced_align(emission, tensor, blank=0)
    token_spans = functional.merge_tokens(tokens_path[0], scores_path[0], blank=0)
    if len(token_spans) != len(compact_targets):
        raise RuntimeError(
            f"OWSM produced {len(token_spans)} spans for {len(compact_targets)} target tokens"
        )

    ratio = n_samples / int(emission.shape[1]) / 16_000

    # ── 스팬 -> 줄별 글자 세그 (다중 글자 토큰은 커버 글자 수만큼 비례 분배) ──
    out_lines: list[dict[str, Any]] = [
        {"segs": [], "total_chars": total, "matched_chars": 0, "tokens": 0} for total in line_totals
    ]
    for span, (line_index, covered) in zip(token_spans, token_owners):
        start_sec = float(span.start) * ratio
        end_sec = float(span.end) * ratio
        entry = out_lines[line_index]
        entry["tokens"] += 1
        confidence = _confidence(float(span.score))
        step = (end_sec - start_sec) / len(covered)
        for position, char_index in enumerate(covered):
            entry["segs"].append(
                {
                    "t": lines[line_index][char_index],
                    "start": round(start_sec + position * step, 3),
                    "end": round(start_sec + (position + 1) * step, 3),
                    "confidence": confidence,
                }
            )
        entry["matched_chars"] += len(covered)

    for entry in out_lines:
        entry["segs"].sort(key=lambda seg: seg["start"])

    response = {
        "audio_sec": round(audio_sec, 3),
        "frames": int(emission.shape[1]),
        "frame_sec": round(ratio, 6),
        "chunks": len(windows),
        "load_sec": load_sec,
        "dtype": dtype,
        "vocab_size": len(token_list),
        "lines": out_lines,
    }
    if device == "cuda":
        response["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        response["vram_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)

    response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OwsmEngine의 격리 venv 서브프로세스 워커")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    return run(Path(args.request), Path(args.response))


if __name__ == "__main__":
    raise SystemExit(main())
