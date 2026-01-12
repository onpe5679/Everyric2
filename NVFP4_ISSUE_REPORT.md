# NVFP4 모델 로딩 문제 리포트

## 환경
- **OS**: WSL2 Ubuntu 24.04
- **GPU**: NVIDIA RTX 5090 32GB VRAM + 56GB 공유 GPU 메모리
- **CUDA**: 12.x
- **Python**: 3.12
- **vLLM**: 0.13.0
- **transformers**: 4.57.3

## 목표
`cybermotaz/Qwen3-Omni-30B-A3B-Instruct-NVFP4` 모델 (25.68GB)을 RTX 5090에서 실행

## 모델 구조
```
Qwen3-Omni-30B-A3B-Instruct-NVFP4/
├── thinker/     # NVFP4 양자화 (20GB) - LLM 코어
├── talker/      # BF16 (6.2GB) - 음성 생성
├── code2wav/    # BF16 (413MB) - 오디오 합성
└── tokenizer files
```

## 시도한 방법들

### 1. 제공된 Docker 이미지 사용
```bash
docker pull mutazai/qwen3omni-30b-nvfp4:1.0
docker run -d --gpus all -v $(pwd)/model:/model -p 8000:8000 mutazai/qwen3omni-30b-nvfp4:1.0
```

**결과**: ❌ 실패
```
WARNING: The requested image's platform (linux/arm64) does not match the detected host platform (linux/amd64/v4)
exec /opt/nvidia/nvidia_entrypoint.sh: exec format error
```
**원인**: Docker 이미지가 ARM64 (Apple Silicon/Blackwell DGX)용으로 빌드됨

### 2. vLLM 직접 로드 (WSL 환경)
```python
from vllm import LLM
llm = LLM(
    model="/model/thinker",
    quantization="modelopt_fp4",
    trust_remote_code=True,
    kv_cache_dtype="fp8",
)
```

**결과**: ❌ 실패
```
ValueError: The checkpoint you are trying to load has model type `qwen3_omni_moe_thinker` 
but Transformers does not recognize this architecture.
```
**원인**: NVFP4 모델의 `model_type`이 `qwen3_omni_moe_thinker`로 설정됨 (표준 아님)

### 3. config.json 수정 후 재시도
```json
{
  "model_type": "qwen3_omni_moe",
  "architectures": ["Qwen3OmniMoeForConditionalGeneration"]
}
```

**결과**: 부분 성공 → 다른 에러
```
AttributeError: 'Qwen3OmniMoeVisionEncoderConfig' object has no attribute 'image_size'
```

### 4. vLLM 소스 코드 패치
`/vllm/model_executor/models/qwen3_omni_moe_thinker.py` 수정:
```python
# Before
self.image_size = vision_config.image_size

# After  
self.image_size = getattr(vision_config, "image_size", 448)
```

**결과**: 부분 성공 → 다른 에러
```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol
```

### 5. flash-attn 제거 후 재시도
```bash
pip uninstall flash-attn -y
```

**결과**: ❌ 실패
```
ValueError: Unsupported model when in features size is not multiple of 16
```
**원인**: vLLM의 ModelOpt NVFP4 양자화 구현이 이 모델 구조를 지원하지 않음

### 6. TensorRT-LLM Triton 컨테이너 사용
```bash
docker run --gpus all -it nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
pip install vllm
```

**결과**: 동일한 에러 발생

## 근본 원인 분석

### 1. Docker 이미지 플랫폼 불일치
- 제공된 이미지: `linux/arm64` (GB10/DGX Spark용)
- 호스트 플랫폼: `linux/amd64` (x86_64)
- RTX 5090은 x86_64 시스템에 설치되므로 ARM64 이미지 실행 불가

### 2. 모델 아키텍처 비호환
- NVFP4 모델은 `qwen3_omni_moe_thinker`라는 커스텀 모델 타입 사용
- 표준 transformers/vLLM은 이 타입을 인식하지 못함
- 모델 제작자가 NVIDIA Model Optimizer로 양자화 시 구조 변경

### 3. vLLM ModelOpt 양자화 제한
- vLLM 0.13.0의 `modelopt_fp4` 양자화는 실험적 기능
- 특정 모델 구조 (features size가 16의 배수)만 지원
- Qwen3-Omni의 MoE 구조와 호환되지 않음

## 가능한 해결 방법

### Option A: AMD64용 Docker 이미지 빌드
- 모델 제작자의 Dockerfile을 받아서 x86_64용으로 재빌드
- 예상 소요 시간: 2-4시간
- 필요한 것: Dockerfile, 빌드 환경

### Option B: TensorRT-LLM으로 엔진 빌드
- 원본 Qwen3-Omni 모델을 TensorRT-LLM으로 직접 양자화
- GitHub Issue #5018 참고 (RTX 5090 FP4 예제 있음)
- 예상 소요 시간: 4-8시간
- 복잡도: 높음

### Option C: vLLM 패치 확장
- `modelopt.py`의 16배수 제한 우회
- `qwen3_omni_moe_thinker.py` 전체 패치
- 예상 소요 시간: 2-4시간
- 위험도: 높음 (런타임 에러 가능)

### Option D: 모델 제작자에게 AMD64 이미지 요청
- HuggingFace Discussion 또는 GitHub Issue 생성
- 가장 확실한 방법
- 소요 시간: 제작자 응답에 따라 다름

## 참고 자료
- 모델 카드: https://huggingface.co/cybermotaz/Qwen3-Omni-30B-A3B-Instruct-NVFP4
- vLLM 5090 지원 이슈: https://github.com/vllm-project/vllm/issues/13306
- TensorRT-LLM FP4 예제: https://github.com/NVIDIA/TensorRT-LLM/issues/5018

## 현재 상태
Docker와 NVIDIA Container Toolkit이 WSL2에 설치됨. TensorRT-LLM Triton 컨테이너 실행 중.

```bash
# 실행 중인 컨테이너
sudo docker ps
CONTAINER ID   IMAGE                                            STATUS
8ef8bdc2fb0b   nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3   Up

# GPU 인식 확인됨
sudo docker exec trtllm nvidia-smi
NVIDIA GeForce RTX 5090, 32607 MiB
```
