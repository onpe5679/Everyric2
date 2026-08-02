"""정렬 엔진이 계산한 전곡 CTC emission을 2패스 리파이너에 노출하는 공용 계약.

동기: 벤치가 검증한 2패스 구조(라인 경계는 검증된 앵커가 정하고, 그 창 안에서만 경량
모델이 음절을 다시 잡는다 — ``scripts/bench_adapters/two_pass.py``)를 서버로 옮기려면
리파이너가 "곡 전체 emission을 한 번만 계산해 라인 창만큼 프레임 축을 잘라 쓴다"는 벤치의
``HFCTCAligner.emission_for``/``HFEmission`` 계약이 서버 엔진에도 있어야 한다. 이 모듈이
그 계약이다 — 2패스 리파이너를 이식하는 쪽은 이 dataclass에 맞춘다.

**모든 엔진이 구현하지는 않는다.** 서브프로세스로 격리된 엔진(``OwsmEngine`` — ESPnet은
메인 venv와 의존성이 충돌해 별도 인터프리터에서 돈다, ``owsm_engine.py`` 모듈 주석 참고)은
emission 텐서가 다른 프로세스에 살아 안전하게 못 넘어온다 — 그 엔진은
``BaseAlignmentEngine.emission_for``의 기본 구현(``None``)을 그대로 쓴다. 2패스가 그
엔진에서 실제로 필요로 하는 것은 텐서가 아니라 **라인 창**이고, 그것은 이미 ``align()``이
돌려주는 ``SyncResult.start_time``/``end_time``으로 충분하다(벤치의 ``two_pass.py``도
앵커의 emission이 아니라 앵커의 **정렬 결과 라인 시각**만 쓰고, 리파이너 쪽 emission만
``emission_for``로 얻는다 — 같은 구조). 인프로세스 wav2vec2 계열 엔진(``OmniASREngine``)만
실제 텐서를 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineEmission:
    """한 곡 전체의 CTC log-softmax emission과 그 프레임 척도.

    Attributes:
        emission: ``[1, T, V]`` log-softmax 텐서(``torch.Tensor``). CPU/GPU 어느 쪽에
            있는지는 엔진마다 다르다 — 호출부가 필요하면 옮긴다.
        blank_id: emission의 CTC blank 열 인덱스.
        frame_sec: 프레임 하나가 차지하는 오디오 초(= 오디오 길이 / T).
        audio_sec: 원본 오디오 길이(초).
        chunks: 이 emission을 만드는 데 쓴 겹침 청크 개수(1이면 통짜 forward와 동일).
        vocab: 토큰 문자열 -> emission 열 인덱스. 리파이너가 자기 타깃 텍스트를 이 vocab으로
            토큰화해야 슬라이스한 열이 실제로 그 글자를 가리킨다.
    """

    emission: Any
    blank_id: int
    frame_sec: float
    audio_sec: float
    chunks: int
    vocab: dict[str, int] = field(default_factory=dict)

    def frame_of(self, seconds: float) -> int:
        """초 -> 이 emission의 프레임 인덱스 (반올림). 경계 클램프는 호출부 책임."""
        return round(seconds / self.frame_sec) if self.frame_sec else 0
