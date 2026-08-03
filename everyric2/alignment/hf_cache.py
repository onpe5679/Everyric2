"""HuggingFace 캐시에서 스냅샷 파일을 찾는 공용 헬퍼.

``owsm_engine.py``와 ``omniasr_engine.py``가 함께 쓴다 — 둘 다 벤치가 쓰던 방식(HF 캐시
스냅샷 탐색)을 그대로 따르되, **캐시에 없으면 조용히 네트워크로 받아오지 않는다.** 벤치의
일부 어댑터(``omni_ctc.py``)는 캐시 미스 시 ``huggingface_hub.hf_hub_download``로
자동 다운로드했지만, 서버 요청 경로 한복판에서 수백MB~수GB짜리 모델을 예고 없이 받는 것은
운영자에게 안 보이는 지연·디스크 사용이라 그 폴백을 들이지 않는다 — 없으면
``EngineNotAvailableError``(호출부 책임)다. 가중치 조달은 배포 시 별도 스크립트로 미리
캐시를 채우는 것을 전제로 한다(OWSM의 ESPnet 스냅샷도 애초에 수동 조달 대상이었다).
"""

from __future__ import annotations

import os
from pathlib import Path


def find_cached_file(model_id: str, filename: str) -> Path | None:
    """``model_id`` 스냅샷 아래에서 ``filename``(글롭 패턴 가능)의 최신 캐시 파일을 찾는다.

    ``TRANSFORMERS_CACHE``와 ``HF_HOME``(비어 있으면 ``~/.cache/huggingface/hub``를 HF의
    기본값으로 보되, 여기서는 명시적으로 설정된 경우만 본다) 아래
    ``models--<org>--<name>/snapshots/*/<filename>``을 뒤진다. 여러 스냅샷이 있으면 mtime이
    가장 최근인 파일을 고른다. 못 찾으면 ``None`` — 예외를 던지지 않는다(호출부가 존재
    여부로 ``is_available()``을 판정할 수 있어야 한다).
    """

    roots: list[Path] = []
    for variable in ("TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value)
        if variable == "HF_HOME" and root.name.lower() != "hub":
            root = root / "hub"
        if root not in roots:
            roots.append(root)

    slug = "models--" + model_id.replace("/", "--")
    matches: list[Path] = []
    for root in roots:
        matches.extend(root.glob(f"{slug}/snapshots/*/{filename}"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None
