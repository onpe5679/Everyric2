"""GPU 스크래치 반환 + idle 상주 가드 — 웜 캐시 상주 워커의 VRAM 위생.

PyTorch 캐싱 앨로케이터는 활성 버퍼가 해제돼도 예약 VRAM을 드라이버에 돌려주지 않고
사재기한다. 웜 캐시(WS2-A)로 프로세스가 상주하는 워커에서는 잡별 활성 스파이크(정렬
emission·f0 배치 등)의 예약이 그대로 남아, 동거 호스트 실측에서 18.4GiB까지 부풀었다
(2026-07-24 — 모델 실중량 합계는 3~6GiB뿐이라 나머지는 전부 사재기). 대책 2단:

1. 잡 경계마다 ``reclaim_after_job()`` — 미사용 예약 블록을 드라이버에 반환한다.
   웜 모델 가중치는 활성 할당이라 그대로 남는다 (웜 캐시 이점 유지).
2. 반환 후에도 예약이 임계(``EVERYRIC_SERVER_WORKER_VRAM_GUARD_GB``, 기본 8)를 넘으면
   진짜 참조 누수 회귀로 보고 웜 캐시(모델 싱글턴)를 버린 뒤 다시 반환한다 — 다음 잡이
   콜드 스타트 비용을 물지만 동거 서비스의 VRAM 기근보다는 싸다. 경고 로그로 회귀를 알린다.

앨로케이터 자체의 사재기 성향은 유닛 환경변수 ``PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True``(deploy/everyric2-worker-user.service)로 추가 완화한다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def release_scratch() -> float | None:
    """캐싱 앨로케이터의 미사용 예약 블록을 드라이버에 반환하고, 반환 후 예약 GiB를
    돌려준다. torch 미설치/CPU 전용이면 None (API 전용 프로세스에서 torch를 끌어오지
    않도록 임포트는 전부 함수 안)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return float(torch.cuda.memory_reserved()) / (1024**3)
    except Exception:
        return None


def guard_should_drop(reserved_gib: float | None, limit_gib: float) -> bool:
    """가드 판정만 분리한 순수 함수 — 테스트가 임계 논리를 GPU 없이 못 박는다."""
    return limit_gib > 0 and reserved_gib is not None and reserved_gib > limit_gib


def reclaim_after_job() -> None:
    """잡 경계 훅 — 스크래치 반환 + idle 상주 가드. 어떤 실패도 잡 처리로 전파하지 않는다."""
    try:
        reserved = release_scratch()
        if reserved is None:
            return
        from everyric2.config.settings import get_settings

        limit = get_settings().server.worker_vram_guard_gb
        if not guard_should_drop(reserved, limit):
            logger.info("idle VRAM 예약 %.2f GiB (한도 %s GiB)", reserved, limit)
            return
        logger.warning(
            "idle VRAM 예약 %.2f GiB > 한도 %s GiB — 참조 누수 회귀 의심, "
            "웜 캐시를 비우고 재적재해요",
            reserved,
            limit,
        )
        drop_warm_caches()
        after = release_scratch()
        if after is not None:
            logger.warning("웜 캐시 정리 후 idle VRAM 예약 %.2f GiB", after)
    except Exception:
        logger.exception("VRAM 회수 훅 실패 — 잡 처리에는 영향 없음")


def drop_warm_caches() -> None:
    """웜 모델 싱글턴 전부 해제 — 다음 잡에서 지연 재적재된다."""
    for name, clear in _clearers():
        try:
            clear()
        except Exception:
            logger.exception("웜 캐시 해제 실패: %s", name)


def _clearers():
    from everyric2.alignment.ctc_engine import clear_shared_ctc_engine
    from everyric2.audio.separator import clear_shared_separator
    from everyric2.melody.extractor import clear_shared_extractor

    return [
        ("ctc", clear_shared_ctc_engine),
        ("separator", clear_shared_separator),
        ("melody", clear_shared_extractor),
    ]
