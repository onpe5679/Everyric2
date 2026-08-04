"""배포 후(그리고 배포 전 기준선) 검증 — 체크리스트를 문서로만 두면 사람이 빠뜨린다.

이번 배치(1.6.0)의 docs/releases/chrome-v1.6.0.md "배포 후 검증 체크리스트"를 그대로
스크립트 하나로 묶는다. 기본 실행은 **전부 읽기 전용**(쓰기·생성 없음) — ④ X-API-Key
우회 차단의 실제 회귀 검사(보호된 쓰기 엔드포인트를 키 없이 호출)는 --check-auth를
명시해야만 돈다(아래 ④ 설명 참고, 2026-08-04 오탐 사고 이후 수정).

실행:
    .venv/Scripts/python.exe scripts/verify_deploy.py [서버URL]
    (서버URL 생략 시 프로드 기본값 https://everyric.moref.co)
    --check-auth 를 추가하면 ④에서 실제 쓰기 호출까지 시도한다(이 배포가 애초에
    api_key를 요구하는 배포일 때만 — 공개 배포면 이 플래그를 줘도 SKIP).

키가 필요한 항목은 없다 — ④의 실제 회귀 검사조차 **키를 안 보내는 것**이 목적이라,
이 스크립트는 admin_api_key/api_key를 아예 다루지 않는다(어떤 출력에도 찍을 값이 없다).

en/zh 발음 의존성 검사(⑤)용 video_id는 환경변수로 바꿀 수 있다:
    EVERYRIC_EN_VIDEO_ID (기본값 M7VSEZOQIlg — 2026-08-04 확인, 프로드에 실싱크 있음)
    EVERYRIC_ZH_VIDEO_ID (기본값 없음 — 이 코퍼스에 실측으로 확인된 zh 실싱크가 아직
      없어서 하드코딩하지 않았다. 값을 안 주면 이 항목은 SKIP으로 표시된다 — "zh 발음이
      비었다"는 오탐을 내느니 운영자가 실제 zh 곡 video_id를 알게 됐을 때 넣게 한다.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_SERVER = "https://everyric.moref.co"
DEFAULT_EN_VIDEO_ID = "M7VSEZOQIlg"
# 정상 형식(11자, _VIDEO_ID_PATTERN)이지만 실제 유튜브 영상일 리 없는 표식 — ④에서
# 우회가 살아 있어도(최악의 경우) 존재하지 않는 영상이라 다운로드 단계에서 바로 죽는다.
FAKE_VIDEO_ID = "_GATECHECK0"
FAKE_LYRICS = "deploy gate check line one\ndeploy gate check line two"


def _http(
    server: str, path: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 15.0
) -> tuple[int | None, object]:
    """읽기/쓰기 공용 최소 HTTP 클라이언트. 반환은 (status, parsed_json_or_raw_or_error)."""
    url = server.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    # Cloudflare가 /api/* 경로에서 urllib 기본 UA("Python-urllib/x.y")를 봇으로 보고
    # 403(error code 1010)으로 차단한다(2026-08-04 실측 — curl 기본 UA는 통과, 같은 curl에
    # python UA를 실어도 막힘 — 우리 앱 로직과 무관한 WAF 규칙임을 재현 확인). 우리
    # origin에 도달하기 전에 막히면 인증 검사(④) 결과가 "우회 살아있음"으로 오판될 수
    # 있어 반드시 일반 브라우저 UA를 실어야 한다.
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, (json.loads(body) if body else None)
            except json.JSONDecodeError:
                return r.status, body[:300]
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, (json.loads(body) if body else None)
        except json.JSONDecodeError:
            return e.code, body[:300]
    except urllib.error.URLError as e:
        return None, {"network_error": str(e.reason)}


_results: list[tuple[str, str, str]] = []  # (label, status, detail)


def _report(label: str, status: str, detail: str = "") -> None:
    """status: PASS / FAIL / PENDING(정상적인 배포 전 미존재) / SKIP"""
    _results.append((label, status, detail))
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


# ── ① /health ────────────────────────────────────────────────────────────
def check_health(server: str) -> None:
    status, body = _http(server, "/health")
    if status == 200 and isinstance(body, dict) and body.get("status") == "healthy":
        _report(
            "① /health",
            "PASS",
            f"engine={body.get('engine')} gpu_available={body.get('gpu_available')} version={body.get('version')}",
        )
    else:
        _report("① /health", "FAIL", f"status={status} body={body}")


# ── ② GET /api/limits/{video_id} ────────────────────────────────────────
def check_limits(server: str, video_id: str) -> tuple[int | None, dict | None]:
    """②를 검사하면서 (status, body)를 **그대로 반환**한다 — ④가 이 응답 하나로 "이 배포가
    api_key를 요구하는가"까지 같이 판별한다(별도 GET을 또 안 보낸다, main.py 미들웨어가
    server.api_key 설정 여부로 /api 전체에 일괄 적용하므로 아무 /api GET이나 대표값이다)."""
    status, body = _http(server, f"/api/limits/{video_id}")
    if status == 404:
        _report("② /api/limits/{video_id}", "PENDING", "404 — 아직 배포 전(라우터 없음, 정상)")
        return status, body if isinstance(body, dict) else None
    if status != 200 or not isinstance(body, dict):
        _report("② /api/limits/{video_id}", "FAIL", f"status={status} body={body}")
        return status, body if isinstance(body, dict) else None
    required = {"enforced", "generate", "link", "upgrade", "destructive", "window_hours"}
    missing = required - set(body.keys())
    if missing:
        _report("② /api/limits/{video_id}", "FAIL", f"스키마 필드 누락: {missing}")
        return status, body
    gen_limit = (body.get("generate") or {}).get("limit")
    up_limit = (body.get("upgrade") or {}).get("limit")
    independent = gen_limit is not None and up_limit is not None and gen_limit != up_limit
    if not independent:
        _report(
            "② /api/limits/{video_id}",
            "FAIL",
            f"upgrade가 generate와 독립적이지 않음(같은 limit 값) — generate.limit={gen_limit} upgrade.limit={up_limit}",
        )
        return status, body
    _report(
        "② /api/limits/{video_id}",
        "PASS",
        f"스키마 OK, enforced={body.get('enforced')}, generate.limit={gen_limit} upgrade.limit={up_limit}(독립 확인)",
    )
    return status, body


# ── ③ GET /api/notices ──────────────────────────────────────────────────
def check_notices(server: str) -> None:
    status, body = _http(server, "/api/notices")
    if status == 404:
        _report("③ /api/notices", "PENDING", "404 — 아직 배포 전(라우터 없음, 정상)")
        return
    if status != 200 or not isinstance(body, dict) or "notices" not in body:
        _report("③ /api/notices", "FAIL", f"status={status} body={body}")
        return
    notices = body.get("notices") or []
    if not notices:
        # 공지가 0건이면 개별 아이템 스키마를 못 본다 — 200 자체는 라우터 생존 증거지만
        # translations 컬럼 마이그레이션 반영까지 확인하려면 공지가 하나는 있어야 한다.
        _report("③ /api/notices", "PASS", "200 OK, 공지 0건이라 translations 필드 유무는 못 봄(라우터는 살아 있음)")
        return
    has_field = "translations" in notices[0]
    if has_field:
        _report(
            "③ /api/notices",
            "PASS",
            f"translations 필드 존재(값={notices[0].get('translations')!r}) — 컬럼 마이그레이션 반영 확인",
        )
    else:
        _report("③ /api/notices", "FAIL", "notices[0]에 translations 키 자체가 없음 — 구버전 스키마로 응답 중")


# ── ④ X-API-Key 우회 차단 ───────────────────────────────────────────────
def _deployment_requires_key(limits_status: int | None) -> str:
    """"required" / "open" / "unknown" — ②(GET /api/limits, 키 없이 호출)의 상태코드만으로
    판별한다(추가 요청 없음).

    2026-08-04 오탐 사고: 처음엔 "키 없는 쓰기 호출이 401이 아니면 우회"로만 판정했다가
    프로드에서 FAIL을 냈는데, team-lead 확인 결과 실은 이 배포가 **애초에 api_key를 안 켠
    공개 배포**(대신 admin_api_key로 쿼터만 강제)였다 — be5f155가 고친 결함은 "api_key가
    **켜진** 배포에서 admin_api_key 미설정 시 None이 허용값이 되는" 경로라 공개 배포에는
    처음부터 해당이 안 된다. main.py 미들웨어가 server.api_key 설정 여부로 /api 전체에
    키 요구를 일괄 적용하므로, 키 없이 보낸 GET 하나(여기서는 ②의 응답 재사용)의
    상태코드가 그대로 "이 배포가 키를 요구하는가"의 답이다: 401이면 요구, 200이면 공개.
    404/네트워크 오류 등은 이 특정 라우트 문제일 뿐 키 요구 여부와 무관해 "unknown"이다."""
    if limits_status == 401:
        return "required"
    if limits_status == 200:
        return "open"
    return "unknown"


def check_api_key_bypass(server: str, limits_status: int | None, limits_body: dict | None, do_write_probe: bool) -> None:
    """④ — be5f155(«api_key 켜짐 + admin_api_key 미설정 시 키 없는 요청이 통과») 회귀 감시.

    이 결함은 **api_key가 켜진 배포에서만** 성립하므로, 먼저 이 배포가 키를 요구하는지부터
    가린다(_deployment_requires_key). 공개 배포(api_key 미설정)면 애초에 해당 없는 검사라
    SKIP — 대신 지금 쿼터(admin_api_key)로 통제 중인지를 같이 찍어 운영 상태를 사람이
    읽을 수 있게 한다.

    키 요구 배포로 판별됐어도 **기본은 실제 쓰기 호출을 안 보낸다**(SKIP, "--check-auth로
    명시" 안내) — 검증 스크립트를 돌릴 때마다 조용히 실제 잡을 만드는 건 옳지 않다(설령
    존재하지 않는 video_id를 써서 곧 실패로 끝나더라도, 그 판단은 사람이 명시적으로 켜야
    한다). --check-auth가 있을 때만 실제로 보호된 쓰기 액션(POST /api/sync/generate)을
    키 없이 호출해 401을 확인한다 — 그래도 200/202가 오면(우회가 살아 있다는 뜻) 심각한
    결함이므로 여기서 스크립트를 즉시 중단하고 크게 경고한다(job_id가 있으면 같이 찍는다
    — 실존하지 않는 FAKE_VIDEO_ID를 쓰므로 다운로드 단계에서 곧 실패하지만, 확인·정리는
    사람 몫이다)."""
    requirement = _deployment_requires_key(limits_status)
    if requirement == "open":
        enforced = (limits_body or {}).get("enforced")
        _report(
            "④ X-API-Key 우회 차단",
            "SKIP",
            f"공개 배포(api_key 미설정): 키 검사 항목 해당 없음. 쿼터 enforced={enforced}로 통제 중"
            + ("(admin_api_key 설정됨)" if enforced else "(admin_api_key도 미설정 — 완전 무제한)"),
        )
        return
    if requirement == "unknown":
        _report("④ X-API-Key 우회 차단", "SKIP", f"이 배포가 키를 요구하는지 판별 불가(②의 status={limits_status}) — 수동 확인 필요")
        return
    # requirement == "required"
    if not do_write_probe:
        _report(
            "④ X-API-Key 우회 차단",
            "SKIP",
            "키 요구 배포로 판별됨 — 실제 회귀 검사(쓰기 호출)는 --check-auth 플래그가 있어야 실행됨(기본은 판별까지만)",
        )
        return
    status, body = _http(
        server, "/api/sync/generate", method="POST",
        payload={"video_id": FAKE_VIDEO_ID, "lyrics": FAKE_LYRICS},
        timeout=30.0,
    )
    if status == 401:
        _report("④ X-API-Key 우회 차단", "PASS", "키 요구 배포에서 키 없이 호출 → 401(정상 차단)")
        return
    if status == 404:
        _report("④ X-API-Key 우회 차단", "FAIL", "status=404 — /api/sync/generate 라우트를 못 찾음(경로 확인 필요)")
        return
    print("\n" + "!" * 70)
    print("!! 경고: 키 요구 배포인데 X-API-Key 없이 보낸 요청이 401이 아니다 — 인증 우회가 살아있을 수 있다 !!")
    print(f"!! status={status} body={body}")
    if isinstance(body, dict) and body.get("job_id"):
        print(f"!! job_id={body['job_id']} — 실제 잡이 생성됐을 수 있다. 서버에서 직접 확인·취소 필요.")
    print("!" * 70 + "\n")
    _report("④ X-API-Key 우회 차단", "FAIL", f"401이 아님(status={status}) — 위 경고 참고, 즉시 확인 필요")
    _summarize_and_exit(server, aborted=True)


# ── ⑤ en cmudict / zh pypinyin 의존성(간접 확인) ────────────────────────
def _pron_coverage(server: str, video_id: str, label: str) -> None:
    """이미 있는 싱크를 읽어 pron 딕셔너리(hangul/romaji/kana/en/ipa)의 채움 정도를 본다.

    왜 이 검사가 필요한가 — 실사고 재현: en 발음은 cmudict(외부 사전 패키지), zh 발음은
    pypinyin이 있어야 나온다. 둘 다 requirements에 **선언이 빠져 있어도** import가
    지연 로드(en_g2p.py의 _load, zh_reading.py)라 서버는 정상 기동하고 다른 곡들도 잘
    돌아간다 — en/zh 곡을 실제로 만들 때만 조용히 pron이 비거나 kana 단독 근사로
    떨어진다. 로컬 개발 환경엔 두 패키지가 우연히 깔려 있어 로컬 테스트는 계속
    통과하고 있었다(그래서 배포 서버에서 따로 실측해야 의미가 있다).

    한계(주석에 명시): 이 API는 "지금 이 서버가 새로 만들면 어떻게 나오는가"를 직접
    묻지 못한다(그런 엔드포인트가 없다) — 기존 싱크가 **과거 어느 시점**에 만들어졌는지
    모르므로, kana만 있고 hangul/romaji/en/ipa가 없는 것이 "이 서버가 지금 cmudict/
    pypinyin을 못 쓴다"는 확증은 아니고 "오래된 세대 데이터"일 수도 있다(worker.py 주석
    "구세대 라틴 곡의 kana 단독 근사" 참고). 그래도 완전히 비어 있거나(pron 자체가 없음)
    최근 생성분에서도 kana만 나온다면 강한 신호이므로 FAIL로 보고한다."""
    status, body = _http(server, f"/api/sync/{video_id}")
    if status != 200 or not isinstance(body, dict) or not body.get("found"):
        _report(f"⑤ 발음 의존성({label})", "SKIP", f"이 서버에 {video_id} 싱크가 없음(status={status})")
        return
    segs = body.get("timestamps")
    if isinstance(segs, dict):
        segs = segs.get("segments")
    segs = segs or []
    if not segs:
        _report(f"⑤ 발음 의존성({label})", "SKIP", "세그먼트가 비어 있음")
        return
    key_counts = {"hangul": 0, "romaji": 0, "kana": 0, "en": 0, "ipa": 0}
    with_any_pron = 0
    for s in segs:
        pron = s.get("pron") or {}
        if pron:
            with_any_pron += 1
        for k in key_counts:
            if pron.get(k):
                key_counts[k] += 1
    total = len(segs)
    # romaji(zh는 pinyin이 이 키에 실린다) 또는 en·ipa·hangul 중 하나라도 있으면
    # cmudict/pypinyin 경로가 실제로 동작한 것 — kana 단독은 구세대 근사와 구분 불가하니
    # 아래에서 별도 언급만 하고 FAIL로 단정하지 않는다.
    rich = key_counts["hangul"] + key_counts["romaji"] + key_counts["en"] + key_counts["ipa"]
    detail = f"segments={total} pron있음={with_any_pron} 키별={key_counts}"
    if rich == 0 and key_counts["kana"] > 0:
        _report(
            f"⑤ 발음 의존성({label})",
            "FAIL",
            detail + " — kana 단독 근사만 있고 hangul/romaji/en/ipa가 전무함. "
            "cmudict/pypinyin 미동작 또는 구세대 데이터(과거 생성분)일 수 있음 — "
            "확실히 하려면 배포 후 이 언어로 새 싱크를 하나 만들어 직접 확인 권장.",
        )
    elif with_any_pron == 0:
        _report(f"⑤ 발음 의존성({label})", "FAIL", detail + " — pron 필드 자체가 전무함")
    else:
        _report(f"⑤ 발음 의존성({label})", "PASS", detail)


def check_pronunciation_dependency(server: str, en_video_id: str, zh_video_id: str | None) -> None:
    _pron_coverage(server, en_video_id, "en/cmudict")
    if zh_video_id:
        _pron_coverage(server, zh_video_id, "zh/pypinyin")
    else:
        _report("⑤ 발음 의존성(zh/pypinyin)", "SKIP", "EVERYRIC_ZH_VIDEO_ID 미지정 — 실측 zh 싱크 video_id를 넣어야 함")


# ── ⑥ vocaro 엔드포인트 생존 ────────────────────────────────────────────
def check_vocaro(server: str) -> None:
    status, body = _http(server, "/api/vocaro/match?title=deploy-gate-check")
    if status == 200:
        _report("⑥ /api/vocaro/match", "PASS", f"200 OK, found={isinstance(body, dict) and body.get('found')}")
    else:
        _report("⑥ /api/vocaro/match", "FAIL", f"status={status} body={body}")


def _summarize_and_exit(server: str, *, aborted: bool = False) -> None:
    print("\n" + "=" * 70)
    print(f"배포 검증 결과 — 서버: {server}")
    print("=" * 70)
    for label, status, detail in _results:
        print(f"  [{status:7s}] {label}")
    fails = [r for r in _results if r[1] == "FAIL"]
    pendings = [r for r in _results if r[1] == "PENDING"]
    skips = [r for r in _results if r[1] == "SKIP"]
    print("-" * 70)
    print(f"PASS={sum(1 for r in _results if r[1] == 'PASS')} FAIL={len(fails)} PENDING(배포전)={len(pendings)} SKIP={len(skips)}")
    if aborted:
        print("** ④에서 심각한 이상 신호로 조기 중단 — 위 경고를 최우선으로 확인하라 **")
    elif fails:
        print("** FAIL 항목 있음 — 배포 후 재실행해 그래도 FAIL이면 실결함으로 본다 **")
    elif pendings:
        print("** 배포 전 상태로 보임(PENDING 항목 있음) — 배포 후 동일 스크립트로 재확인할 것 **")
    elif skips:
        print("** 전부 PASS(SKIP 항목은 이 배포 설정상 해당 없음 — 위 사유 참고) **")
    else:
        print("** 전부 PASS **")
    sys.exit(1 if (fails or aborted) else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server", nargs="?", default=DEFAULT_SERVER, help="대상 서버 URL(기본: 프로드)")
    parser.add_argument("--en-video-id", default=None, help="⑤용 en 실싱크 video_id(기본: 환경변수 또는 내장 기본값)")
    parser.add_argument("--zh-video-id", default=None, help="⑤용 zh 실싱크 video_id(기본: 환경변수, 없으면 SKIP)")
    parser.add_argument(
        "--check-auth", action="store_true",
        help="④에서 실제 쓰기 호출(POST /api/sync/generate, 키 없이)까지 시도한다. "
             "이 배포가 api_key를 요구할 때만 의미 있고, 공개 배포면 이 플래그를 줘도 SKIP된다. 기본은 꺼짐.",
    )
    args = parser.parse_args()

    import os

    en_id = args.en_video_id or os.environ.get("EVERYRIC_EN_VIDEO_ID") or DEFAULT_EN_VIDEO_ID
    zh_id = args.zh_video_id or os.environ.get("EVERYRIC_ZH_VIDEO_ID") or None

    print(f"대상 서버: {args.server}\n")
    check_health(args.server)
    limits_status, limits_body = check_limits(args.server, en_id)
    check_notices(args.server)
    # SKIP은 중단 사유가 아니다 — ⑤·⑥까지 항상 마저 돈다(2026-08-04 수정). 진짜 우회가
    # 확인된 경우(아래 함수 안에서)만 여기서 프로세스가 조기 종료된다.
    check_api_key_bypass(args.server, limits_status, limits_body, do_write_probe=args.check_auth)
    check_pronunciation_dependency(args.server, en_id, zh_id)
    check_vocaro(args.server)
    _summarize_and_exit(args.server)


if __name__ == "__main__":
    main()
