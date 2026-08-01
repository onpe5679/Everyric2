# -*- coding: utf-8 -*-
"""뷰어 전체 재생성 — **UST 준정답 레인 인자를 코드에 박아 둔다.**

karaoke_review.py를 ``--all``만으로 돌리면 UST 레인이 통째로 사라진다(인자로만 들어가므로).
그러면 준정답이 없어져 라인·음절 채점이 조용히 죽는다 — 2026-08-01에 실제로 그렇게 날렸고,
오프셋은 사람이 뷰어에서 드래그로 확정한 값이라 다시 만들 수 없다. 그래서 명령줄이 아니라
**소스에 남긴다**. 앞으로 뷰어 재생성은 이 스크립트로 한다.

오프셋 출처
    사람 확정 7곡(뷰어 드래그) + 자동 적합 11곡(텍스트 앵커 합의, 사람 확정 5곡 재현 오차
    20~30ms 검증 통과). 근거는 ``docs/research/2026-07-30-model-replacement/
    ust-precision-comparison.md``의 「UST 준정답 확장 정밀비교」 절.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmark"
USTS = BENCH / "usts"

# (video_id, usts/ 아래 상대 경로, 전역 오프셋 초)
# 한 곡에 여러 줄이면 레인이 여러 개다(듀엣·하모니·병행 레이어).
UST_LANES: tuple[tuple[str, str, float], ...] = (
    # ── 사람 확정 (뷰어 드래그) ────────────────────────────────────────────
    ("b2NTglk9tvI", "熱異常_ust+ustx/netsuijo_main.ust", -0.10),
    ("b2NTglk9tvI", "熱異常_ust+ustx/netsuijo_harm.ust", -0.10),
    ("vjBFftpQxxM", "butcher vanity.ustx", -0.46),
    ("VWVtIg5cdDU", "The Disappearance of Hatsune Miku (Lmocinemod's UST)/_Melody.ust", 26.35),
    ("VWVtIg5cdDU", "The Disappearance of Hatsune Miku (Lmocinemod's UST)/_Fast (Melody).ust", 26.35),
    ("VWVtIg5cdDU", "The Disappearance of Hatsune Miku (Lmocinemod's UST)/_Glitch.ust", 26.35),
    ("zyRt-nBM3dY", "Cynical Night Plan - UST/MAIN - Cynical Night Plan.ust", 0.15),
    ("icBDYkfxpMs", "loopingtherooms_untuned_main.ust", 14.83),
    ("s4kAOHUSvT8", "トーストになっちゃった ustx/トーストになっちゃった！！.ustx", -1.93),
    # numb numb — 제안 δ+0.03을 사용자가 "정확함"으로 확정(2026-07-31), 표본 편입
    ("ba7YbGO2aq4", "numb numb USTX by spayde-173P/ustx/numb numb untuned.ustx", 0.03),
    # ── 자동 적합 (앵커≥0.4 → 레인 합의≤0.2s·IQR≤0.6 → 드리프트≤0.008) ────
    ("2CwBFr-Eoxg", "Deep Sea Girl CV UST/Deep sea girl.ust", -0.14),
    ("7GxXhrePnA0", "Nekomimi Archive UST/NekomimiArchive_main.ust", -0.33),
    ("KgczJh0uX5o", "Pseudo-Hope Syndrome USTs by Levin/main.ust", 0.02),
    ("hUaVxNUCbc4", "TeikokuShojo_ust/TeikokuShojo.ust", -0.95),
    ("ThOViq2fHqg", "[2023-02-03] 君が死んでも許してあげるよ - Kikuo/君が死んでも許してあげるよ main.ust", 5.77),
    ("17PC17BVVdA", "dame ningen da/main.ust", 12.92),
    ("owv06htaoI8", "rookie/rookie.ustx", 0.01),
    ("Ljr2wMSBHqU", "みむかｩわナイストライ by さっぱりあんずジャム/みむかｩわナイストライ.ustx", 6.71),
    ("tZvn-cdHgnc", "花めかない UST/main.ust", 0.07),
    ("2t1NMRse6aI", "Alien Alien UST by Zansatsu/CV/Alien Alien Main.ust", 0.05),
    # 듀엣 — 두 성부 모두 리드라 harmony 제외 규칙이 안 걸린다
    ("ApkajNBHxqo", "君はいなせなガール ust/IA.ust", -0.16),
    ("ApkajNBHxqo", "君はいなせなガール ust/miki.ust", -0.15),
    # 자동 적합이 낸 -0.04는 앵커 0.46 경계값이라 신뢰도가 낮았다. 사용자 청취로 -0.06 확정
    # (2026-08-01) — en에서 UST 채점이 되는 유일한 곡이라 이 값이 곧 채점의 기준선이다.
    ("M7VSEZOQIlg", "weathergirl by mothman.ustx", -0.06),
    # 확인 대기 — 세 파일 전부 초반 10~57s의 병행 레이어(cho=하모니, main2=5s 조각)
    ("qXkkhP0d_iM", "秋の未確認生物_ust/뢌궻뼟둴봃맯븿_ust/뢌궻뼟둴봃맯븿main.ust", 1.26),
    ("qXkkhP0d_iM", "秋の未確認生物_ust/뢌궻뼟둴봃맯븿_ust/뢌궻뼟둴봃맯븿cho.ust", 1.26),
    ("qXkkhP0d_iM", "秋の未確認生物_ust/뢌궻뼟둴봃맯븿_ust/뢌궻뼟둴봃맯븿main2.ust", 1.26),
)

# 부적합 판정으로 뺀 것(되살리지 말 것): ハナタバ(다른 곡의 UST — 사용자 판정) /
# 忘れじの言の葉(드리프트 0.616, eval 오디오가 다른 판본) / Get Your Wish(섹션 4파일에 전역
# 오프셋 부재) / ライラック(커버 오디오, UST 26%만 커버) / 弱虫モンブラン(GBK 이중 모지바케) /
# 飼猫(eval 대응 vid 없음) / magical cure·Kikuo 등 VIEWER_EXCLUDE_SONGS 곡.

# 뷰어 E키로 내보낸 소절별 오프셋 — 레인 전역 오프셋 위에 얹힌다.
UST_SHIFTS: tuple[tuple[str, str], ...] = (("VWVtIg5cdDU", "shifts_VWVtIg5cdDU.json"),)

# 자막 참조 레인(곡의 원본이 아니라 같은 곡 다른 업로드의 자막인 경우가 있다).
CAPTIONS: tuple[tuple[str, str], ...] = (
    ("b2NTglk9tvI", "captions/hDhjRh-Gt4g.ja.srt"),
)


def main() -> int:
    argv = ["karaoke_review.py", "--all"]
    missing: list[str] = []
    for video_id, rel, offset in UST_LANES:
        path = USTS / rel
        if not path.is_file():
            missing.append(f"{video_id}  {rel}")
            continue
        argv += ["--ust", f"{video_id}={path}@{offset}"]
    for video_id, name in UST_SHIFTS:
        path = USTS / name
        if path.is_file():
            argv += ["--ust-shifts", f"{video_id}={path}"]
        else:
            missing.append(f"(shifts)  {name}")
    for video_id, rel in CAPTIONS:
        path = BENCH / rel
        if path.is_file():
            argv += ["--srt", f"{video_id}={path}"]
        else:
            missing.append(f"(srt)  {rel}")

    lanes = sum(1 for a in argv if a == "--ust")
    print(f"UST 레인 {lanes}개 · shifts {sum(1 for a in argv if a == '--ust-shifts')}개 "
          f"· 자막 {sum(1 for a in argv if a == '--srt')}개")
    if missing:
        print("\n★ 파일을 못 찾았다 — 그만큼 준정답이 빠진다:")
        for item in missing:
            print(f"   {item}")
        print()

    sys.path.insert(0, str(REPO))
    from scripts import karaoke_review

    sys.argv = argv
    return karaoke_review.main()


if __name__ == "__main__":
    raise SystemExit(main())
