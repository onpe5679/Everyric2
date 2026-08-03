# Everyric Chrome 확장 1.6.0

**[한국어](#한국어)** | **[English](#english)** | **[日本語](#日本語)**

**주의(team-lead 검수 필요)**: 이 파일은 `git log chrome-v1.5.5..HEAD`(48개 커밋)를
주제별로 묶어 정리한 초안이다. 사용자 관점 문구·우선순위는 확정 전 재검토 바람.

## 한국어

- **PIP(팝업 재생) 전면 재작업**: 메인 가사창과 PIP가 이제 완전히 같은 UI 구현을 공유합니다.
- **가라오케 레인 개선**: 음절 소실·발음 표시 타이밍·카운트다운 오탐·다음 줄 미리보기 등
  여러 결함을 고쳤고, 좌측 컬럼·퀵 토글·타이밍 안내 배너가 추가됐습니다.
- **설정 화면 개편**: 범주별로 접었다 펼 수 있게 바뀌었고, 검색으로 원하는 설정을 바로
  찾을 수 있습니다. 별점·오류 제보가 분리됐고, 공지사항·내 기여 페이지가 새로 생겼습니다.
- **재생목록 부착 패널**: 다음 영상 카드 클릭 이동, "재생목록에 속하지 않아요" 오탐 재시도,
  서버 싱크 존재 배지 등을 수리했습니다.
- **진행 표시 정확도**: 분석 깊이(fast/medium/heavy) 배지, 남은 예상 시간(ETA)이 깊이
  전환 시에도 정확한 값으로 갱신됩니다. 실제로 하지 않는 작업(예: 보컬 분리)을 진행
  단계로 잘못 표시하던 문제도 고쳤습니다(서버 업데이트 필요 — 아래 참고).
- **쿼터("남은 횟수") 표시**: 기여 페이지 상단에 싱크 생성·초기화·커버 잇기·정렬 업그레이드
  각각의 남은 횟수를 표시합니다. 무제한 배포에서도 "무제한 · 사용 N회"로 안내합니다
  (서버 업데이트 필요 — 아래 참고).
- **번역·발음 정확도**: 생성 완료 직후 번역이 안 뜨던 문제, 일부 곡의 위키 번역 반복
  구간 누락, IPA 발음 옵션 복원, en 곡 원문 중복 발음 줄 등을 고쳤습니다.
- **곡 인식 개선**: 커버 영상·게임/개인 채널 업로드 보카로 곡의 오검출을 줄였고, 곡명
  부분열이 다른 단어에 잘못 걸리던 오탐(예: "AC/DC"→"AC")을 고쳤습니다.
- **안정성**: 확장 리로드 시 고아 탭 무반응, rAF 루프 정지로 인한 무반응 등 2건의 프리즈
  버그를 고쳤습니다. 로케일 파일의 `placeholders` 블록 누락으로 확장 로드 자체가 거부되던
  회귀도 수복했습니다.
- 참고: 아래 서버 업데이트 필요 항목은 접속 중인 서버(everyric.moref.co 또는 자체
  호스팅)가 구버전이어도 그 기능만 조용히 빠질 뿐 다른 기능은 정상 동작합니다 — 진행
  단계 정직화, 쿼터 표시(커버 잇기·정렬 업그레이드 포함), 재생목록 존재 배지 즉시 갱신,
  vocaro 동명이곡·롱/숏 버전 오채택 방지.

## English

- **PiP overhaul** — the main lyrics panel and the picture-in-picture window now share the exact same UI implementation.
- **Karaoke lane fixes** — fixed dropped syllables, mistimed pronunciation display, false-positive countdowns, and next-line preview; added a left-side column, quick toggles, and a timing-guide banner.
- **Settings redesign** — settings are now grouped into collapsible categories with a search box. Rating and error reports are now separate actions, and new Notices / My Contributions pages were added.
- **Playlist attachment panel** — fixed the next-video card not being clickable, a false "not part of a playlist" message with retry, and server sync-exists badges.
- **More accurate progress display** — the analysis-depth badge (fast/medium/heavy) and the estimated time remaining (ETA) now update correctly right when the depth changes mid-job. Also fixed a stage label that claimed work (e.g. vocal separation) was happening when it wasn't (requires a server update — see below).
- **Quota ("remaining uses") display** — the Contributions page now shows remaining counts for sync generation, reset, cover linking, and depth upgrades at the top. Unlimited deployments now say "Unlimited · used N times" instead of hiding the block (requires a server update — see below).
- **Translation/pronunciation accuracy** — fixed translations not appearing right after generation completes, missing repeated lines in some songs' wiki translations, restored the IPA pronunciation option, and removed duplicate pronunciation lines for English songs.
- **Better song detection** — reduced false detections for cover videos and Vocaloid songs uploaded to gaming/personal channels, and fixed substring false matches corrupting song titles (e.g. "AC/DC" → "AC").
- **Stability** — fixed two freeze bugs (orphaned tabs after extension reload, and a dead requestAnimationFrame loop). Also fixed a regression where a missing `placeholders` block in a locale file caused the extension to fail to load entirely.
- Note: the server-dependent items below degrade gracefully on an older server (the feature is simply absent, nothing breaks) — stage-label accuracy, the quota display (incl. cover-linking/upgrade), instant playlist-badge refresh, and vocaro same-title/long-vs-short-version disambiguation.

## 日本語

- **PiP全面刷新** — メイン歌詞パネルとPiPウィンドウが完全に同じUI実装を共有するようになりました。
- **カラオケレーン改善** — 音節の欠落・発音表示のタイミングずれ・誤検出のカウントダウン・次の行のプレビューなど複数の不具合を修正し、左側カラム・クイックトグル・タイミング案内バナーを追加しました。
- **設定画面の刷新** — カテゴリ別に折りたたみ可能になり、検索で目的の設定にすぐアクセスできます。評価とエラー報告が分離され、お知らせ・貢献履歴ページが新設されました。
- **再生リスト添付パネル** — 次の動画カードがクリックできない問題、「再生リストに属していません」の誤表示、サーバー同期存在バッジを修正しました。
- **進行状況表示の精度向上** — 解析の深さ(fast/medium/heavy)バッジと推定残り時間(ETA)が、深さが切り替わった直後でも正確な値に更新されるようになりました。実際には行っていない作業(例:ボーカル分離)を進行段階として誤表示していた問題も修正しました(効果にはサーバー更新が必要 — 下記参照)。
- **クォータ(「残り回数」)表示** — 貢献ページの上部に、同期生成・リセット・カバー連携・解析深度アップグレードそれぞれの残り回数を表示します。無制限デプロイでもブロックを隠さず「無制限 · 使用N回」と案内します(効果にはサーバー更新が必要 — 下記参照)。
- **翻訳・発音の精度向上** — 生成完了直後に翻訳が表示されない問題、一部楽曲でウィキ翻訳の繰り返し部分が欠落する問題を修正し、IPA発音オプションを復元、英語曲での原文重複発音行を削除しました。
- **曲検出の改善** — カバー動画やゲーム/個人チャンネルに投稿されたボカロ曲の誤検出を減らし、曲名の部分一致が別の単語を壊してしまう誤検出(例:"AC/DC"→"AC")を修正しました。
- **安定性** — 拡張機能リロード時の孤立タブ無反応、rAFループ停止による無反応という2件のフリーズバグを修正しました。ロケールファイルの`placeholders`ブロック欠落で拡張機能自体が読み込み拒否されるリグレッションも修復しました。
- 注記: 以下のサーバー依存項目は旧サーバーでも機能が単に表示されないだけで、他の動作には影響しません — 進行段階の正確な表示、クォータ表示(カバー連携・アップグレード含む)、再生リストバッジの即時更新、vocaro同名曲・ロング/ショート版の誤選択防止。

---

## 공지사항(인앱 알림) 게시용 초안 — 한국어만, 게시 여부는 별도 확인 필요

`notices` 테이블에 직접 넣지 않았다(실사용자에게 즉시 노출되는 게시 행위라 별도 승인
필요 — 아래 형식만 준비).

- **title**: `1.6.0 업데이트 — PIP 재작업 · 가라오케 레인 개선 · 쿼터 표시`
- **body**: `메인 창과 PIP UI가 통합되고, 가라오케 레인·재생목록·번역 표시의 여러 결함을
  고쳤습니다. 기여 페이지에서 남은 생성·초기화 횟수를 확인할 수 있습니다. 자세한 내용은
  확장 업데이트 노트를 참고하세요.`
- **level**: `info`
