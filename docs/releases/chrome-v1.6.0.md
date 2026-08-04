# Everyric Chrome 확장 1.6.0

**[한국어](#한국어)** | **[English](#english)** | **[日本語](#日本語)**

이 릴리스는 1.5.5(7/28, 스토어 심사로 오래 묶여 있었음) 사용자 기준 "그 사이 새로 생긴
모든 것"을 다룬다. 그래서 신설(NEW) 기능을 맨 앞에 독립 항목으로 두고, 기존 기능이
크게 좋아진 것과 자잘한 수정은 뒤로 모았다 — 신설이 "개선" 톤에 묻히지 않게.

---

## 한국어

### 스토어 changelog (웹스토어 "변경사항" 칸용)

- **커버 잇기** — 원곡 싱크를 커버·인스트 영상에 그대로 연결해서 볼 수 있어요
- PIP 창에서 이제 검색·설정·다른 영상 연결까지 메인 화면과 똑같이 다 돼요 (크로마키
  스트리밍 모드 포함)
- 영상 위 자막 표시, 다음 영상 정보, 별점·오류 제보, 공지사항·내 기여·남은 횟수 페이지 신설
- 가라오케 레인을 메인 창에서도 켤 수 있고, 음절 표시·발음 타이밍 정확도가 크게 좋아졌어요
- 분석 깊이 올리기 버튼, 디버그 A/B 비교 등 더 정밀하게 다듬을 수 있는 도구 추가
- 재생목록 전용 패널, 진행률·남은 시간(ETA) 표시가 더 정직해졌어요
- 커버 영상·동명이곡·롱숏버전 인식 정확도 개선
- 발음 표시, 재생목록, 안정성·보안 버그 다수 수정

### 신설

- **커버 잇기(다른 영상 싱크 연결)**: 검색 화면 하단에서 원곡 영상 주소를 넣으면, 커버·
  인스트 영상에 원곡 싱크를 그대로 연결해 씁니다.
- **PIP에서 메인과 완전히 같은 기능**: 검색·설정·오프셋 조정·다른 영상 연결까지 PIP 창
  안에서 전부 됩니다. 재생목록 같은 모듈은 창마다 따로 켜고 끌 수 있습니다. **크로마키
  스트리밍 모드**(배경 제거 + 글자 외곽선)로 방송 화면에도 바로 쓸 수 있습니다.
- **가라오케 레인, 메인 창에서도**: 이전엔 PIP 전용이던 피아노롤 레인을 메인 화면에서도
  모듈로 켤 수 있습니다. 좌측 컬럼 배치, 퀵 토글, 타이밍 안내 배너가 함께 생겼습니다.
- **영상 위 자막 표시 + 다음 영상 정보**: 유튜브 자막처럼 영상 위에 가사(+발음·번역)를
  띄우는 모듈과, 다음 재생될 영상 정보를 보여주는 모듈이 새로 생겼습니다.
- **분석 깊이 올리기 버튼**: 결과의 분석 깊이(빠름/보통/정밀) 배지를 확인하고, 버튼
  하나로 더 정밀한 재분석을 요청할 수 있습니다.
- **별점 · 오류 제보**: 별 하나 클릭으로 정렬 품질을 평가하고, 필요하면 별도 버튼으로
  자세한 오류를 제보할 수 있습니다.
- **공지사항 · 내 기여 · 남은 횟수 페이지**: 헤더 아이콘으로 공지사항(다국어 지원)과
  "내가 만든 곡" 기여 이력을 확인하고, 오늘 싱크 생성·초기화·커버 잇기·정렬 업그레이드
  각각 몇 번 남았는지 볼 수 있습니다(정렬 업그레이드는 싱크 생성과 완전히 별도 예산으로
  관리됩니다).
- **재생목록 부착 패널**: 전체 재생목록·이전/다음 곡·현재 곡 강조·서버에 싱크가 있는지
  배지까지 한 패널에서 봅니다.
- **디버그 패널 개편**: 엔진 버전·분석 깊이를 한눈에 보고, 이전 결과와 지금 결과를
  겹쳐 보는 A/B 비교, 서버에 남은 이전 세대와의 버전 비교까지 가능합니다.
- **매칭된 가사 제목 표시 + 즉시 제보**: 자동으로 찾은 가사가 어느 곡인지 상단에
  보여주고, 잘못 매칭됐으면 "이 가사가 아니에요"로 바로 제보합니다.
- 그 외 신설: 설정 화면 범주별 정리 + 검색, 원문/번역이 줄마다 교차된 형식의 수동입력
  가사 인식, 노래하는 구간에 가사창이 은은하게 빛나는 효과.

### 크게 좋아진 것

- **가라오케 정확도**: 음절이 사라지던 문제를 고쳐 부착률이 87.9%에서 97.4%로 올랐고,
  발음 이중표시가 원문보다 빠르거나 늦게 채워지던 오차를 대폭 줄였습니다(실측 49곡 중
  어긋나던 16%가 0%로). 조용한 구간에서 카운트다운이 잘못 뜨는 오탐도 28.3% 줄었습니다.
- **진행 표시 · 남은 시간(ETA)**: 남은 시간이 초 단위로 실시간으로 줄어들고, 분석 깊이가
  자동으로 올라갈 때도 바로 정확한 값으로 갱신됩니다. 실제로 하지 않는 "보컬 분리" 단계를
  표시하던 것도 없앴습니다.
- **곡 인식 정확도**: 커버 영상·게임/개인 채널에 올라온 보카로 곡 오검출을 줄였고,
  동명이곡·롱/숏 버전 오채택, "AC/DC"처럼 곡명 일부가 다른 단어에 잘못 걸리던 오탐을
  고쳤습니다.

### 고친 것

- **발음**: IPA 표기 복원, 영어 곡 발음 중복 표시 제거, 일부 곡 위키 번역 반복 구간 누락,
  생성 직후 번역이 안 뜨던 것, 붙여넣기로 만든 뒤 "번역이 안 붙었다"고 잘못 초기화를
  권하던 것, 일부 배포 환경에서 영어·중국어 발음 표기가 아무 안내 없이 빠지던 것.
- **재생목록**: 다음 영상 카드가 안 눌리던 것, "재생목록에 속하지 않아요" 오탐, 생성
  완료 직후 배지가 낡은 채로 남던 것.
- **초기화**: "재생성" 버튼을 "초기화"로 이름을 바꾸고, 이제 타이밍 오프셋·번역까지
  완전히 함께 초기화됩니다.
- 검색·설정을 열 때 화면 중간부터 보이던 것 — 이제 항상 맨 위부터 보입니다.
- **스트리밍 모드**: 크로마키가 가사 목록에는 안 먹히던 것, 지금 부르는 줄만 외곽선이
  사라지던 것, 외곽선 크기가 화면 배율과 안 맞던 것을 고쳤고, PiP 닫기 확인 화면에서
  "항상 메인 창 유지"를 바로 켤 수 있게 했습니다.
- **안정성**: 확장 리로드 뒤 고아 탭이 멈추는 문제, 애니메이션 루프가 멈추는 무반응 2건,
  로케일 파일 문제로 확장 자체가 로드 거부되던 회귀를 고쳤습니다.
- **보안**: 인증 키를 아예 안 보낸 요청이 통과되던 우회를 차단했습니다(X-API-Key).

### 참고

아래는 접속 중인 서버(everyric.moref.co 또는 자체 호스팅)가 구버전이면 그 기능만
조용히 빠질 뿐 다른 기능은 정상 동작합니다 — 진행 단계 정직화, 남은 횟수(쿼터) 표시
(커버 잇기·정렬 업그레이드 포함), 재생목록 존재 배지 즉시 갱신, 동명이곡·롱/숏 버전
오채택 방지, 공지사항 다국어 표시, 영어·중국어 발음 표기.

---

## English

### Store changelog (for the Web Store "What's new" field)

- **Link a cover to its original** — connect a cover/instrumental video to the original
  song's sync and watch it there directly
- PiP now does everything the main panel does — search, settings, linking another video's
  sync (plus a chroma-key streaming mode)
- New: on-video captions, next-up video info, star-rating & issue reports, and a
  Notices/Contributions/remaining-quota page
- The karaoke lane can now run in the main window too, and syllable/pronunciation timing
  accuracy improved substantially
- New tools for fine-tuning: a depth-upgrade button and a debug-panel A/B comparison
- A dedicated playlist panel, and more honest progress/ETA display
- Better recognition for cover videos, same-title songs, and long/short versions
- Numerous fixes to pronunciation display, playlist behavior, stability, and security

### New

- **Link a cover to its original (link another video's sync)**: enter the original video's
  URL at the bottom of the search screen to connect a cover or instrumental video straight
  to the original song's sync.
- **PiP now does everything the main panel does**: search, settings, offset adjustment, and
  linking another video's sync all work inside the PiP window. Modules like the playlist can
  be turned on/off **per window**. A new **chroma-key streaming mode** (background removal +
  text outline) makes it usable directly in broadcast layouts.
- **Karaoke lane, now in the main window too**: the piano-roll lane that used to be
  PiP-only can now run as a module in the main panel — with a left-side column, quick
  toggles, and a timing-guide banner.
- **On-video captions + next-up video info**: two new display modules — captions (with
  pronunciation/translation) overlaid on the video like YouTube's own captions, and a
  module showing what's playing next.
- **Depth-upgrade button**: see the analysis-depth badge (fast/medium/heavy) on a result
  and request a more thorough re-analysis with one click.
- **Star rating & issue reports**: rate alignment quality with a single star click, or use
  a separate button to report a detailed issue.
- **Notices, Contributions, and remaining-quota pages**: check in-app notices (translated
  into your language) and your "songs I made" history from header icons, and see how many
  sync generations, resets, cover-links, and depth upgrades you have left today (depth
  upgrades now run on a budget completely separate from sync generation).
- **Playlist attachment panel**: the full playlist, previous/next songs, current-song
  highlighting, and server sync-exists badges, all in one panel.
- **Debug panel overhaul**: see the engine version and analysis depth at a glance, compare
  the current result against a previous one with an A/B ghost overlay, and compare against
  older generations still stored on the server.
- **Matched-lyrics title display + instant report**: the header now shows which song the
  auto-matched lyrics came from, and a "these aren't the right lyrics" button reports it
  immediately.
- Also new: categorized/searchable settings, a manual-paste lyrics parser that handles
  original/translation lines interleaved, and a soft glow on the lyrics panel during sung
  sections.

### Significantly improved

- **Karaoke accuracy**: fixed dropped syllables, raising note-attachment coverage from
  87.9% to 97.4%, and greatly reduced how far the dual pronunciation display could fill
  ahead of or behind the original line (measured across 49 songs: 16% of lines were off,
  now 0%). False-positive countdowns during quiet sections dropped 28.3%.
- **Progress display & ETA**: the remaining time now counts down live, second by second,
  and updates correctly the instant the analysis depth changes mid-job. Also removed a
  stage label that claimed work (vocal separation) was happening when it wasn't.
- **Song-detection accuracy**: reduced false detections for cover videos and Vocaloid songs
  on gaming/personal channels, and fixed mismatches from same-title songs, long/short
  versions, and substring false matches (e.g. "AC/DC" matching "AC").

### Fixed

- **Pronunciation**: restored the IPA option, removed duplicate pronunciation lines for
  English songs, fixed missing repeated lines in some songs' wiki translations, fixed
  translations not appearing right after generation completes, fixed a false "translation
  didn't attach" prompt nudging users to reset after a manual paste, and fixed English/
  Chinese pronunciation silently going missing on some deployments.
- **Playlist**: fixed the next-video card not being clickable, a false "not part of a
  playlist" message, and a stale sync-exists badge right after generation finished.
- **Reset**: renamed "Regenerate" to "Reset," which now also fully clears the timing offset
  and translations.
- Fixed search/settings opening scrolled to the middle of the page instead of the top.
- **Streaming mode**: fixed chroma key not applying to the lyrics-list column, the
  currently-sung line losing its text outline, and the outline size not scaling with the
  display; also added a quick "always keep the main window" toggle on the PiP-close
  confirmation screen.
- **Stability**: fixed two freeze bugs (an orphaned tab after extension reload, and a dead
  animation loop), and a regression where a broken locale file caused the extension to fail
  to load entirely.
- **Security**: closed a bypass that let requests through without an API key
  (X-API-Key).

### Note

The items below degrade gracefully on an older server — the feature is simply absent,
nothing else breaks: honest progress-stage labels, the remaining-quota display (incl.
cover-linking/depth-upgrade), instant playlist-badge refresh, same-title/long-vs-short
version disambiguation, translated notices, and English/Chinese pronunciation.

---

## 日本語

### ストア changelog(ウェブストアの「変更点」欄用)

- **カバーをリンク** — 原曲の同期をカバー・インスト動画にそのまま接続して見られます
- PiPでも検索・設定・別動画の同期リンクまでメイン画面と同じように全部できるように
  (配信向けクロマキーモード込み)
- 動画上の字幕表示、次の動画情報、星評価・不具合報告、お知らせ・貢献・残り回数ページを新設
- カラオケレーンがメインウィンドウでも使えるようになり、音節表示・発音タイミングの精度が
  大幅改善
- 解析深度アップグレードボタン、デバッグA/B比較など、より精密に調整できるツールを追加
- 再生リスト専用パネル、進行状況・残り時間(ETA)表示がより正直に
- カバー動画・同名異曲・ロング/ショート版の認識精度を改善
- 発音表示・再生リスト・安定性・セキュリティの不具合を多数修正

### 新機能

- **カバーをリンク(別動画の同期リンク)**: 検索画面下部で原曲動画のURLを入力すると、
  カバー・インスト動画に原曲の同期をそのまま接続して使えます。
- **PiPがメインパネルと完全に同じ機能に**: 検索・設定・オフセット調整・別動画の同期
  リンクまで、すべてPiPウィンドウ内でできます。再生リストなどのモジュールは**ウィンドウ
  ごとに個別に**オン/オフできます。新設の**クロマキー配信モード**(背景除去+文字の
  縁取り)で配信レイアウトにそのまま組み込めます。
- **カラオケレーンがメインウィンドウでも**: これまでPiP専用だったピアノロール式レーンを
  メイン画面でもモジュールとして表示できます。左側カラム配置・クイックトグル・タイミング
  案内バナーも追加されました。
- **動画上の字幕表示 + 次の動画情報**: YouTube自体の字幕のように動画の上に歌詞(+発音・
  翻訳)を重ねて表示するモジュールと、次に再生される動画の情報を表示するモジュールが
  新設されました。
- **解析深度アップグレードボタン**: 結果の解析深度(fast/medium/heavy)バッジを確認し、
  ボタン一つでより精密な再解析をリクエストできます。
- **星評価・不具合報告**: 星をワンクリックで同期品質を評価でき、別ボタンから詳細な
  不具合を報告できます。
- **お知らせ・貢献・残り回数ページ**: ヘッダーアイコンからお知らせ(多言語対応)と
  「自分が作った曲」の貢献履歴を確認でき、今日の同期生成・リセット・カバーリンク・
  深度アップグレードそれぞれの残り回数を確認できます(深度アップグレードは同期生成とは
  完全に別予算で管理されます)。
- **再生リスト添付パネル**: 再生リスト全体・前後の曲・現在の曲のハイライト・サーバー
  同期存在バッジを1つのパネルで確認できます。
- **デバッグパネル刷新**: エンジンバージョン・解析深度を一目で確認でき、現在の結果と
  以前の結果を重ねて比較するA/B比較、サーバーに残る過去世代とのバージョン比較も
  できます。
- **マッチした歌詞タイトル表示 + 即時報告**: 自動でマッチした歌詞がどの曲のものかを
  上部に表示し、間違っていれば「この歌詞ではありません」でその場で報告できます。
- その他新機能: 設定画面のカテゴリー別整理+検索、原文/翻訳が行ごとに交互になっている
  形式の手動貼り付け歌詞認識、歌っている区間で歌詞パネルがほのかに光る演出。

### 大幅に改善

- **カラオケ精度**: 音節が欠落する問題を修正し、ノート付着率が87.9%から97.4%に向上、
  発音の二重表示が原文より先または遅れて塗られるずれを大幅に縮小しました(実測49曲中
  ずれていた16%が0%に)。静かな区間での誤検出カウントダウンも28.3%減少しました。
- **進行状況表示・残り時間(ETA)**: 残り時間が秒単位でリアルタイムにカウントダウンし、
  解析深度が処理中に切り替わった直後でも正確な値にすぐ更新されます。実際には行っていない
  「ボーカル分離」を進行段階として表示していた問題も解消しました。
- **曲検出精度**: カバー動画やゲーム/個人チャンネルに投稿されたボカロ曲の誤検出を減らし、
  同名異曲・ロング/ショート版の誤選択、"AC/DC"が"AC"に誤マッチするような部分一致の
  誤検出を修正しました。

### 修正

- **発音**: IPA表記オプションを復元、英語曲での原文重複発音行を削除、一部楽曲でウィキ
  翻訳の繰り返し部分が欠落する問題、生成完了直後に翻訳が表示されない問題、手動貼り付け後に
  「翻訳が付きませんでした」と誤ってリセットを促していた問題、一部のサーバー環境で英語・
  中国語の発音表記が何の案内もなく欠けていた問題を修正しました。
- **再生リスト**: 次の動画カードがクリックできない問題、「再生リストに属していません」の
  誤表示、生成完了直後にバッジが古いまま残る問題を修正しました。
- **リセット**: 「再生成」ボタンを「リセット」に改名し、タイミングオフセットと翻訳も
  完全にリセットされるようになりました。
- 検索・設定を開いたときに画面の途中から表示されていた問題を修正し、常に先頭から
  表示されるようになりました。
- **配信モード**: クロマキーが歌詞リスト列に効かなかった問題、いま歌っている行だけ
  文字の縁取りが消えていた問題、縁取りのサイズが表示倍率と合っていなかった問題を修正し、
  PiPを閉じる確認画面で「常にメインウィンドウを維持」をワンタップでオンにできるように
  しました。
- **安定性**: 拡張機能リロード後の孤立タブの無反応、アニメーションループ停止による
  無反応という2件のフリーズバグと、ロケールファイルの不備で拡張機能自体が読み込み
  拒否されるリグレッションを修正しました。
- **セキュリティ**: 認証キーを送らないリクエストが通過してしまう抜け道を塞ぎました
  (X-API-Key)。

### 注記

以下の項目は旧サーバーでもその機能が単に表示されないだけで、他の動作には影響しません
— 進行段階の正直な表示、残り回数(クォータ)表示(カバーリンク・深度アップグレード含む)、
再生リストバッジの即時更新、同名異曲・ロング/ショート版の誤選択防止、お知らせの多言語
表示、英語・中国語の発音表記。

---

## 공지사항(인앱 알림) 게시용 초안 — ko/en/ja 3언어, 게시 여부는 별도 확인 필요

`notices` 테이블에 직접 넣지 않았다(실사용자에게 즉시 노출되는 게시 행위라 별도 승인
필요 — 아래 형식만 준비). 2026-08-04 다국어화(`Notice.translations` JSON 컬럼) 반영 —
`title`/`body`(아래 ko)는 기본/폴백 언어, `en`·`ja`는 `POST /api/notices`의 `translations`
필드에 그대로 실어 보내면 된다(`{"en": {"title", "body"}, "ja": {"title", "body"}}`).

- **level**: `info`

### ko (기본 — title/body)

- **title**: `1.6.0 업데이트 — PIP 재작업 · 가라오케 레인 개선 · 쿼터 표시`
- **body**: `메인 창과 PIP가 이제 같은 가사창을 씁니다 — PIP에서도 검색·설정·오프셋 조정이
  그대로 됩니다. 가라오케 레인(음절 표시·발음 타이밍), 재생목록, 번역 표시의 여러 결함을
  고쳤고, 일부 환경에서 영어·중국어 발음 표기가 빠지던 문제도 해결했습니다. 기여 페이지에서
  싱크 생성·초기화·커버 잇기·정렬 업그레이드의 남은 횟수를 각각 확인할 수 있습니다.`

### en (translations.en)

- **title**: `1.6.0 update — PiP rework · karaoke lane improvements · quota display`
- **body**: `The main window and PiP now share the same lyrics panel — search, settings, and
  offset adjustment all work the same way in PiP. We fixed several bugs in the karaoke lane
  (syllable display, pronunciation timing), playlist, and translation display, and fixed
  English/Chinese pronunciation going missing on some setups. The Contributions page now
  shows your remaining quota separately for sync creation, reset, cover-linking, and sort
  upgrades.`

### ja (translations.ja)

- **title**: `1.6.0アップデート — PIP刷新・カラオケレーン改善・クォータ表示`
- **body**: `メインウィンドウとPIPが同じ歌詞パネルを共有するようになりました — PIPでも
  検索・設定・オフセット調整がそのまま使えます。カラオケレーン(音節表示・発音タイミング)、
  再生リスト、翻訳表示の複数の不具合を修正し、一部の環境で英語・中国語の発音表記が
  欠けていた問題も解決しました。貢献ページでは同期作成・リセット・カバー連携・
  並べ替えアップグレードの残り回数をそれぞれ確認できるようになりました。`

---

## 배포 후 검증 체크리스트 (지금 실행 대상 아님 — 서버 프로드 배포 후에만)

- [ ] `GET /api/limits/{video_id}` → 200 + `{enforced, generate, link, upgrade, destructive, window_hours}` 정상 스키마.
- [ ] **X-API-Key 인증 우회 차단 확인**(team-lead 확정, 2026-08-04): 이전 배치 커밋
  `be5f155 fix(security): X-API-Key 부재 요청이 인증을 통과하던 우회 차단`이 대상.
  결함 내용 — 허용 집합이 `(api_key, admin_api_key or None)`이라 admin_api_key
  **미설정** 배포에서 None이 허용값이 돼, 헤더를 아예 안 보낸 요청(provided=None)이
  401 없이 인증을 통과했다(엣지 감사 3.1). 프로드는 admin 키 **강제** 배포(커밋
  959376a 실측: "이 배포는 모든 /api가 401")인데, 그 우회 결함 때문에 지금 구서버는
  키 없는 요청도 통과시키고 있을 수 있다. 수리는 로컬에만 있고 프로드는 아직 이 배치를
  안 받았다. 검증 방법: 프로드에 `X-API-Key` 헤더 **없이** 보호된 쓰기 액션(예:
  `POST /api/sync/generate`)을 호출 → 배포 후에는 **401**이 나와야 한다(배포 전 구서버는
  통과해버릴 수 있다 — 그것이 이 결함의 재현 증거). 정상 키를 실어 보낸 요청은 배포
  전후 동일하게 통과해야 한다(회귀 없음 확인).
