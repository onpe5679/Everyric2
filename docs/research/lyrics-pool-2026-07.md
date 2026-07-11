# 가사 자동 가져오기 소스 풀 확장 리서치

조사 범위: 코드 읽기(everyric2-chrome, everyric2 서버) + 웹 조사(나무위키 실제 접근 시도 포함).
결론만 필요하면 맨 아래 "권고 로드맵"만 봐도 됨.

---

## 1. 현재 구조

### 1.1 클라이언트 폴백 체인 (`everyric2-chrome/src/background.ts:135` `fetchLyricsChain`)

```
1) 서버 싱크 조회 (lookupSync) — 항상 최우선, 단어 타이밍 보존
2) settings.lyricsSourcePriority === 'vocaro' 면 여기서 스킵하고 콘텐츠 스크립트가
   VOCARO_LOOKUP → (미스 시) FETCH_LRCLIB 순서로 재시도 (content.ts:465-495)
   'lrclib' 면 fetchLyricsChain 안에서 바로 LRCLIB(fetchFromLrclib) 시도
```

- 보카로 위키 조회는 2단 폴백: 클라이언트 `vocaro.ts`(슬러그 추측 → 초성/영문 인덱스 매칭, 한국어 독음 제목 기준)가 먼저 시도되고, 실패하면 서버 `vocaro_index.py`(원제/일본어 인덱스, `VOCARO_LOOKUP` → `vocaroMatch`)로 다시 시도한다. 유튜브 영상 제목이 일본어 원제인 경우 클라 인덱스(한국어 독음 기준)로는 못 찾기 때문.
- 서버 싱크(everyric)로 이미 생성된 곡이라도, 위키의 발음/사람번역을 텍스트 매칭으로 병합하는 `enrichFromVocaro`가 별도로 존재 (content.ts:497-500).
- 싱크 생성 시(`handleGenerate`, content.ts:659) 위키 소스면 `lineMeta`(줄별 발음/번역)와 `attribution`(출처명+URL)을 서버에 함께 저장해 다른 사용자도 그대로 받는다.

### 1.2 데이터 계약 (신규 소스 추가 시 맞춰야 하는 인터페이스)

- `VocaroResult` / `VocaroLine` (`lib/vocaro.ts:13-25`) — 사실상 "위키형 소스"의 표준 반환 형태:
  ```ts
  interface VocaroLine { text: string; pronunciation?: string; translation?: string }
  interface VocaroResult { pageUrl: string; pageTitle: string; slug: string; lines: VocaroLine[] }
  ```
- `LyricsData.source: 'everyric' | 'lrclib' | 'vocaro'` (`types.ts:62`) — 소스 종류가 유니온 타입이라 신규 소스는 여기 값 추가 필요.
- `LineMeta { text, pronunciation?, translation? }` — 싱크 생성 시 서버에 영구 저장되는 줄별 메타(`types.ts:144`).
- `SourceAttribution { name, url? }` — 푸터에 표시되는 출처 표기(`types.ts:112`).
- `SearchCandidate` 유니온(`types.ts:281`) — 수동 검색 후보 리스트, 현재 `lrclib` | `vocaro` 두 분기뿐.
- `humanTranslated` 플래그는 **소스 단위**로만 존재(`lines.some(l => l.translation)`) — 줄 단위로 사람/LLM 출처가 섞이는 걸 구분 못 함. 소스를 늘릴수록 이 한계가 커진다(5장 설계 메모 참고).

**신규 소스 추가 시 건드릴 파일**: `src/lib/<source>.ts`(vocaro.ts 패턴 복제) → `types.ts`(LyricsSource, SearchCandidate 유니온 확장) → `background.ts`(fetchLyricsChain/searchCandidates에 훅, BgRequest 메시지 타입 추가) → `content.ts`(adoptVocaroResult 같은 어댑터, 우선순위 UX) → `settings.ts`+옵션 UI(`lyricsSourcePriority` 확장). 원제→슬러그 매칭이 필요한 소스(유튜브 일본어 제목 대응)는 서버에 `vocaro_index.py`류 인덱스도 필요.

### 1.3 보카로 위키 인덱스 구축 (`everyric2/server/vocaro_index.py`)

42개 "수록곡 일람" 인덱스 페이지(한글 초성 14 + 영문 26 + 숫자/기호 2)를 순회해 슬러그+한국어 제목을 모으고, **신규 슬러그만** 곡 페이지를 6-way 동시 fetch해 title-cell(일본어 원제)을 채우는 증분 방식. JSON으로 원자적 저장, 매칭은 (1) 정규화 정확일치 → (2) 상호 포함 + 길이비 ≥0.5 포함매칭 2단계. 이 구조 자체가 "새 위키 소스 인덱스"를 만들 때 재사용 가능한 템플릿이다.

---

## 2. 나무위키 조사

### 2.1 가사 표 구조 — 실증 확인 실패, 정황 근거로 추정

이번 세션에서 나무위키 문서 3건(천본앵, Vocaloid Lyrics Wiki 확인용 크로스체크 포함)을 WebFetch로 직접 열람 시도했으나 **전부 HTTP 403**으로 막혔다. 즉 실제 표 마크업(HTML class, 행 개수)은 이번 조사로 확인하지 못했다.

정황 근거(검색 스니펫 + 국내 팬덤 관례)로 보면, VOCALOID/JPOP/애니 원곡 문서는 "가사" 문단에 원문(일본어)/독음(한글)/해석(번역) 3행 1세트 표를 병기하는 게 일반적이다(vocaro 위키와 사실상 같은 국내 팬 편집 관례이며, 실제로 나무위키 기여자가 vocaro 위키 등에서 내용을 가져오는 경우도 흔함). 다만:
- **문서마다 편집자 재량**이라 vocaro 위키만큼 포맷이 표준화돼 있지 않다 — 독음을 생략한 문서, 표 대신 산문으로 나열한 문서가 섞여 있을 수 있다.
- 이 구조를 100% 확정하려면 실제 페이지 HTML을 봐야 하는데, 아래 2.2 사유로 그 자체가 난관이다.

### 2.2 크롤링 난이도 — 매우 높음 (핵심 리스크)

- 나무위키는 Cloudflare **Enterprise CDN + 봇 탐지/차단**을 적용 중(나무위키 자체 문서 "봇 탐지 및 차단 솔루션"에 명시).
- 커뮤니티 보고(루리웹: "나무위키는 크롤링 진짜 빡세게 막아놓는 것 같네", 관련 GitHub 이슈들: "클라우드플레어 장벽에 막혀서 실패", "대부분 서버 블랙리스트에 올라가 캡챠로 차단")에 따르면 **User-Agent 스푸핑 정도로는 뚫리지 않는다**.
- 이번 세션 WebFetch도 동일 문서에 2회 모두 403 — 최소한 비-브라우저 UA/클라우드 IP 기준으로는 즉시 차단됨을 재확인.
- 우회하려면 헤드리스 브라우저 + 안티디텍션(undetected-chromedriver, FlareSolverr 프록시 등) 같은 무거운 인프라가 필요하고, 이는 robots.txt/ToS 위반 소지와 IP 밴 리스크를 동반한다. 42개 인덱스 페이지 + 수천 곡 페이지를 도는 `vocaro_index.py` 같은 **전수 인덱스 방식은 사실상 불가능**(즉시 밴 가능성 높음).

### 2.3 라이선스

CC BY-NC-SA 2.0 KR (저작자표시-비영리-**동일조건변경허락**). 사용자 프로젝트가 이미 비상업 전제(MMS-1B-all이 CC-BY-NC)라 "비영리" 자체는 문제 없지만, vocaro 위키(CC BY 4.0, SA 조항 없음)와 달리 **동일조건변경허락(SA)** 조항이 있어 나무위키발 콘텐츠를 조합한 산출물을 배포하면 같은 라이선스 조건을 승계해야 하는 제약이 추가로 걸린다는 점은 유의해야 한다.

### 2.4 비공식 대안: DB 덤프

나무위키는 2020~2021년 시점 전체 JSON 덤프가 HuggingFace(`heegyu/namuwiki`, `heegyu/namuwiki-extracted`)/GitLab(`beomi/namuwiki-dump`)/Archive.org에 공개돼 있다. 라이브 크롤링 없이 오프라인으로 가사 표를 파싱해 원문/독음/번역 사전을 구축하는 게 이론상 가능하지만, **덤프가 5년 가까이 정체**돼 있어 최근 발매곡(2022년 이후)은 커버 못 한다. 최신곡 보강을 위해선 결국 라이브 접근이 필요해 2.2의 문제로 되돌아간다.

### 2.5 제목 매칭 문제

나무위키 URL은 한국어 문서명(예: `/w/천본앵(VOCALOID 오리지널 곡)`)이라 vocaro.ts처럼 "ASCII면 슬러그 추측" 전략이 통하지 않는다. 반드시 자체 검색(`/Search?q=`, 이 역시 Cloudflare 뒤) 또는 사전 구축 제목→URL 인덱스가 필요하고, "(동음이의어)" 괄호 표기·"VOCALOID 오리지널 곡" 접미사 같은 문서명 규칙까지 고려한 후보 생성 로직이 vocaro_index.py 수준으로 필요하다.

**결론**: 데이터 품질(독음+번역+원문이 국내 관례로 이미 정리돼 있음)은 매력적이지만, 접근성 리스크가 조사한 소스 중 가장 크다. 우선순위 낮음, 강행 시 설계는 5장 참고.

---

## 3. 기타 후보 소스 평가

| 소스 | 원문 | 발음 | 번역 | 라이선스/ToS | 접근 난이도 |
|---|---|---|---|---|---|
| LRCLIB (기 통합) | O (싱크 포함) | X | X | 커뮤니티 오픈 데이터, 인증 불필요 | 매우 낮음 |
| Vocaloid Lyrics Wiki (Miraheze) | O | 로마자만 | O(영어) | 위키 CC BY-SA 4.0 + 번역자별 개별 라이선스 | 낮음 (표준 MediaWiki API) |
| LyricsTranslate.com | O | 로마자 (Transliteration 섹션) | O(한국어 포함 다국어) | 곡/번역자별 상이, 불투명 | 중간 (비공식 HTML 파싱) |
| Genius API | 미제공(공식 API) | X | X | ToS 명시적 스크레이핑 금지 | 배제 |
| Musixmatch API | O(유료) | X | 유료 | 무료=30% 프리뷰, 상업 라이선스 필요 | 배제(개인 규모 부적합) |
| uta-net.com / j-lyric.net | O(30만+곡) | X | X | 명시적 크롤링 정책 불명 | 중간(광고 모달 등 잡음) |
| 나무위키 | O | 한글 독음(추정) | O(한국어) | CC BY-NC-SA 2.0 KR (SA 승계 의무) | 매우 높음(Cloudflare) |
| Naver/Daum 블로그·DC 갤러리 | O | 한글 독음(존재는 함) | O | 포스트별 제각각, 표기 없음 | 구조화 불가 |

세부:

- **LyricsTranslate.com**: 이번 조사에서 Senbonzakura 한국어 번역 페이지를 실제로 열어 확인 — 원문/번역이 2-column 또는 row-by-row 뷰로 나란히 표시되고, **"Transliteration" 섹션에 로마자 발음이 별도로 존재**(버전 2개 확인). 페이지 하단에 `© 2008-2026 LyricsTranslate.com`, Copyrights/Privacy Policy 링크, 번역자 프로필(제출일·기여도)까지 명시. 커버리지가 보카로/JPOP에 국한되지 않고 전세계 대중음악으로 훨씬 넓다는 게 최대 장점. 단점은 (1) 로마자 발음이지 한글 독음이 아니라 이 프로젝트가 기대하는 `pronunciation` 필드와 스크립트가 다름(LLM 변환 한 단계 필요), (2) 번역 저작권이 "번역자 개인 소유, 사이트 밖 재사용은 원칙적으로 원저작권자 허가 필요"(포럼/FAQ 검색 결과 종합)라 vocaro/Miraheze보다 법적으로 불투명하고, MusixMatch 라이선스 곡은 그 라이선스가 LyricsTranslate 안에서만 유효하다는 제약도 있음. (3) 공식 API가 없어 URL 슬러그가 자유형이라 자체 검색에 의존.

- **Vocaloid Lyrics Wiki (Miraheze, vocaloidlyrics.miraheze.org)**: 이번 조사에서 페이지 직접 fetch는 3회 모두 403이었으나(WebFetch 툴의 UA 문제로 추정 — Miraheze/MediaWiki 계열은 Cloudflare 엔터프라이즈급 봇 차단을 쓰는 사이트가 아니고, 표준 MediaWiki `action=parse` API(`/w/api.php`)는 수많은 봇·덤프 도구가 실사용 중인 공개 인터페이스라 나무위키와는 리스크 성격이 다르다), 검색 결과로 구조는 확인됨: 12만 4천+ 문서, 보카로/우타이테 전문, 원문(가나)+로마자+영어 번역 3열이 표준 관례. 라이선스는 위키 base가 CC BY-SA 4.0이지만, `Module:TranslatorLicense/data`에 번역자별 개별 제약(permission, non_commercial, no_reprints, no_retranslation 등)이 걸려 있어 **문서별 라이선스 모듈까지 함께 파싱해 필터링**해야 안전하다. 한글 독음은 없음(영어 위키), 보카로 한정이라 커버리지 확장(비-보카로 J-POP/애니) 목적으로는 나무위키/LyricsTranslate에 못 미침.

- **Genius API**: 공식 API는 가사 텍스트 자체를 반환하지 않고(메타데이터/곡 URL만), 실제 가사를 얻으려면 페이지 스크레이핑이 필요한데 이는 ToS에서 "data mining, robots, scraping or similar" 를 명시적으로 금지. 법적 리스크가 가장 뚜렷해 배제 권고.

- **Musixmatch API**: 무료 티어는 가사 30% 프리뷰만 제공(개발/테스트용), 전체·싱크 가사와 번역은 상업 라이선스 계약이 필요. 개인 프로젝트 규모에 맞지 않아 배제.

- **uta-net.com / j-lyric.net**: 일본어 원문 커버리지는 압도적(uta-net 30만곡+)이나 번역/독음이 아예 없어 "원문 확장" 용도로만 유효하고, 나머지는 전적으로 LLM 폴백에 의존. 크롤링 ToS가 명확히 공개돼 있지 않고, j-lyric은 광고 모달 때문에 자동화가 까다롭다는 보고가 있음(GitHub 이슈 스니펫). 발음/번역 부가가치가 없어 우선순위 최하위권.

- **한국 팬 블로그(Naver/Daum/DC 등)**: 사용자가 말한 "발음+번역 병기" 포맷이 실제로 여기저기 존재함을 확인(예: 이번 조사에서 Daum 카페 "크메르의 세계" — 전혀 무관한 커뮤니티에도 센본자쿠라 독음+번역 게시물이 있었음). 다만 사이트/포맷이 전혀 표준화돼 있지 않고, 네이버/다음에는 공식 검색 API가 없어 포털 검색 결과 자체를 스크레이핑해야 하는 문제, 저작권 표시가 거의 없어 라이선스 판단도 어려움. **구조화된 자동 소스로는 부적합** — 대신 "이 곡은 못 찾았는데 블로그 URL을 알고 있다"는 사용자를 위한 온디맨드 단건 파서(사용자가 URL을 주면 그 페이지 하나만 파싱) 정도가 현실적 활용법.

---

## 4. 권고 로드맵 (우선순위)

1. **[즉시, 리스크 낮음] Vocaloid Lyrics Wiki(Miraheze)를 vocaro 다음 2차 위키 소스로 추가.** 표준 MediaWiki API라 Cloudflare류 차단 리스크가 없고, vocaro 위키가 놓친 최신곡/우타이테 커버를 상호보완한다. 원문+영어번역+로마자가 확보되므로 번역은 그대로 쓰거나 한국어로 재번역(NIM/Gemini)하고, 로마자는 "완전 생성"이 아니라 "로마자를 한글 독음으로 변환"하는 LLM 프롬프트를 쓰면 환각 위험이 낮다. 단, 문서별 `TranslatorLicense`를 확인해 `no_reprints` 등이 걸린 번역은 제외하는 필터가 필요.
2. **[중기, 리스크 중간] LyricsTranslate.com을 3~4순위 폴백으로 추가.** 보카로/JPOP 밖(팝, 외국어 커버 등)까지 커버리지를 넓히는 사실상 유일한 후보. 공식 API가 없어 HTML 파싱 + 자체 검색 의존이고 번역자별 라이선스가 불투명하니, 상업적 재배포 없이 개인 캐시 용도로만 시도할 것을 권고. 로마자(Transliteration)가 있어 Miraheze와 동일하게 LLM 한글 변환 파이프라인을 재사용할 수 있다.
3. **[장기/보류] 나무위키.** 데이터 매력은 최상(독음+번역+원문이 국내 관례로 이미 정리돼 있고 신곡 반영도 빠름)이지만 Cloudflare가 조사한 소스 중 가장 큰 장벽. 강행한다면: (a) 헤드리스 브라우저 기반 별도 fetch 서비스를 두고, (b) 트래픽을 "사용자가 명시적으로 선택한 곡 1건" 온디맨드로 극도 제한(전수 인덱스 시도 금지 — 밴 확정적), (c) SA 라이선스 승계 의무를 지킬 것. 1·2로 커버리지가 부족할 때만 재검토 권고.
4. **[배제]** Genius API(ToS 명시 금지 + 가사 미제공), Musixmatch(유료/상업 전제). uta-net/j-lyric은 원문만 있고 발음/번역 부가가치가 없어 "원문 최후 폴백" 이상의 우선순위는 주지 않음.
5. **[UX로 흡수]** Naver/Daum/DC 등 한국 팬 블로그는 구조화 크롤링 대상이 아니라, 이미 계획된 "가사 수동입력/수동검색 UX 개선"(태스크 #21)에서 사용자가 URL을 붙여넣으면 서버가 그 한 페이지만 파싱하는 온디맨드 파서로 흡수하는 게 유일하게 실용적.

---

## 5. 나무위키 파서 설계 스케치 (2와 3순위로 커버리지가 부족해질 경우를 대비한 골격만)

- **문서 검색**: 슬러그 추측 불가 → `/Search?q=<제목>` 결과에서 후보 추출(이 자체도 헤드리스 필요) 또는 vocaro_index.py처럼 사전에 낮은 빈도로 축적한 제목→URL 인덱스 사용.
- **가사 표 추출**: `[include(틀:가사)]` 류 템플릿이 렌더링된 `<table>`을 vocaro.ts의 `parseSongPage`와 같은 원리로 파싱하되, "3행 1세트"를 그대로 가정하면 위험 — 헤더 텍스트("원문"/"가사", "독음"/"읽는 법", "해석"/"번역" 등 동의어 사전)로 행 종류를 판별하는 방어적 분류가 필요(나무위키는 vocaro보다 포맷 표준화가 약함).
- **매핑**: 추출 결과를 `VocaroLine`과 동일한 `{text, pronunciation?, translation?}` 계약으로 맞추면 `LyricsData`/`LineMeta` 파이프라인을 거의 그대로 재사용 가능 — `LyricsSource`에 값 하나만 추가하면 나머지 로직 변경은 최소화된다.
- **LLM 폴백과의 역할 분담**: 위키류 소스에서 원문만 확보되고 독음·번역이 비어있는 라인은 기존 `translator.py`(NIM/Gemini)로 채우는 게 맞다. 다만 현재 `humanTranslated`는 **소스 단위** 플래그(`lines.some(l => l.translation)`)라, 사람이 단 라인과 LLM이 채운 라인이 한 곡 안에 섞이면 구분이 안 된다. 소스를 늘릴수록 이 문제가 커지므로, 장기적으로는 `LyricLine`에 줄 단위 출처 플래그(`pronunciationSource: 'human'|'llm'`, `translationSource: 'human'|'llm'`)를 추가하는 걸 권장한다.
