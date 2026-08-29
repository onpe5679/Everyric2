"""개인정보처리방침 페이지 — ``GET /privacy``가 이 HTML을 그대로 돌려준다.

**왜 서버 라우트인가**: 크롬 웹스토어는 방침을 **공개 URL**로 요구하고 그 링크를 스토어
리스팅(설치 *전* 화면)에 표시한다. 확장 내장 페이지(``chrome-extension://``)는 확장 ID에
묶여 외부에서 열 수 없어 심사자도 못 본다. 도메인이 확장의 기본 서버와 같으면 심사자가
대조할 수 있으므로 GitHub 링크보다 이쪽을 택했다.

**왜 파이썬 모듈에 문자열로 두는가**: markdown 라이브러리도 정적 파일 서빙도 이 서버에 없다.
새 의존성이나 패키징 설정(``package-data``)을 들이지 않고 배포를 확실히 하는 가장 단순한 길이
모듈 상수다. 이 프로젝트가 설정 description에 긴 근거를 담는 것과 같은 방식이다.

**정본은 이 파일 하나다.** 한때 ``everyric2-chrome/PRIVACY.md``에 같은 내용을 두려 했는데,
같은 문서를 두 곳에 두면 반드시 갈라진다 — 공개되는 쪽을 정본으로 남겼다.

내용은 **확장 소스를 읽어 확인한 사실만** 담는다. 확인하지 못한 것은 문서 안에 그대로
"확인하지 못했다"고 쓴다 — 방침에 없는 사실을 적으면 그것이 곧 거짓 신고가 된다.
"""

# 마지막 갱신일. 내용을 고치면 **반드시** 함께 올려라 — 방침은 날짜가 사실의 일부다.
PRIVACY_UPDATED = "2026-08-29"

PRIVACY_HTML = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Everyric 개인정보처리방침 · Privacy Policy</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0 auto; padding: 2rem 1.25rem 6rem; max-width: 46rem;
    font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic",
          "Apple SD Gothic Neo", sans-serif;
    color: #1a1a1a; background: #fff;
  }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-top: .75rem;
       border-top: 1px solid #e5e5e5; }
  h3 { font-size: 1rem; margin: 1.75rem 0 .5rem; }
  .updated { color: #666; font-size: .875rem; margin: 0 0 2rem; }
  .lead { background: #f6f8fa; border-left: 3px solid #999; padding: .875rem 1rem;
          margin: 1.5rem 0; border-radius: 0 4px 4px 0; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .9375rem;
          display: block; overflow-x: auto; }
  th, td { border: 1px solid #ddd; padding: .5rem .625rem; text-align: left;
           vertical-align: top; }
  th { background: #f6f8fa; font-weight: 600; white-space: nowrap; }
  code { background: #f0f0f0; padding: .1em .35em; border-radius: 3px;
         font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .875em; }
  ul { padding-left: 1.375rem; }
  li { margin: .375rem 0; }
  hr.lang { border: 0; border-top: 3px double #ccc; margin: 4rem 0 2.5rem; }
  a { color: #0969da; }
  @media (prefers-color-scheme: dark) {
    body { color: #e6e6e6; background: #131313; }
    h2 { border-top-color: #333; }
    .updated { color: #999; }
    .lead { background: #1c1f22; border-left-color: #555; }
    th, td { border-color: #333; }
    th { background: #1c1f22; }
    code { background: #222; }
    hr.lang { border-top-color: #333; }
    a { color: #58a6ff; }
  }
</style>
</head>
<body>

<h1>Everyric 개인정보처리방침</h1>
<p class="updated">마지막 갱신 __UPDATED__ · Chrome 확장 &ldquo;Everyric - Synced Lyrics&rdquo;</p>

<p class="lead">
아래 내용은 확장 소스 코드를 직접 읽고 확인한 사실만 담았습니다.
확인하지 못한 것은 &ldquo;확인하지 못했다&rdquo;고 그대로 적었습니다.
</p>

<h2>1. 이 확장이 하는 일</h2>
<p>
Everyric은 YouTube/YouTube Music에서 재생 중인 곡의 시간 동기화 가사(발음 표기·번역 포함)를
찾아 화면에 보여줍니다. 그 기능을 위해 곡 정보와 가사 텍스트를 서버 및 공개 가사 데이터베이스에
전송합니다. 무엇을 어디로 왜 보내는지 전부 아래에 나열했습니다.
</p>

<h2>2. 전송되는 데이터</h2>

<h3>Everyric 서버 (기본 <code>https://everyric.moref.co</code>, 설정에서 변경 가능)</h3>
<table>
<tr><th>보내는 것</th><th>왜</th></tr>
<tr><td>YouTube 영상 ID</td><td>어떤 영상의 싱크 가사를 조회·생성할지 지정</td></tr>
<tr><td>곡 제목·아티스트명<br>(각 최대 256자·128자로 자름)</td><td>서버가 곡을 식별하고 같은 곡의 기존 싱크를 찾는 데 사용</td></tr>
<tr><td>원본 영상 제목·채널명·제목 후보<br>(위키 매칭 시)</td><td>제목의 곡명/아티스트 방향을 검증하고 오매칭을 피하기 위해 사용</td></tr>
<tr><td><strong>가사 텍스트 전문</strong><br>(자동으로 찾은 것 또는 이용자가 붙여넣은 것)</td><td>음성 정렬로 시간 동기화를 생성할 때 필요. 이 확장의 핵심 기능이 가사와 오디오를 맞추는 것이므로 가사 본문 없이는 동작할 수 없습니다</td></tr>
<tr><td>발음 표기·번역 메타데이터</td><td>가사 위키 등에서 가져온 사람이 쓴 발음·번역을 서버에 함께 저장해 다음 이용자가 재사용할 수 있게</td></tr>
<tr><td>번역할 텍스트와 목표 언어</td><td>&ldquo;가사 번역 표시&rdquo;를 켰을 때만</td></tr>
<tr><td>API 키 (<code>X-API-Key</code> 헤더)</td><td><strong>이용자가 설정에 직접 입력했을 때만</strong> 서버 인증용으로 전송합니다. 입력하지 않으면 헤더 자체를 보내지 않습니다</td></tr>
</table>
<p>
서버 주소는 설정에서 이용자가 직접 구동하는 서버로 바꿀 수 있습니다(자체 호스팅).
바꾸면 위 데이터는 그 주소로 전송됩니다.
</p>
<p>
<strong>확장이 보내지는 않지만 서버가 보게 되는 것 &mdash; 접속 IP.</strong>
위 표는 확장이 <em>담아 보내는</em> 것이고, 그와 별개로 인터넷 요청에는 접속 IP가 따라옵니다
(이 확장뿐 아니라 모든 웹 요청이 그렇습니다). Everyric 서버는 남용을 막기 위해 그 IP로부터
<strong>되돌릴 수 없는 해시</strong>를 만들어 <strong>사용량 제한 단위로만</strong> 씁니다.
</p>
<ul>
<li><strong>IP 원문은 저장하지 않습니다.</strong> 솔트를 넣은 해시만 남깁니다 &mdash; 솔트가
    없으면 IPv4 주소 공간이 좁아 해시에서 원래 주소를 되찾을 수 있기 때문입니다.</li>
<li><strong>1일 한도는 싱크 생성만 셉니다.</strong> 가사 조회는 한도에서 세지 않습니다
    (서버 자원을 거의 쓰지 않기 때문입니다). 다만 <strong>조회 요청도 서버 운영 기록에는
    남습니다</strong> &mdash; 한도에 세지 않는 것과 기록이 없는 것은 다릅니다.</li>
<li>이 값은 사용량 제한에만 쓰이며, 이용자를 식별하거나 이용 내역을 프로필로 모으는 데
    쓰지 않습니다.</li>
<li>설정에 API 키를 입력했다면 그 키가 제한 단위로 대신 쓰입니다.</li>
</ul>
<p>
<strong>CDN 사업자를 거칩니다.</strong> 기본 서버 주소는 Cloudflare를 통해 연결되므로,
확장의 요청은 우리 서버에 닿기 전에 그 사업자를 지나갑니다. 그 과정에서 사업자가 접속 IP를
보게 되며 <strong>자체 로그를 남길 수 있습니다</strong> &mdash; 우리가 통제하지 않는
영역이고, 그쪽 처리 방침은 Cloudflare의 정책을 따릅니다. 자체 호스팅 서버로 바꾸면 이
경로를 거치지 않습니다.
</p>
<p>
이 처리 때문에 <strong>확장이 새로 보내는 데이터는 없습니다</strong> &mdash; 이용자를 구분하기
위해 확장이 식별자를 만들어 심는 방식(설치 ID·기기 지문 등)을 쓰지 않기로 한 결과입니다.
</p>

<h3>LRCLIB (<code>lrclib.net</code>) — 공개 가사 데이터베이스</h3>
<table>
<tr><th>보내는 것</th><th>왜</th></tr>
<tr><td>곡 제목, 아티스트명, 재생시간, 직접 입력한 검색어</td><td>이미 존재하는 시간 동기화 가사를 찾기 위해</td></tr>
</table>

<h3>보카로 가사 위키 (<code>vocaro.wikidot.com</code>)</h3>
<p>
이용자 데이터를 담아 보내지 않습니다. 위키 페이지를 읽어오는 GET 요청만 보내며, 요청에 실리는
것은 확장이 스스로 계산한 페이지 경로뿐입니다.
</p>

<h3>YouTube / YouTube Music 페이지</h3>
<p>
확장이 이 페이지들로 별도 데이터를 보내지 않습니다. 페이지에서 현재 재생 중인 곡의
제목·아티스트·채널명·영상 ID를 <strong>읽기만</strong> 합니다(위 서버 요청에 쓰기 위해).
Everyric 서버는 위키 제목 매칭이 모호하거나 실패했을 때 그 영상 ID로 YouTube의 공개
oEmbed 제목·채널 메타데이터를 조회할 수 있습니다.
YouTube 자체가 페이지 방문으로 수집하는 정보는 이 확장과 무관하며 Google의 정책 영역입니다.
</p>

<h2>3. 전송하지 않는 것</h2>
<p>아래는 코드에서 해당 경로가 없음을 확인한 것입니다.</p>
<ul>
<li><strong>쿠키</strong> — 쿠키를 읽거나 보내는 코드가 확장 어디에도 없습니다.</li>
<li><strong>시청 이력</strong> — 지금 재생 중인 영상 하나에 대한 정보만 그때그때 조회에 씁니다.
    시청 목록을 모아 저장하거나 전송하는 코드가 없습니다.</li>
<li><strong>마이크 오디오</strong> — &ldquo;마이크 음정 표시&rdquo;(기본 꺼짐, 켜야만 브라우저가
    마이크 권한을 묻습니다)를 켜면 마이크 입력을 받지만, 그 오디오는 <strong>브라우저 안에서만</strong>
    실시간 분석되어 음 높이 값만 추출됩니다. 원본 오디오도 추출된 값도 네트워크로 전송하는 코드가
    없고, 최근 몇 초치만 메모리에 두었다가 화면 표시에 쓰고 버립니다.</li>
<li><strong>이름·이메일·주소 등 개인 식별 정보</strong> — 요청하거나 입력받는 곳이 없습니다.</li>
<li><strong>위치 정보</strong> — 요청하는 코드가 없습니다.</li>
</ul>

<h2>4. 이용자가 통제할 수 있는 것</h2>
<p>가사 패널의 설정(⚙)에서 직접 켜고 끌 수 있습니다.</p>
<ul>
<li><strong>서버 주소</strong> — 기본 서버 대신 직접 구동하는 서버를 지정</li>
<li><strong>API 키</strong> — 입력하지 않으면 인증 헤더 자체가 전송되지 않습니다</li>
<li><strong>가사 번역 표시</strong> — 끄면 번역 요청이 나가지 않습니다</li>
<li><strong>발음 표기 표시</strong></li>
<li><strong>마이크 음정 표시</strong> — 기본 꺼짐. 끄면 마이크 스트림을 즉시 정지합니다</li>
<li><strong>자동 가사 검색</strong> — 끄면 재생 시 자동으로 가사를 찾지 않습니다</li>
<li><strong>디버그 정보 표시</strong>, <strong>완료 알림</strong></li>
</ul>
<p>
그리고 <strong>서버에 저장된 자기 데이터를 직접 삭제할 수 있습니다</strong> — 검색 화면의
&ldquo;이 영상 싱크 초기화(서버 저장 삭제)&rdquo;를 누르면 그 영상의 정렬·발음·번역 저장본이
서버에서 삭제됩니다.
</p>

<h2>5. 브라우저에 로컬로 저장되는 것</h2>
<p>
아래는 이용자의 브라우저 안에만 저장되며 그 자체로는 외부로 전송되지 않습니다
(2번 항목의 값들이 요청 시점에 서버로 나가는 것과는 별개입니다).
</p>
<table>
<tr><th>항목</th><th>내용과 용도</th></tr>
<tr><td>설정</td><td>4번의 모든 설정값(서버 주소, API 키 포함). 확장을 다시 열어도 유지되도록</td></tr>
<tr><td>패널 위치·크기</td><td>youtube.com과 music.youtube.com 각각 기억</td></tr>
<tr><td>진행 중 작업</td><td>싱크 생성이 진행 중인 영상과 작업 번호. 여러 탭에서 진행률을 보기 위해</td></tr>
<tr><td>위키 연결·목록 캐시</td><td>영상과 가사 위키 페이지의 연결, 위키 목록 페이지 캐시(24시간)</td></tr>
</table>
<p>
API 키를 포함한 설정값은 <strong>암호화되지 않은 평문으로</strong> 저장됩니다. 브라우저 확장
저장소의 일반적인 동작이며 이 확장만의 특이사항은 아닙니다.
</p>

<h2>6. 보관 기간</h2>
<ul>
<li><strong>브라우저 로컬 저장</strong> — 확장을 제거하거나 브라우저 데이터를 지울 때까지
    유지됩니다(위키 목록 캐시는 24시간 후 자동 갱신).</li>
<li><strong>서버</strong> — 생성된 싱크는 다음 이용자가 같은 곡을 볼 때 재사용하기 위해
    보관됩니다. <strong>자동 삭제 기간은 정해져 있지 않습니다.</strong> 이용자는 4번에 적은
    방법으로 특정 영상의 저장본을 언제든 직접 삭제할 수 있습니다.</li>
<li><strong>LRCLIB</strong> — 이 확장은 검색어만 보냅니다. LRCLIB 자체의 보관 정책은 제3자
    영역이며 이 문서의 범위를 벗어납니다.</li>
</ul>

<h2>7. 가사 저작권</h2>
<p>
<strong>가사의 저작권은 원저작자 및 관련 권리자에게 있습니다.</strong> Everyric은 가사에 대한
어떠한 권리도 주장하지 않으며, 소유하거나 독점하지 않습니다. 이 서비스가 하는 일은 이용자가
보고 있는 곡의 가사를 오디오와 맞추어 표시 시각을 계산하는 것입니다.
</p>
<ul>
<li><strong>출처를 표시합니다</strong> — 가사를 어디서 가져왔는지(공개 가사 데이터베이스, 가사
    위키, 유튜브 자막, 이용자 입력) 화면에 함께 보여줍니다.</li>
<li><strong>저작권자의 요청이 있으면 해당 자료를 즉시 삭제합니다.</strong> 아래 문의 창구로
    연락해 주시면 해당 영상·곡의 저장본을 지웁니다.</li>
<li><strong>이용자도 직접 삭제할 수 있습니다</strong> — 4번에 적은 방법으로 특정 영상의
    저장본을 언제든 지울 수 있습니다.</li>
</ul>

<h2>8. 문의</h2>
<p>
이 확장의 소스 코드는 공개되어 있으며, 문의·삭제 요청·저작권 관련 연락은
<a href="mailto:perion5679@naver.com">perion5679@naver.com</a>으로 받습니다.
</p>

<hr class="lang">

<h1>Everyric Privacy Policy</h1>
<p class="updated">Last updated __UPDATED__ · Chrome extension &ldquo;Everyric - Synced Lyrics&rdquo;</p>

<p class="lead">
Everything below was verified by reading the extension's source code directly.
Anything that could not be verified is stated as such.
</p>

<h2>1. What this extension does</h2>
<p>
Everyric shows time-synced lyrics (with pronunciation and translation) for the song currently
playing on YouTube/YouTube Music. To do that it sends song information and lyrics text to a
backend server and to public lyrics databases. Everything sent is listed below.
</p>

<h2>2. Data that is sent</h2>

<h3>Everyric server (default <code>https://everyric.moref.co</code>, changeable in settings)</h3>
<table>
<tr><th>Sent</th><th>Why</th></tr>
<tr><td>YouTube video ID</td><td>Identifies which video's synced lyrics to fetch or generate</td></tr>
<tr><td>Song title / artist<br>(truncated to 256 / 128 chars)</td><td>Lets the server identify the song and find an existing sync for it</td></tr>
<tr><td>Original video title, channel name and title candidates<br>(during wiki matching)</td><td>Verifies which side of a video title is the song and avoids adopting a different song</td></tr>
<tr><td><strong>Full lyrics text</strong><br>(auto-found or pasted by the user)</td><td>Required to generate the time-sync by audio alignment. The core function of this extension is matching lyrics to audio, so it cannot work without the lyrics body</td></tr>
<tr><td>Pronunciation / translation metadata</td><td>Stores human-written pronunciation and translation (e.g. from a lyrics wiki) alongside the sync so the next user can reuse it</td></tr>
<tr><td>Text to translate and target language</td><td>Only when &ldquo;show translation&rdquo; is enabled</td></tr>
<tr><td>API key (<code>X-API-Key</code> header)</td><td>Sent <strong>only if the user entered one in settings</strong>. Otherwise no such header is sent at all</td></tr>
</table>
<p>
The server address can be pointed at a server the user runs themselves (self-hosting). If changed,
the data above goes to that address instead.
</p>
<p>
<strong>Not sent by the extension, but visible to the server &mdash; the connecting IP.</strong>
The table above lists what the extension <em>puts into</em> its requests. Separately from that,
every internet request carries the connecting IP (true of all web requests, not just this
extension). To prevent abuse, the Everyric server derives a <strong>non-reversible hash</strong>
from that IP and uses it <strong>solely as a rate-limiting key</strong>.
</p>
<ul>
<li><strong>The raw IP is not stored.</strong> Only a salted hash is kept &mdash; without a salt
    the IPv4 address space is small enough to recover the original address from the hash.</li>
<li><strong>The daily cap counts sync generation only.</strong> Lyrics lookups are not counted
    against it (they consume almost no server resources). However, <strong>lookup requests do
    appear in server operational logs</strong> &mdash; not counting toward a cap is not the same
    as leaving no record.</li>
<li>The value is used for rate limiting only. It is not used to identify users or to build a
    profile of what anyone listens to.</li>
<li>If an API key is entered in settings, that key is used as the rate-limiting key instead.</li>
</ul>
<p>
<strong>Requests pass through a CDN provider.</strong> The default server address is served via
Cloudflare, so requests traverse that provider before reaching our server. In doing so the
provider sees the connecting IP and <strong>may keep its own logs</strong> &mdash; that is outside
our control and governed by Cloudflare's own policies. Pointing the extension at a self-hosted
server avoids this path entirely.
</p>
<p>
Because of this design the <strong>extension sends no additional data</strong> &mdash; a
deliberate choice not to mint an identifier (install ID, device fingerprint) to tell users apart.
</p>

<h3>LRCLIB (<code>lrclib.net</code>) — public lyrics database</h3>
<table>
<tr><th>Sent</th><th>Why</th></tr>
<tr><td>Track name, artist name, duration, user-typed search query</td><td>Looking up an existing time-synced lyrics entry</td></tr>
</table>

<h3>Vocaloid lyrics wiki (<code>vocaro.wikidot.com</code>)</h3>
<p>
No user data is sent. Only GET requests to fetch wiki pages; the only thing in the request is a
page path the extension computes itself.
</p>

<h3>YouTube / YouTube Music pages</h3>
<p>
The extension sends no separate data to these pages. It only <strong>reads</strong> the currently
playing song's title, artist, channel name and video ID from the page (to use in the server requests above).
When wiki title matching is ambiguous or fails, the Everyric server may use that video ID to request
public title/channel metadata from YouTube's oEmbed endpoint.
What YouTube itself collects from a page visit is unrelated to this extension and is covered by
Google's policies.
</p>

<h2>3. Data that is NOT sent</h2>
<p>Each item below was verified as having no such code path.</p>
<ul>
<li><strong>Cookies</strong> — no code anywhere in the extension reads or transmits cookies.</li>
<li><strong>Browsing / watch history</strong> — only information about the single video currently
    playing, used for a one-off lookup. No code accumulates or transmits a watch list.</li>
<li><strong>Microphone audio</strong> — the &ldquo;mic pitch display&rdquo; feature (off by
    default; the browser asks for microphone permission only if you turn it on) reads microphone
    input, but the audio is analysed <strong>entirely inside the browser</strong> to extract a
    pitch value. No code transmits the raw audio or the extracted values over the network; only
    the last few seconds are held in memory for on-screen display and then discarded.</li>
<li><strong>Personally identifying information</strong> (name, email, address) — never requested
    or collected.</li>
<li><strong>Location data</strong> — no code requests it.</li>
</ul>

<h2>4. What the user controls</h2>
<p>All of the following can be toggled in the lyrics panel settings (⚙).</p>
<ul>
<li><strong>Server address</strong> — point it at a self-hosted server instead of the default</li>
<li><strong>API key</strong> — leave it empty and no auth header is sent at all</li>
<li><strong>Show lyrics translation</strong> — off means no translation request is made</li>
<li><strong>Show pronunciation</strong></li>
<li><strong>Mic pitch display</strong> — off by default; turning it off stops the mic stream
    immediately</li>
<li><strong>Auto lyrics search</strong> — off means no automatic lookup on playback</li>
<li><strong>Debug info</strong>, <strong>completion notification</strong></li>
</ul>
<p>
Users can also <strong>delete their own stored data on the server</strong> — the search screen's
&ldquo;reset this video's sync (delete server copy)&rdquo; action deletes that video's stored
alignment, pronunciation and translation from the server.
</p>

<h2>5. Stored locally in the browser</h2>
<p>
These stay in the browser and are not transmitted by themselves (separate from the values in
section 2, which are sent to a server at request time).
</p>
<table>
<tr><th>Item</th><th>Content and purpose</th></tr>
<tr><td>Settings</td><td>Everything from section 4, including server address and API key. Persisted across sessions</td></tr>
<tr><td>Panel position / size</td><td>Remembered separately for youtube.com and music.youtube.com</td></tr>
<tr><td>In-progress jobs</td><td>Which videos have a sync being generated, to show progress across tabs</td></tr>
<tr><td>Wiki link and index cache</td><td>Which wiki page is linked to a video; cached wiki index pages (24h)</td></tr>
</table>
<p>
Settings — including the API key, if set — are stored <strong>in plain text</strong>, since
browser extension storage does not encrypt data. This is standard behaviour, not specific to this
extension.
</p>

<h2>6. Retention</h2>
<ul>
<li><strong>Local browser storage</strong> — kept until the extension is removed or browser data
    is cleared (the wiki index cache refreshes after 24 hours).</li>
<li><strong>Server</strong> — a generated sync is kept so the next user viewing the same song can
    reuse it. <strong>There is no fixed automatic deletion period.</strong> Users can delete a
    given video's stored copy at any time using the action described in section 4.</li>
<li><strong>LRCLIB</strong> — this extension only sends search terms. LRCLIB's own retention
    policy is a third-party matter outside the scope of this document.</li>
</ul>

<h2>7. Lyrics copyright</h2>
<p>
<strong>Copyright in the lyrics belongs to the original authors and rights holders.</strong>
Everyric claims no rights in any lyrics and neither owns nor licenses them. What this service does
is compute display timings by matching the lyrics of the song you are listening to against its
audio.
</p>
<ul>
<li><strong>Sources are attributed</strong> — the on-screen panel shows where the lyrics came from
    (public lyrics database, lyrics wiki, YouTube captions, or the user's own paste).</li>
<li><strong>On a rights holder's request, the material is deleted immediately.</strong> Please
    contact us through the channel below and the stored copy for that video/song will be removed.</li>
<li><strong>Users can delete it themselves too</strong> — the action described in section 4 removes
    a given video's stored copy at any time.</li>
</ul>

<h2>8. Contact</h2>
<p>
The source code for this extension is public. For questions, deletion requests, or copyright
matters, please contact
<a href="mailto:perion5679@naver.com">perion5679@naver.com</a>.
</p>

</body>
</html>
""".replace("__UPDATED__", PRIVACY_UPDATED)
