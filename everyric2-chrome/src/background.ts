import { fetchFromLrclib, getLrclibById, searchTracksLrclib } from './lib/lrclib';
import { attachLineMeta, cancelJob, checkServerStatus, fetchCaptionLines, fetchLimits, fetchNotices, fetchPreviousSync, fetchSyncVersion, fetchSyncVersions, fetchViewStats, findLinkCandidates, generateSync, generateSyncFromCaption, getJobStatus, getLinkJobStatus, getServerLog, linkSync, listSyncs, lookupSync, regenerateSync, resetSync, saveTranslationLayer, saveUserOffset, submitFeedback, syncExists, translateLyrics, unlinkSync, vocaroMatch, type FailureSink, type ServerConfig } from './lib/everyric-api';
import { parseLRC, parsePlainLyrics, segmentsToLines } from './lib/lyrics-parser';
import { mirahezeLookup } from './lib/miraheze';
import { fetchSongPage, vocaroLookup } from './lib/vocaro';
import { getSettings } from './lib/settings';
import type { BgRequest, ContentMessage, LRCLibTrack, LyricsData, MessageResponse, SearchCandidate, SongInfo, SourceAttribution } from './types';

async function getServerConfig(): Promise<ServerConfig> {
  const { serverUrl, apiKey } = await getSettings();
  return { serverUrl, apiKey };
}

/**
 * SourceAttribution(확장 내부, camelCase)을 서버 요청 바디 모양(snake_case)으로.
 *
 * ``license``는 이름이 같아 그대로 통과하지만 ``sourceId``→``source_id``는 명시적 변환이
 * 필요하다 — everyric-api.ts의 request()가 payload를 그대로 JSON.stringify하고, 서버
 * pydantic Attribution 모델엔 캐멀케이스 별칭이 없어(다른 모델도 전부 그렇다) 낯선 키를
 * 조용히 버린다. generateSync 등의 매개변수 타입은 여전히 SourceAttribution이라(이
 * 함수의 반환값도 그 타입으로 캐스트한다 — everyric-api.ts 타입 변경은 이 작업 범위 밖).
 */
function toWireAttribution(a: SourceAttribution | undefined): SourceAttribution | undefined {
  if (!a) return a;
  const wire: Record<string, unknown> = { name: a.name, url: a.url };
  if (a.license) wire.license = a.license;
  if (a.sourceId) wire.source_id = a.sourceId;
  return wire as unknown as SourceAttribution;
}

/** 서버 응답의 attribution(snake_case dict 그대로 온다)을 확장 내부 표기로 — 위 함수의 역. */
function fromWireAttribution(raw: SourceAttribution | null | undefined): SourceAttribution | undefined {
  if (!raw) return undefined;
  const wireRaw = raw as SourceAttribution & { source_id?: string | null };
  return { name: raw.name, url: raw.url, license: raw.license ?? undefined, sourceId: raw.sourceId ?? wireRaw.source_id ?? undefined };
}

/**
 * 서버 호출 한 건을 응답으로 감싼다 — 실패하면 코드**와 사유**를 함께 실어 보낸다.
 *
 * 예전에는 `res ? { data: res } : { error: '...' }`였다. 그래서 콘텐츠 쪽은 실패한 건
 * 알아도 "왜"는 영영 알 수 없었다. sink를 여기서 만들어 넘기므로 호출부마다
 * 보일러플레이트가 늘지 않는다.
 */
async function call<T>(
  errorCode: string, fn: (sink: FailureSink) => Promise<T | null>,
): Promise<MessageResponse<T>> {
  const sink: FailureSink = {};
  const data = await fn(sink);
  return data === null ? { error: errorCode, failure: sink.failure } : { data };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  handleMessage(message as BgRequest)
    .then(sendResponse)
    .catch((error: unknown) => {
      sendResponse({ error: error instanceof Error ? error.message : String(error) });
    });
  return true;
});

// everyric.com 웹사이트에서 싱크 생성 완료 알림 (기존 플로우 유지)
chrome.runtime.onMessageExternal.addListener((message, sender, sendResponse) => {
  const origin = sender.origin ?? '';
  const isEveryric = origin === 'https://everyric.com' || origin.endsWith('.everyric.com');
  const videoId = (message?.payload as { videoId?: unknown } | undefined)?.videoId;
  if (isEveryric && message?.type === 'SYNC_COMPLETE' && typeof videoId === 'string') {
    // 방금 생겼으니 존재 캐시가 있다면 낡았다 — 지워서 다음 조회가 서버에 다시 묻게 한다
    existsCache.delete(videoId);
    void broadcastToYouTubeTabs({ type: 'SYNC_GENERATED', payload: { videoId } });
    sendResponse({ success: true });
  }
  return true;
});

/**
 * 재생목록 패널의 존재 배지 캐시 — 있음은 30분, 없음은 2분만 믿는다(방금 생성된 싱크를
 * "없음"으로 오래 오해하지 않으려는 비대칭 TTL). 서비스워커가 잠들면 사라지는 휘발성
 * 캐시로 충분하다(다음 조회가 다시 채운다). SYNC_GENERATED 발신 지점에서 해당 videoId를
 * 무효화한다(위 onMessageExternal).
 */
const EXISTS_TTL_HIT_MS = 30 * 60 * 1000;
const EXISTS_TTL_MISS_MS = 2 * 60 * 1000;
const existsCache = new Map<string, { exists: boolean; expiresAt: number }>();

// 툴바 아이콘 클릭 → 해당 탭의 오버레이 토글
chrome.action.onClicked.addListener(tab => {
  if (tab.id !== undefined) {
    chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_OVERLAY' }).catch(() => {
      /* content script가 없는 탭(비 YouTube)은 무시 */
    });
  }
});

/**
 * 핫키(Alt+Shift+D) → 해당 탭의 디버그 정보 표시 토글.
 *
 * 키 선택 근거: 유튜브가 알파벳 단일키를 거의 다 쓴다(k j l f t i m c o w, 0-9, , . < > /,
 * Space, 화살표). 크롬은 Ctrl+Shift+{I J C M T W B O N **D** A}를 예약하므로 Ctrl+Shift+D는
 * 쓸 수 없다(모든 탭 북마크). Alt+Shift 조합은 유튜브·크롬·OS 어디에도 겹치지 않고, 패널은
 * 키 이벤트를 stopPropagation으로 끊으므로(overlay.ts) 패널 안 타이핑과도 충돌하지 않는다.
 *
 * macOS에서는 Alt가 Option으로 매핑돼 Option+Shift+D가 특수문자 입력과 겹칠 수 있다.
 * commands가 먼저 가로채므로 실동작에는 문제가 없을 것으로 보지만 **실측하지 않았다** —
 * 문제가 있으면 사용자가 chrome://extensions/shortcuts에서 재지정할 수 있다.
 */
chrome.commands.onCommand.addListener((command, tab) => {
  if (command !== 'toggle-debug' || tab?.id === undefined) return;
  chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_DEBUG' }).catch(() => {
    /* content script가 없는 탭(비 YouTube)은 무시 */
  });
});

/**
 * 호스트 권한이 허용·철회되면 열려 있는 유튜브 탭들에 알린다.
 *
 * 허용은 **다른 탭**(옵션 페이지)이나 chrome://extensions에서 일어난다. 가사창의 서버
 * 상태는 권한을 근거로 판정되므로(everyric-api.request의 가드), 이 알림이 없으면 허용을
 * 마친 뒤에도 패널에는 "권한이 없어요" 배너가 그대로 남아 사용자가 허용이 실패한 줄 안다.
 * 폴링 대신 이벤트로 닫는다 — 철회도 같은 경로로 즉시 반영된다.
 */
chrome.permissions.onAdded.addListener(() => {
  void broadcastToYouTubeTabs({ type: 'PERMISSIONS_CHANGED' });
});
chrome.permissions.onRemoved.addListener(() => {
  void broadcastToYouTubeTabs({ type: 'PERMISSIONS_CHANGED' });
});

async function handleMessage(message: BgRequest): Promise<MessageResponse> {
  switch (message.type) {
    case 'FETCH_LYRICS': {
      // 여기서 null은 "가사가 없다"일 수도, "서버가 못 줬다"일 수도 있다 — 그 둘을
      // 화면이 구분할 수 있도록 서버 조회 실패 사유를 data와 함께 실어 보낸다
      const sink: FailureSink = {};
      const data = await fetchLyricsChain(message.payload, message.payload.skipLrclib === true, sink);
      return { data, failure: sink.failure };
    }

    case 'FETCH_LRCLIB': {
      const track = await fetchFromLrclib(message.payload);
      return { data: track ? lrclibToLyricsData(track) : null };
    }

    case 'SEARCH_CANDIDATES':
      return { data: await searchCandidates(message.payload) };

    case 'PICK_LRCLIB': {
      const track = await getLrclibById(message.payload.id);
      return { data: track ? lrclibToLyricsData(track) : null };
    }

    case 'GENERATE_SYNC': {
      const server = await getServerConfig();
      return call('generate_request_failed', sink => generateSync(server, {
        video_id: message.payload.videoId,
        lyrics: message.payload.lyrics,
        language: message.payload.language,
        line_meta: message.payload.lineMeta,
        line_meta_pending: message.payload.lineMetaPending,
        attribution: toWireAttribution(message.payload.attribution),
        title: message.payload.title,
        artist: message.payload.artist,
        target_lang: message.payload.targetLang,
        line_meta_lang: message.payload.lineMetaLang,
      }, sink));
    }
    case 'ATTACH_LINE_META': {
      const server = await getServerConfig();
      return call('attach_line_meta_failed', sink => attachLineMeta(server, message.payload.jobId, {
        line_meta: message.payload.lineMeta,
        attribution: toWireAttribution(message.payload.attribution),
        title: message.payload.title,
        artist: message.payload.artist,
        line_meta_lang: message.payload.lineMetaLang,
      }, sink));
    }

    case 'REGENERATE_SYNC': {
      const server = await getServerConfig();
      return call('regenerate_request_failed', sink => regenerateSync(server, {
        video_id: message.payload.videoId,
        lyrics: message.payload.lyrics,
        line_meta: message.payload.lineMeta,
        attribution: toWireAttribution(message.payload.attribution),
        title: message.payload.title,
        artist: message.payload.artist,
        target_lang: message.payload.targetLang,
        line_meta_lang: message.payload.lineMetaLang,
        min_depth: message.payload.minDepth,
      }, sink));
    }

    case 'JOB_STATUS': {
      const server = await getServerConfig();
      return call('job_status_failed', sink => getJobStatus(server, message.payload.jobId, sink));
    }

    case 'JOB_CANCEL': {
      const server = await getServerConfig();
      return call('job_cancel_failed', sink => cancelJob(server, message.payload.jobId, sink));
    }

    case 'NOTIFY': {
      // 전사 잡 종료를 다른 탭/창에 있어도 알 수 있게 OS 알림으로. 같은 id로 만들면
      // 여러 탭이 같은 잡을 폴링해도 알림이 중복되지 않고 갱신된다.
      try {
        chrome.notifications.create(message.payload.id ?? `ey-${Date.now()}`, {
          type: 'basic',
          iconUrl: chrome.runtime.getURL('icons/icon128.png'),
          title: message.payload.title,
          message: message.payload.message,
        });
      } catch { /* 알림 실패는 치명적이지 않다 */ }
      return { data: true };
    }

    case 'TRANSLATE': {
      const server = await getServerConfig();
      return call('translate_failed', sink => translateLyrics(
        server, message.payload.text, message.payload.targetLang,
        {
          title: message.payload.title,
          artist: message.payload.artist,
          persist: message.payload.persist,
          videoId: message.payload.videoId,
        }, sink,
      ));
    }

    case 'SAVE_TRANSLATION_LAYER': {
      const server = await getServerConfig();
      return call('save_translation_layer_failed', sink => saveTranslationLayer(
        server, message.payload.videoId,
        {
          target_lang: message.payload.targetLang,
          lines: message.payload.lines,
          origin: message.payload.origin,
          attribution: toWireAttribution(message.payload.attribution),
        }, sink,
      ));
    }

    case 'SERVER_HEALTH': {
      // 권한이 없으면 checkServerStatus 안의 첫 요청이 'permission'으로 실패하고 그대로
      // ServerStatus.kind === 'permission'이 된다 (everyric-api.request의 가드 →
      // failureToStatus). 그래서 여기에는 권한 분기가 없다 — 판정 경로를 하나로 두면
      // 개별 API 호출의 실패와 헬스체크가 같은 사유를 말한다.
      return { data: await checkServerStatus(await getServerConfig()) };
    }

    // content script는 chrome.permissions를 쓸 수 없고, service worker에서 request()를
    // 부르면 사용자 제스처가 없어 실패한다 — 허용은 확장 페이지에서만 가능하므로 여기서는
    // 그 페이지를 여는 것까지가 전부다.
    case 'OPEN_OPTIONS': {
      try {
        await chrome.runtime.openOptionsPage();
        return { data: true };
      } catch (error) {
        return { error: error instanceof Error ? error.message : 'open_options_failed' };
      }
    }

    // 최근 서버 요청 몇 건 — 패널·PiP의 접이식 로그가 펼쳐질 때만 가져간다.
    // 경로·본문의 키류는 everyric-api가 기록 시점에 이미 마스킹했다.
    case 'SERVER_LOG':
      return { data: getServerLog() };

    /**
     * 위키 곡 페이지 조회 — **서버 원제 인덱스가 1차, 클라이언트 초성 인덱스는 폴백.**
     *
     * 순서를 뒤집었다(예전엔 클라 우선). 클라 인덱스는 한국어 독음 제목만 답할 수 있는데
     * 유튜브 제목은 대개 일본어 원제라, 답할 수 없는 자리에서 포함 매칭이 엉뚱한 곡을
     * 집어 **그 가사를 원곡 가사로** 띄우는 사고가 났다(「我ら！ゴミ分別団」 실측).
     * 서버 인덱스는 원제·한국어 표기를 함께 갖고 다중 후보 가드도 거기 있다 — 정답을 알
     * 가능성이 높은 쪽을 먼저 묻고, 그 뒤에야 좁아진 클라 경로로 내려간다
     * (lib/vocaro.indexPageFor가 이제 지원 못 하는 제목에 null을 준다).
     */
    case 'VOCARO_LOOKUP': {
      const server = await getServerConfig();
      // hint = 정리 전 영상 제목 — 다중 버전 페이지(원곡/리믹스)에서 맞는 표를 고른다
      const hint = message.payload.hint ?? message.payload.title;
      const matched = await vocaroMatch(server, message.payload.title);
      if (matched?.found && matched.slug) {
        const page = await fetchSongPage(server, matched.slug, hint);
        if (page) return { data: page };
      }
      // 서버가 미발견이거나 페이지를 못 읽은 경우에만 클라 경로 — 초성 인덱스가 답할 수
      // 있는 제목(한글·라틴·숫자 시작)에서만 결과가 나온다
      return { data: await vocaroLookup(server, message.payload.title, hint) };
    }

    // 가사 본문 없이 원제 매칭만 — 제목 확인·후보 표시 경로가 페이지 조회 없이 쓴다
    case 'VOCARO_MATCH': {
      const server = await getServerConfig();
      return call('vocaro_match_failed', sink =>
        vocaroMatch(server, message.payload.title, sink, message.payload.hint));
    }

    case 'MIRAHEZE_LOOKUP': {
      return { data: await mirahezeLookup(message.payload.title) };
    }

    case 'SYNC_LINK': {
      const server = await getServerConfig();
      return call('link_failed', sink => linkSync(server, {
        video_id: message.payload.videoId,
        source_video_id: message.payload.sourceVideoId,
        offset_sec: message.payload.offsetSec,
        rate: message.payload.rate,
      }, sink));
    }

    // 커버 자동 연결: 같은 곡 후보 탐색 (서버가 검증 잡까지 자동 제출한다).
    // 이 엔드포인트가 없는 구버전 서버는 404 → error가 되고, content는 조용히 포기한다.
    case 'LINK_CANDIDATES': {
      const server = await getServerConfig();
      return call('link_candidates_unavailable', sink => findLinkCandidates(server, message.payload.videoId, {
        title: message.payload.title,
        artist: message.payload.artist,
      }, sink));
    }

    case 'LINK_JOB_STATUS': {
      const server = await getServerConfig();
      return call('link_job_status_failed', sink =>
        getLinkJobStatus(server, message.payload.linkJobId, sink));
    }

    case 'SYNC_UNLINK': {
      const server = await getServerConfig();
      return call('unlink_failed', sink => unlinkSync(server, message.payload.videoId, sink));
    }

    // 확장 자신의 잡 완료 경로(content.ts pollJobs)가 부른다 — onMessageExternal의
    // SYNC_COMPLETE와 같은 목적(existsCache 무효화)을 내부 채널로 재현한다. 그 핸들러가
    // 웹사이트발 완료만 알아, 확장이 직접 생성한 잡은 최대 EXISTS_TTL_MISS_MS(2분)까지
    // 재생목록 배지가 "없음"으로 낡아 있었다(감사 A3 실측).
    case 'SYNC_CREATED': {
      existsCache.delete(message.payload.videoId);
      return { data: { ok: true } };
    }

    case 'SYNC_RESET': {
      const server = await getServerConfig();
      return call('sync_reset_failed', sink => resetSync(server, message.payload.videoId, sink));
    }

    case 'SYNC_OFFSET': {
      const server = await getServerConfig();
      return call('sync_offset_failed', sink =>
        saveUserOffset(server, message.payload.videoId, message.payload.offsetSec, sink));
    }

    case 'SYNC_PREVIOUS': {
      // 디버그 패널의 A/B 고스트 비교 — 이력이 없으면 서버가 found=false를 준다
      const server = await getServerConfig();
      return call('sync_previous_failed', sink =>
        fetchPreviousSync(server, message.payload.videoId, sink));
    }

    case 'SYNC_VERSIONS': {
      // 디버그 패널의 깊이·버전 비교 — 저장된 버전 목록(최신순 ≤10)
      const server = await getServerConfig();
      return call('sync_versions_failed', sink =>
        fetchSyncVersions(server, message.payload.videoId, sink));
    }

    case 'SYNC_VERSION_GET': {
      // 목록에서 고른 버전 하나의 전체 타임스탬프 — 모르는 id면 서버가 404(→ null)
      const server = await getServerConfig();
      return call('sync_version_get_failed', sink =>
        fetchSyncVersion(server, message.payload.videoId, message.payload.resultId, sink));
    }

    case 'SYNC_FEEDBACK': {
      const server = await getServerConfig();
      return call('sync_feedback_failed', sink => submitFeedback(server, {
        video_id: message.payload.videoId,
        rating: message.payload.rating,
        category: message.payload.category,
        comment: message.payload.comment,
        // 제보 대상 싱크의 분석 깊이 — 없으면(구싱크·라우팅 메타 부재) 키 자체를 안 보낸다
        depth: message.payload.depth,
      }, sink));
    }

    // 공지·한도·조회수 — 셋 다 구버전 서버엔 없다(404 → error). 호출부는 조용히 포기한다.
    case 'NOTICES_GET': {
      const server = await getServerConfig();
      return call('notices_unavailable', sink => fetchNotices(server, sink));
    }

    case 'LIMITS_GET': {
      const server = await getServerConfig();
      return call('limits_unavailable', sink => fetchLimits(server, message.payload.videoId, sink));
    }

    case 'STATS_VIEWS': {
      const server = await getServerConfig();
      return call('stats_views_unavailable', sink =>
        fetchViewStats(server, message.payload.videoIds, sink));
    }

    // 재생목록 패널 존재 배지 — 캐시로 먼저 채우고, 남는 것만 서버에 묻는다.
    // 서버 조회가 실패해도 캐시로 채운 분은 그대로 돌려준다(부분 성공을 error로 뭉개지 않는다).
    case 'SYNC_EXISTS': {
      const ids = [...new Set(message.payload.videoIds)].slice(0, 100);
      const now = Date.now();
      const result: Record<string, boolean> = {};
      const need: string[] = [];
      for (const id of ids) {
        const cached = existsCache.get(id);
        if (cached && cached.expiresAt > now) result[id] = cached.exists;
        else need.push(id);
      }
      if (need.length > 0) {
        const fetched = await syncExists(await getServerConfig(), need);
        if (fetched) {
          for (const id of need) {
            const exists = fetched[id] ?? false;
            existsCache.set(id, { exists, expiresAt: now + (exists ? EXISTS_TTL_HIT_MS : EXISTS_TTL_MISS_MS) });
            result[id] = exists;
          }
        }
        // 조회 실패(구버전 서버·오프라인)는 조용히 넘어간다 — 그 항목만 배지가 안 뜰 뿐,
        // 목록 자체는 계속 동작해야 한다(다른 additive 엔드포인트와 같은 규칙)
      }
      return { data: result };
    }

    case 'SYNC_LIST': {
      // 검색 필터가 생겨 후보를 넉넉히 받는다 — 서버 목록은 최신순.
      // 빈 배열([])과 "서버가 못 줬다"를 화면이 구분할 수 있게 실패 사유를 함께 보낸다
      const sink: FailureSink = {};
      const res = await listSyncs(await getServerConfig(), 200, sink);
      return { data: res ?? [], failure: sink.failure };
    }

    case 'VOCARO_PAGE':
      return {
        data: await fetchSongPage(
          await getServerConfig(), message.payload.slug, message.payload.hint,
        ),
      };

    // 자막 **본문**은 서버(yt-dlp) 경유 — 워치 페이지에서 긁은 timedtext URL은
    // POT(proof-of-origin) 강제로 브라우저 플레이어 밖에선 빈 응답이 온다 (실측).
    // 트랙 **목록**은 content가 워치 페이지에서 직접 읽으므로 서버를 부르지 않는다.
    case 'YT_CAPTION_TEXT': {
      const server = await getServerConfig();
      const res = await call('caption_text_failed', sink => fetchCaptionLines(
        server, message.payload.videoId, message.payload.lang, message.payload.auto, sink,
      ));
      return res.data ? { data: res.data.lines } : { error: res.error, failure: res.failure };
    }

    // 자막을 보고 있다가 누른 '싱크 생성' — 서버가 자막을 직접 읽는 전용 경로.
    // 아직 배포되지 않은 서버면 null이 오고, content가 기존 생성 경로로 폴백한다.
    case 'GENERATE_FROM_CAPTION': {
      const server = await getServerConfig();
      return call('generate_from_caption_unavailable', sink =>
        generateSyncFromCaption(server, message.payload.videoId, sink));
    }

    default:
      return { error: 'unknown_message_type' };
  }
}

// E2E 스모크 테스트가 SW 컨텍스트에서 직접 호출하기 위한 노출 — 프로덕션 동작에는 영향 없음.
// 서버 프록시 전환(1.5.5) 후에도 테스트는 제목 하나만 넘긴다 — 서버 설정은 여기서 해석한다.
(globalThis as { __vocaroLookup?: (title: string) => ReturnType<typeof vocaroLookup> }).__vocaroLookup =
  async title => vocaroLookup(await getServerConfig(), title);

/** 우선순위: Everyric 서버(단어 타이밍 보존) → (skipLrclib가 아니면) LRCLIB 싱크 → LRCLIB 일반 */
async function fetchLyricsChain(
  song: SongInfo & { lang?: string }, skipLrclib = false, sink?: FailureSink,
): Promise<LyricsData | null> {
  // 제목·아티스트를 함께 보낸다 — 서버가 기존 싱크의 빈 제목을 이 기회에 채운다(백필).
  // 재생성 없이 코퍼스에 제목이 쌓이는 유일한 경로이고, 그게 없으면 커버 링크 후보
  // 탐색이 영원히 빈손이다.
  const sync = await lookupSync(
    await getServerConfig(), song.videoId,
    { title: song.title, artist: song.artist, lang: song.lang }, sink,
  );
  // 서버에 저장된 영상별 사용자 오프셋 — 싱크가 없어도(found=false) 내려온다
  const userOffset = sync?.user_offset ?? undefined;
  if (sync?.found && sync.timestamps && sync.timestamps.length > 0) {
    const { lines, translationsByLang } = segmentsToLines(sync.timestamps, sync.translations_by_lang);
    if (lines.length > 0) {
      return {
        source: 'everyric',
        synced: true,
        lines,
        plainText: lines.map(l => l.text).join('\n'),
        // 서버가 사람 번역(위키 병합분)을 내려줬으면 기계번역으로 덮어쓰지 않는다
        humanTranslated: lines.some(l => l.translation),
        // lang 파라미터를 준 조회에만 의미가 있다 — 구서버·미지정이면 undefined로 남아
        // content.ts의 translationLangMatches 가드가 예전 규칙으로 동작한다
        translationLang: sync.translation_lang ?? undefined,
        // 제목바 언어 칩용 — 구서버는 필드 자체가 없어 undefined(칩 숨김)
        availableLangs: sync.available_langs ?? undefined,
        // 세션 언어별 캐시 선채움용(V2 확장) — segmentsToLines가 lines와 같은 순서로
        // 이미 재정렬해 왔다. 구서버는 undefined(기존 폴백 그대로).
        translationsByLang,
        // 엔진 정체(스택 스탬프·변형)는 응답 최상위 필드 — 디버그 표시가 한 곳(debugMeta)만
        // 보면 되도록 여기서 접어 넣는다. **메타가 전무한 구서버 응답에도 합성한다**(코덱스
        // 감사 Med): meta 자체가 없으면 updateDepthButton이 버튼을 통째로 숨겨, 구서버
        // 싱크(=구세대)에서 노란 업그레이드 유도가 사라졌다 — engine_version:null 합성이
        // 곧 "구세대" 신호다.
        debugMeta: {
          ...(sync.debug ?? {}),
          engine_version: sync.engine_version ?? null,
          engine_variant: sync.engine_variant ?? null,
        },
        attribution: fromWireAttribution(sync.attribution),
        // 번역 출처(additive, 서버 동시 배포 중) — 구서버는 필드 자체가 없어 undefined,
        // content.applyLyricsData가 있을 때만 배지에 반영한다(없으면 기존 동작 그대로).
        translationOrigin: sync.translation_origin ?? undefined,
        translationAttribution: fromWireAttribution(sync.translation_attribution),
        tempo: sync.tempo ?? undefined,
        key: sync.key ?? undefined,
        qualityScore: sync.quality_score ?? undefined,
        createdAt: sync.created_at ?? undefined,
        linked: sync.linked
          ? {
              sourceVideoId: sync.linked.source_video_id,
              offsetSec: sync.linked.offset_sec,
              rate: sync.linked.rate ?? undefined,
              // 서버가 안 내려주는 구버전이면 undefined → 화면은 '검증 여부 모름'으로 표시
              verified: sync.linked.verified ?? undefined,
            }
          : undefined,
        userOffset,
      };
    }
  }

  // 보카로 위키 우선 설정이면 LRCLIB은 content 쪽에서 위키 미스 이후에 별도로 시도한다
  if (skipLrclib) return null;
  const track = await fetchFromLrclib(song);
  const data = track ? lrclibToLyricsData(track) : null;
  if (data) data.userOffset = userOffset;
  return data;
}

function lrclibToLyricsData(track: LRCLibTrack): LyricsData | null {
  if (track.syncedLyrics) {
    const lines = parseLRC(track.syncedLyrics);
    if (lines.length > 0) {
      return {
        source: 'lrclib',
        synced: true,
        lines,
        plainText: track.plainLyrics ?? lines.map(l => l.text).join('\n'),
      };
    }
  }
  if (track.plainLyrics) {
    const lines = parsePlainLyrics(track.plainLyrics);
    if (lines.length > 0) {
      return { source: 'lrclib', synced: false, lines, plainText: track.plainLyrics };
    }
  }
  return null;
}

/** 수동 검색 후보: LRCLIB 트랙들 + 보카로 위키 매칭(서버 원제 인덱스 → 클라 독음 인덱스) */
async function searchCandidates(query: { title: string; artist: string; duration: number }): Promise<SearchCandidate[]> {
  const [tracks, wikiMatch] = await Promise.all([
    searchTracksLrclib(query),
    vocaroMatch(await getServerConfig(), query.title),
  ]);

  const candidates: SearchCandidate[] = [];
  if (wikiMatch?.found && wikiMatch.slug) {
    candidates.push({
      source: 'vocaro',
      slug: wikiMatch.slug,
      title: wikiMatch.ja ?? wikiMatch.ko ?? query.title,
      url: wikiMatch.page_url ?? `http://vocaro.wikidot.com/${wikiMatch.slug}`,
    });
  } else {
    // 서버 원제 인덱스가 미스 — 독음 인덱스 매칭으로 한 번 더 (페이지까지 확보되면 그 제목 사용)
    const direct = await vocaroLookup(await getServerConfig(), query.title);
    if (direct) {
      candidates.push({ source: 'vocaro', slug: direct.slug, title: direct.pageTitle, url: direct.pageUrl });
    }
  }
  for (const t of tracks) {
    candidates.push({
      source: 'lrclib',
      id: t.id,
      title: t.trackName ?? '(제목 없음)',
      artist: t.artistName ?? '',
      duration: t.duration ?? 0,
      synced: Boolean(t.syncedLyrics),
    });
  }
  return candidates;
}

async function broadcastToYouTubeTabs(message: ContentMessage): Promise<void> {
  const tabs = await chrome.tabs.query({
    url: ['*://www.youtube.com/*', '*://music.youtube.com/*'],
  });
  for (const tab of tabs) {
    if (tab.id !== undefined) {
      chrome.tabs.sendMessage(tab.id, message).catch(() => {
        /* content script 미주입 탭은 무시 */
      });
    }
  }
}
