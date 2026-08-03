import type {
  BgRequest, ContribEntry, LimitBucket, LimitsResponse, LyricLine, MessageResponse, NoticesResponse,
  SearchCandidate, ServerLogEntry, ServerNotice, ServerStatus, SongInfo, ViewStatsResponse,
} from '../types';
import { CONTRIB_STORAGE_KEY } from '../types';
import { t } from '../lib/i18n';
import { describeRemoved, stripPartMarkers } from '../lib/lyrics-clean';
import { formatLogEntry, needsHostPermission, serverBlockedTip, serverKnownBad, serverUsable, statusLine } from '../lib/server-status';
import { h, icon, ICONS } from './dom';

/**
 * 가사창 UI 조각들의 공용 모듈.
 *
 * 메인 패널(LyricsOverlay)과 PiP 창이 **같은 UI를 동시에** 띄울 수 있어야 하는데,
 * 예전에는 이 조각들이 `this.body`/`this.callbacks`/`this.searchResultsEl` 같은
 * LyricsOverlay 인스턴스 필드를 직접 만졌다 — 필드가 하나뿐이라 나중에 그린 쪽이
 * 이전 참조를 덮어써 두 창에 동시에 띄울 수 없었다.
 *
 * 그래서 karaoke.ts의 appendTimedSpans/appendKaraokeSpans와 같은 규약으로 뽑았다:
 * **"콜백과 상태를 받아 엘리먼트를 만들고, 나중에 갱신할 참조만 돌려준다."**
 * 호스트(패널·PiP)는 돌려받은 참조를 자기 필드에 보관한다.
 *
 * h()가 전역 document.createElement를 쓰지만 PiP에서도 안전하다 — 다른 document의
 * 노드를 appendChild하면 브라우저가 자동으로 adopt한다 (pip.ts가 이미 그렇게 동작 중).
 */

/** 패널 조각들이 호스트에 되돌려 보내는 동작 — 패널·PiP 모두 content.ts의 같은 핸들러로 배선된다 */
export interface PanelCallbacks {
  /** 가사 텍스트로 싱크 생성. attribution은 사용자가 적어 넣은 출처(선택) */
  onGenerate: (lyrics: string, attribution?: string) => void;
  /** 자동 검색 다시 실행 — query가 없으면 자동 인식으로 되돌린다 */
  onRetrySearch: (query?: { title: string; artist: string }) => void;
  /** 후보 리스트 요청 — 결과는 호스트가 renderCandidateList로 받아 그린다 */
  onCandidateSearch: (query: { title: string; artist: string }) => void;
  /** 후보 리스트에서 사용자가 직접 선택 */
  onPickCandidate: (candidate: SearchCandidate) => void;
  /** 상세 검색 시트 열기 (호스트가 자기 창 안에서 연다) */
  onOpenSearch: () => void;
  /** 설정 열기 — 인증 실패 배너에서 API 키 칸으로 바로 보내기 위한 것 */
  onOpenSettings: () => void;
  /** 서버 상태 다시 확인 (배너의 '다시 확인') */
  onRecheckServer: () => void;
  /** 권한 관리 페이지(options_ui) 열기 — 로컬 서버 호스트 권한을 허용/철회하는 곳.
   *  여기서 직접 `permissions.request()`를 부를 수 없다: 이 조각은 content script와 PiP
   *  document에서 그려지고, `chrome.permissions`는 확장 페이지에서만 쓸 수 있다. */
  onOpenPermissions: () => void;
}

/** 조각을 만들 때 필요한 호스트 능력 — 콜백 + 서버 상태에 연동되는 생성 버튼 팩토리 */
export interface PanelContext {
  callbacks: PanelCallbacks;
  /** 호스트가 만든 생성 버튼 (자기 목록에 등록해 setServerStatus로 일괄 갱신한다) */
  makeGenerateButton: (label: string, onClick: () => void) => HTMLButtonElement;
  /**
   * 지금 서버를 쓸 수 있는가 + 못 쓴다면 왜인지.
   * 조각들은 이걸 보고 "가사를 찾지 못했어요"와 "서버가 거부했어요"를 구분해 말한다.
   */
  server: ServerStatus;
  /** 디버그 모드 — 서버가 정상일 때도 요청 로그를 노출할지 결정한다 */
  debug: boolean;
  /** 최근 서버 요청 로그 — 접힌 섹션을 펼칠 때만 호출된다 */
  loadServerLog: () => Promise<ServerLogEntry[]>;
}

/** 서버 미가용 시 비활성 + 사유 툴팁이 붙는 '싱크 생성' 계열 버튼 */
export function createGenerateButton(
  label: string, server: ServerStatus, onClick: () => void,
): HTMLButtonElement {
  const btn = h('button', { className: 'ey-primary-btn ey-generate-btn', on: { click: onClick } },
    icon(ICONS.sparkle), label);
  applyServerGate(btn, server);
  return btn;
}

/** 서버가 필요한 컨트롤 하나를 상태에 맞춰 잠그거나 푼다 (사유는 툴팁으로) */
export function applyServerGate(
  btn: HTMLButtonElement, server: ServerStatus, enabledTitle = '',
): void {
  const usable = serverUsable(server);
  btn.disabled = !usable;
  btn.title = usable ? enabledTitle : serverBlockedTip(server);
  btn.classList.toggle('ey-server-blocked', !usable);
}

// ── 서버 상태 배너 + 요청 로그 ───────────────────────────────────

/** 상태 종류별 배너·상태 화면 아이콘 — 자물쇠는 "권한", 열쇠는 "인증", 그 밖은 장애 */
function serverBarIcon(kind: ServerStatus['kind']): string {
  if (kind === 'auth') return '🔑';
  if (kind === 'permission') return '🔒';
  return '⚠️';
}

/**
 * 최근 서버 요청 로그 (기본 접힘).
 *
 * **언제 보이나**: 서버 상태가 정상이 아닐 때는 디버그 모드와 무관하게 항상,
 * 정상일 때는 디버그 모드에서만. 근거 — 뭔가 깨졌을 때 로그는 사용자가 원인을 짚거나
 * 그대로 옮겨 신고할 수 있는 유일한 증거라 숨기면 안 되고, 멀쩡할 때는 평상시 화면을
 * 어지럽히는 잡음일 뿐이다. 어느 경우든 **접힌 채로** 시작해 화면을 점유하지 않는다.
 */
export function buildServerLogSection(ctx: PanelContext): HTMLDetailsElement | null {
  if (!serverKnownBad(ctx.server) && !ctx.debug) return null;

  const list = h('div', { className: 'ey-log-list ey-state-sub', text: t('panels.serverLog.loadHint') });
  const details = h('details', { className: 'ey-log' },
    h('summary', { className: 'ey-log-summary', text: t('panels.serverLog.title') }),
    list,
  );
  let loading = false;
  const refresh = () => {
    if (loading) return;
    loading = true;
    list.replaceChildren(h('div', { className: 'ey-state-sub', text: t('panels.serverLog.loading') }));
    void ctx.loadServerLog().then(entries => {
      loading = false;
      if (entries.length === 0) {
        list.replaceChildren(h('div', { className: 'ey-state-sub', text: t('panels.serverLog.empty') }));
        return;
      }
      // 경로·본문의 키류는 기록 시점에 이미 마스킹됐다 — 여기서는 그대로 그린다
      list.replaceChildren(...entries.map(e =>
        h('div', { className: `ey-log-row${e.ok ? '' : ' bad'}`, text: formatLogEntry(e) })));
    });
  };
  details.addEventListener('toggle', () => {
    if (details.open) refresh();
  });
  return details;
}

/**
 * 서버 문제 배너 — 사유 한 줄 + 원인 코드 + 복구 동작 + 접이식 로그.
 *
 * 메인 패널과 PiP가 **같은 이 조각**을 쓴다. 다만 그리는 자리는 각자의 창 골격
 * (헤더 아래 / 푸터 위)이다 — 본문 상태 화면들이 각자 또 그리면 같은 배너가 두 번
 * 보인다. 그래서 아래 build*State들은 이 함수를 부르지 않고 호스트에 맡긴다.
 * 정상이거나 아직 확인 전이면 null이다 (첫 확인 전 빨간 배너는 없는 오류를 깜빡인다).
 */
export function buildServerStatusBar(ctx: PanelContext): HTMLDivElement | null {
  const status = ctx.server;
  if (!serverKnownBad(status)) return null;

  const needsPerm = needsHostPermission(status);
  const bar = h('div', { className: `ey-server-bar ey-server-${status.kind}` });
  const head = h('div', { className: 'ey-server-bar-head' },
    h('span', { className: 'ey-server-bar-icon', text: serverBarIcon(status.kind) }),
    h('span', { className: 'ey-server-bar-text', text: statusLine(status) }),
  );
  bar.append(head);
  // 서버가 준 힌트가 있으면 그대로 — 사용자가 서버 로그를 뒤지지 않아도 되게
  if (status.detail) bar.append(h('div', { className: 'ey-server-bar-detail', text: status.detail }));
  // 권한은 서버 장애가 아니다 — 왜 갑자기 이 상태가 됐는지까지 말해 준다. 자체 호스팅으로
  // 쓰던 사람은 확장 업데이트로 권한이 회수돼 여기 왔을 수 있고, 그때 이 한 줄이 없으면
  // 멀쩡히 쓰던 기능이 이유 없이 깨진 것처럼 보인다.
  if (needsPerm) {
    bar.append(h('div', {
      className: 'ey-server-bar-detail',
      text: t('panels.serverBar.permissionNeeded'),
    }));
  }

  const actions = h('div', { className: 'ey-server-bar-actions' });
  if (needsPerm) {
    actions.append(h('button', {
      className: 'ey-primary-btn',
      text: t('panels.serverBar.openPermissions'),
      attrs: { title: t('panels.serverBar.openPermissionsTitle') },
      on: { click: () => ctx.callbacks.onOpenPermissions() },
    }));
  }
  actions.append(
    h('button', {
      className: 'ey-secondary-btn',
      text: t('panels.serverBar.recheck'),
      attrs: { title: t('panels.serverBar.recheckTitle') },
      on: { click: () => ctx.callbacks.onRecheckServer() },
    }),
    h('button', {
      className: 'ey-secondary-btn',
      text: t('panels.serverBar.openSettings'),
      attrs: { title: t('panels.serverBar.openSettingsTitle') },
      on: { click: () => ctx.callbacks.onOpenSettings() },
    }),
  );
  bar.append(actions);
  const log = buildServerLogSection(ctx);
  if (log) bar.append(log);
  return bar;
}

/**
 * 호스트 골격(메인 패널 헤더 아래 / PiP 푸터 위)에 넣을 서버 상태 표시 한 덩어리.
 *
 * - 서버 고장: 배너(사유·코드·복구 버튼·접이식 로그)
 * - 정상 + 디버그: 접이식 로그만
 * - 그 밖(정상, 또는 아직 확인 전): 아무것도 안 그린다
 */
export function buildServerStatusSlot(ctx: PanelContext): HTMLElement | null {
  return buildServerStatusBar(ctx) ?? buildServerLogSection(ctx);
}

/** 리스트 자리(.ey-result-list)에 한 줄 상태 메시지를 표시 */
export function setListStatus(listEl: HTMLElement | null, message: string): void {
  listEl?.replaceChildren(h('div', { className: 'ey-state-sub', text: message }));
}

// ── 검색 폼 ──────────────────────────────────────────────────────

export interface SearchFormRefs {
  el: HTMLDivElement;
  titleInput: HTMLInputElement;
  artistInput: HTMLInputElement;
  /** 현재 입력값으로 검색 실행 (제목이 비면 아무것도 하지 않는다) */
  submit: () => void;
}

/**
 * 제목·아티스트 입력 + 실행 버튼.
 * submitOnEnter는 호출부마다 다르다 — 빈 상태 폼은 예전부터 Enter를 처리하지 않았고
 * 검색 시트만 처리했다. 추출하면서 동작이 바뀌지 않도록 옵션으로 유지한다.
 *
 * **initial.title은 이미 정리된 제목이다**(【】·MV·feat·업로더 접두 등이 걷힌 것) — 검색은
 * 그 형태에서 가장 잘 맞으므로 기본값을 바꾸지 않는다. 다만 화면에는 그게 "정리된 값"이라는
 * 사실이 어디에도 없어서, 사용자는 영상 제목을 통째로 다시 쳐야 하는 줄 알고 매번 지웠다
 * 썼다 했다(제보). 그래서 두 가지를 명시한다: 무엇을 넣어야 하는지(안내 한 줄)와, 정리가
 * 과했을 때 원본으로 되돌아가는 길(rawTitle 버튼). rawTitle이 정리본과 같으면 버튼은
 * 나오지 않는다 — 눌러도 아무 일도 없는 버튼은 없느니만 못하다.
 */
export function buildSearchForm(
  initial: { title: string; artist: string; rawTitle?: string },
  opts: { buttonLabel: string; submitOnEnter: boolean; onSubmit: (q: { title: string; artist: string }) => void },
): SearchFormRefs {
  const titleInput = h('input', { className: 'ey-input', attrs: { placeholder: t('panels.search.titlePlaceholder') } });
  titleInput.value = initial.title;
  const artistInput = h('input', { className: 'ey-input', attrs: { placeholder: t('panels.search.artistPlaceholder') } });
  artistInput.value = initial.artist;

  const submit = () => {
    const title = titleInput.value.trim();
    if (!title) return;
    opts.onSubmit({ title, artist: artistInput.value.trim() });
  };
  if (opts.submitOnEnter) {
    titleInput.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
    artistInput.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
  }

  const raw = initial.rawTitle?.trim() ?? '';
  const rawBtn = raw && raw !== initial.title.trim()
    ? h('button', {
      className: 'ey-secondary-btn ey-search-raw-btn',
      text: t('panels.search.useRawTitle'),
      attrs: { type: 'button', title: t('panels.search.useRawTitleTitle', [raw]) },
      on: {
        click: () => {
          titleInput.value = raw;
          titleInput.focus();
          titleInput.select(); // 원본을 넣자마자 손보고 싶은 경우가 많다
        },
      },
    })
    : null;

  const el = h('div', { className: 'ey-search-form' },
    h('div', { className: 'ey-search-hint', text: t('panels.search.titleHint') }),
    titleInput,
    rawBtn,
    artistInput,
    h('button', { className: 'ey-primary-btn', text: opts.buttonLabel, on: { click: submit } }),
  );
  return { el, titleInput, artistInput, submit };
}

// ── 가사 검색 링크 (외부 사이트로 보내기만 한다) ────────────────────

/**
 * 가사를 못 찾았을 때 나무위키·구글·Genius 검색 결과로 보내는 링크.
 *
 * **스크래핑이 아니다** — 각 사이트의 검색 URL을 만들어 새 탭으로 여는 것이 전부이며,
 * 확장이 그 페이지에 요청을 보내거나 본문을 가져오지 않는다. 사용자가 자기 브라우저로
 * 방문해 가사를 확인하고, 원하면 붙여넣기 칸에 직접 옮겨 담는다.
 */
export function buildLyricsSearchLinks(song: SongInfo | null): HTMLDivElement {
  const wrap = h('div', { className: 'ey-search-links' });
  const query = [song?.title?.trim(), song?.artist?.trim()].filter(Boolean).join(' ');
  if (!query) return wrap; // 곡 정보가 없으면 검색어를 만들 수 없다 (CSS로 빈 wrap은 숨김)

  const enc = encodeURIComponent;
  const targets: { label: string; url: string }[] = [
    { label: t('panels.search.namuwiki'), url: `https://namu.wiki/Search?q=${enc(query)}` },
    { label: t('panels.search.google'), url: `https://www.google.com/search?q=${enc(`${query} ${t('panels.search.lyricsWord')}`)}` },
    { label: 'Genius', url: `https://genius.com/search?q=${enc(query)}` },
  ];
  wrap.append(
    h('div', { className: 'ey-state-sub', text: t('panels.search.externalLinksLabel') }),
    h('div', { className: 'ey-search-links-row' }, ...targets.map(target =>
      h('a', {
        className: 'ey-secondary-btn ey-search-link',
        text: target.label,
        attrs: { href: target.url, target: '_blank', rel: 'noopener noreferrer', title: t('panels.search.externalLinkTitle', [target.label, query]) },
      }))),
  );
  return wrap;
}

// ── 가사 붙여넣기 섹션 ────────────────────────────────────────────

/**
 * 가사 직접 붙여넣기 섹션 — 텍스트영역+생성 버튼을 **처음부터** 보여준다.
 *
 * 예전에는 접힌 토글 버튼("가사 직접 붙여넣기")을 한 번 눌러야 이 폼이 나타나는
 * 2단계였다 — 사용자 제보("붙여넣기 버튼 한 번 누르고 나서야 메뉴 뜨는 거 없애줘")로
 * 그 접힘 단계 자체를 없앴다(2026-07). 생성 버튼의 활성/비활성·유효성 검사(빈 채로
 * 누르면 안내)·attribution 입력은 그대로다 — 폼 자체는 손대지 않고 항상 보이게만 했다.
 */
export function buildPasteSection(ctx: PanelContext): HTMLDivElement {
  const lyricsArea = h('textarea', {
    className: 'ey-textarea',
    attrs: {
      placeholder: t('panels.paste.placeholder'),
      rows: '6',
    },
  });
  // 붙여넣기 칸의 한 줄 안내 자리 (빈 채로 생성을 누른 경우 등).
  // 예전에는 여기에 '이 영상 자막에서 가사 가져오기' 버튼과 트랙 목록이 붙어 있었지만,
  // 자막은 이제 싱크가 없을 때 content.ts가 알아서 폴백하므로 고르는 단계 자체가 없다.
  const statusEl = h('div', { className: 'ey-result-list' });
  // 출처는 선택 입력 — 어디서 옮겨온 가사인지 남겨두면 나중에 출처 표기를 되살릴 수 있다.
  // 비워도 생성은 그대로 진행된다 (강제하지 않는다).
  const attributionInput = h('input', {
    className: 'ey-input',
    attrs: {
      placeholder: t('panels.paste.attributionPlaceholder'),
      title: t('panels.paste.attributionTitle'),
    },
  });
  const pasteSection = h('div', { className: 'ey-paste-section' },
    lyricsArea,
    h('div', {
      className: 'ey-state-sub',
      text: t('panels.paste.filterNote'),
    }),
    statusEl,
    attributionInput,
    ctx.makeGenerateButton(t('panels.paste.generateButton'), () => {
      const text = lyricsArea.value.trim();
      if (!text) {
        // 빈 채로 눌렀을 때 무반응이면 버튼이 죽은 줄 안다 — 안내 후 입력칸으로 포커스
        setListStatus(statusEl, t('panels.paste.emptyWarning'));
        lyricsArea.focus();
        return;
      }
      // 파트 표기·주석을 걷어내고, **무엇을 걷어냈는지 반드시 보여준다** — 조용히 지우면
      // 사용자에게는 가사가 사라진 것으로 보인다. content.handleGenerate도 같은 필터를
      // 한 번 더 통과시키지만(멱등) 사용자에게 알리는 자리는 붙여넣은 이 화면이다.
      const cleaned = stripPartMarkers(text);
      const note = describeRemoved(cleaned);
      if (note) setListStatus(statusEl, note);
      else statusEl.replaceChildren();
      ctx.callbacks.onGenerate(cleaned.text, attributionInput.value.trim() || undefined);
    }),
  );
  return h('div', { className: 'ey-paste-wrap' }, pasteSection);
}

// ── 상태 화면 ────────────────────────────────────────────────────

/**
 * 서버를 쓸 수 없을 때의 화면 — "가사를 찾지 못했어요" 대신 여기로 온다.
 *
 * 사용자가 겪은 문제가 정확히 이것이다: 서버가 401을 돌려줬는데 화면은 곡에 가사가
 * 없는 것처럼 굴었다. 그래서 여기서는 **서버 문제라고 먼저 말하고**, 서버가 필요한
 * 조작(싱크 생성·후보 검색·붙여넣기)은 아예 내놓지 않는다 — 눌러도 실패할 버튼을
 * 보여 주는 건 작동하는 척하는 것이다. 반대로 **서버 없이 되는 것**(다른 사이트에서
 * 가사 찾기, 자동 검색 재시도)은 그대로 남긴다.
 */
export function buildServerDownState(ctx: PanelContext, song: SongInfo | null): HTMLDivElement {
  const status = ctx.server;
  // 사유·원인 코드·복구 버튼·로그는 호스트가 그리는 배너(buildServerStatusBar)에 있다.
  // 여기서는 "이 화면이 왜 비었는지"만 말한다.
  const el = h('div', { className: 'ey-state' },
    // 장애 아이콘만 배너와 다르다 — 여기서는 "이 화면이 비었다"는 맥락이라 플러그가 낫다
    h('div', {
      className: 'ey-state-emoji',
      text: status.kind === 'offline' || status.kind === 'error' ? '🔌' : serverBarIcon(status.kind),
    }),
    h('div', { className: 'ey-state-text', text: status.reason || t('panels.serverDown.text') }),
  );
  if (status.code) el.append(h('div', { className: 'ey-state-sub', text: status.code }));
  el.append(
    // 서버가 죽어도 LRCLIB·위키 조회는 백그라운드가 직접 하므로, 여기까지 왔다는 건
    // 외부 소스에도 이 곡 가사가 없었다는 뜻이다 — 둘 다 사실대로 말한다
    h('div', {
      className: 'ey-state-sub',
      text: t('panels.serverDown.note'),
    }),
    buildLyricsSearchLinks(song),
    h('button', {
      className: 'ey-secondary-btn',
      text: t('panels.serverDown.retry'),
      attrs: {
        title: status.kind === 'permission'
          ? t('panels.serverDown.retryTitlePermission')
          : t('panels.serverDown.retryTitleOther'),
      },
      on: { click: () => ctx.callbacks.onRetrySearch() },
    }),
  );
  return el;
}

/** "가사를 찾지 못했어요" — 재검색 폼 + 외부 검색 링크 + 붙여넣기 */
export function buildEmptyState(ctx: PanelContext, song: SongInfo | null): HTMLDivElement {
  // 서버가 고장난 것이 확인됐다면 "가사를 찾지 못했어요"는 거짓말이 된다 — 서버 문제를
  // 먼저 말한다. 아직 확인 전(unknown)이면 단정하지 않고 평소 화면을 그대로 쓴다.
  if (serverKnownBad(ctx.server)) return buildServerDownState(ctx, song);

  const form = buildSearchForm(
    { title: song?.title ?? '', artist: song?.artist ?? '' },
    {
      buttonLabel: t('panels.empty.searchAgain'),
      submitOnEnter: false,
      onSubmit: q => ctx.callbacks.onRetrySearch(q),
    },
  );
  return h('div', { className: 'ey-state' },
    h('div', { className: 'ey-state-emoji', text: '🎵' }),
    h('div', { className: 'ey-state-text', text: t('panels.empty.title') }),
    form.el,
    h('button', {
      className: 'ey-secondary-btn',
      text: t('panels.empty.detailedSearch'),
      on: { click: () => ctx.callbacks.onOpenSearch() },
    }),
    buildLyricsSearchLinks(song),
    h('div', { className: 'ey-divider' }),
    buildPasteSection(ctx),
  );
}

/** 검색 중 스켈레톤 */
export function buildLoadingState(ctx: PanelContext, message: string): HTMLDivElement {
  const skeleton = h('div', { className: 'ey-skeleton' });
  for (let i = 0; i < 3; i++) skeleton.append(h('div', { className: 'ey-skeleton-bar' }));
  return h('div', { className: 'ey-state' },
    skeleton,
    h('div', { className: 'ey-state-text', text: message }),
    // 자동 검색을 기다릴 필요 없이 바로 수동 검색으로 전환
    h('button', {
      className: 'ey-secondary-btn',
      text: t('panels.loading.manualSearch'),
      on: { click: () => ctx.callbacks.onOpenSearch() },
    }),
  );
}

/**
 * 오류 상태 — 무엇이 실패했는지(message)와 왜인지(서버 상태 배너)를 같이 보여 준다.
 * detail은 호출부가 아는 추가 사유(예: 서버가 준 힌트)로, 배너와 별개로 한 줄 더 남긴다.
 */
export function buildErrorState(ctx: PanelContext, message: string, detail?: string): HTMLDivElement {
  const el = h('div', { className: 'ey-state' },
    h('div', { className: 'ey-state-emoji', text: '⚠️' }),
    h('div', { className: 'ey-state-text', text: message }),
  );
  if (detail) el.append(h('div', { className: 'ey-state-sub', text: detail }));
  el.append(h('button', {
    className: 'ey-primary-btn',
    text: t('panels.error.retry'),
    on: { click: () => ctx.callbacks.onRetrySearch() },
  }));
  return el;
}

export interface GeneratingStateRefs {
  el: HTMLDivElement;
  bar: HTMLDivElement;
  text: HTMLDivElement;
}

/** 싱크 생성 진행 상태 (진행 바 + 문구) */
export function buildGeneratingState(pct: number, text: string): GeneratingStateRefs {
  const bar = h('div', { className: 'ey-progress-bar' });
  bar.style.width = `${pct}%`;
  const textEl = h('div', { className: 'ey-state-text', text });
  const el = h('div', { className: 'ey-state' },
    h('div', { className: 'ey-state-emoji', text: '✨' }),
    textEl,
    h('div', { className: 'ey-progress' }, bar),
    h('div', { className: 'ey-state-sub', text: t('panels.generating.note') }),
  );
  return { el, bar, text: textEl };
}

/** 타임싱크 없는 일반 가사 목록 */
export function buildPlainLines(lines: LyricLine[]): { el: HTMLDivElement; lineEls: HTMLElement[] } {
  const list = h('div', { className: 'ey-lines ey-lines-plain' });
  const lineEls: HTMLElement[] = [];
  for (const line of lines) {
    const el = h('div', { className: 'ey-line ey-line-plain', text: line.text, attrs: { dir: 'auto' } });
    if (line.pronunciation) el.append(h('div', { className: 'ey-line-pron', text: line.pronunciation, attrs: { dir: 'auto' } }));
    if (line.translation) el.append(h('div', { className: 'ey-line-tr', text: line.translation, attrs: { dir: 'auto' } }));
    lineEls.push(el);
    list.append(el);
  }
  return { el: list, lineEls };
}

// ── 검색 시트 ────────────────────────────────────────────────────

export interface SearchSheetRefs {
  el: HTMLDivElement;
  /** 후보 결과가 그려지는 자리 */
  results: HTMLDivElement;
  /** 현재 입력값으로 후보 검색 실행 */
  runSearch: () => void;
}

/**
 * 상시 재검색 시트 — 검색 폼 + 후보 리스트 + 붙여넣기.
 * extras는 호스트별 추가 섹션(메인 패널의 '다른 영상 싱크 연결'·'싱크 초기화' 등)을
 * 붙여넣기 아래에 그대로 이어 붙인다.
 */
export function buildSearchSheet(
  ctx: PanelContext,
  // rawTitle은 그대로 buildSearchForm으로 흘려보낸다 — 시트가 쓰는 값이 아니라
  // 폼의 '영상 제목 그대로' 탈출구가 쓰는 값이라, 여기서 해석하지 않고 통과만 시킨다
  state: { title: string; artist: string; rawTitle?: string },
  opts: { onBack: () => void; extras?: (Node | null)[] },
): SearchSheetRefs {
  const results = h('div', { className: 'ey-result-list' });
  const form = buildSearchForm(state, {
    buttonLabel: t('panels.searchSheet.searchButton'),
    submitOnEnter: true,
    onSubmit: q => {
      setListStatus(results, t('panels.searchSheet.searching'));
      ctx.callbacks.onCandidateSearch(q);
    },
  });
  const el = h('div', { className: 'ey-state ey-search-state' },
    h('button', {
      className: 'ey-secondary-btn ey-search-back',
      text: t('panels.searchSheet.back'),
      on: { click: () => opts.onBack() },
    }),
    h('div', { className: 'ey-state-text', text: t('panels.searchSheet.title') }),
    h('div', { className: 'ey-state-sub', text: t('panels.searchSheet.sub') }),
  );
  // 후보 검색 자체는 서버가 없어도 된다 (LRCLIB은 확장이 직접 조회한다) — 그래서 잠그지
  // 않는다. 다만 위키 원제 매칭은 서버 인덱스를 쓰므로 결과가 줄어들 수 있음을 알린다.
  // (사유·복구 버튼은 호스트가 그리는 배너에 있으므로 여기서 반복하지 않는다)
  if (serverKnownBad(ctx.server)) {
    el.append(h('div', {
      className: 'ey-state-sub',
      text: t('panels.searchSheet.serverBadNote'),
    }));
  }
  el.append(
    form.el,
    results,
    h('div', { className: 'ey-divider' }),
    buildPasteSection(ctx),
  );
  for (const extra of opts.extras ?? []) {
    if (extra) el.append(extra);
  }
  return { el, results, runSearch: form.submit };
}

// ── 결과 리스트 렌더 ─────────────────────────────────────────────

/** SEARCH_CANDIDATES 응답을 후보 버튼 목록으로 그린다 */
export function renderCandidateList(
  listEl: HTMLElement, candidates: SearchCandidate[], onPick: (c: SearchCandidate) => void,
): void {
  if (candidates.length === 0) {
    setListStatus(listEl, t('panels.results.empty'));
    return;
  }
  const fmt = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, '0')}`;
  listEl.replaceChildren(...candidates.map(c => {
    const isWiki = c.source === 'vocaro';
    const label = isWiki ? c.title : `${c.title}${c.artist ? ' — ' + c.artist : ''}`;
    const meta = isWiki
      ? t('panels.results.pronTranslationMeta')
      : `${c.synced ? t('panels.results.syncedMeta') : t('panels.results.plainMeta')}${c.duration > 0 ? ` · ${fmt(c.duration)}` : ''}`;
    const btn = h('button', {
      className: 'ey-result-item',
      on: { click: () => onPick(c) },
    },
      h('span', { className: `ey-result-src${isWiki ? ' vocaro' : ''}`, text: isWiki ? t('panels.results.vocaroLabel') : t('panels.results.lrclibLabel') }),
      h('span', { className: 'ey-result-title', text: label }),
      h('span', { className: 'ey-result-meta', text: meta }),
    );
    btn.title = isWiki ? c.url : label;
    return btn;
  }));
}

// ══════════════════════════════════════════════════════════════════
//  2026-08 UI 개편 조각들 — 설정 / 평가 · 제보 / 공지 / 기여.
//  공통 원칙은 파일 머리말과 같다: 상태를 호스트 필드에 쓰지 않고 콜백과 값을 받아
//  엘리먼트를 돌려준다. 그래야 메인 패널과 PiP가 같은 조각을 동시에 띄울 수 있다.
// ══════════════════════════════════════════════════════════════════

/**
 * 백그라운드에 직접 묻는 최소 통로 (debug-panel.ts의 sendMessageDirect와 같은 규약).
 *
 * 공지·한도·조회수는 **지금 보는 가사와 무관한 조회**라서 overlay·content에 콜백을 새로
 * 뚫을 이유가 없다 — 여기서 바로 묻는다. 확장이 리로드된 뒤 남은 content script(orphan)
 * 에서 sendMessage를 부르면 예외가 터지므로 `chrome.runtime?.id` 가드를 먼저 통과한다.
 * 실패는 전부 `{error}`로 평평해지고 호출부는 그 기능만 조용히 숨긴다 — 이 셋은 구버전
 * 서버에 아예 없는 엔드포인트라 404가 비정상이 아니라 **정상 경로**다.
 */
async function sendPanelMessage<T>(message: BgRequest): Promise<MessageResponse<T>> {
  if (!chrome.runtime?.id) return { error: 'extension context invalidated' };
  try {
    return await chrome.runtime.sendMessage(message) as MessageResponse<T>;
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/** 분석 깊이의 사람 말 이름 — 제보 안내와 기여 목록 배지가 같은 낱말을 써야 한다 */
function depthLabel(depth: 'fast' | 'medium' | 'heavy'): string {
  if (depth === 'fast') return t('panels.depth.fast');
  if (depth === 'medium') return t('panels.depth.medium');
  return t('panels.depth.heavy');
}

/**
 * "오늘 / 어제 / N일 전 / 날짜".
 * 일수는 **자정 경계**로 센다(경과 시간이 아니라) — 23시에 만든 것이 다음 날 1시에
 * "0일 전"으로 보이면 어제 한 일이 오늘 것처럼 읽힌다.
 */
function relativeDate(ms: number): string {
  if (!Number.isFinite(ms)) return '';
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((startOfDay(new Date()) - startOfDay(new Date(ms))) / 86_400_000);
  if (days <= 0) return t('panels.date.today');
  if (days === 1) return t('panels.date.yesterday');
  if (days < 7) return t('panels.date.daysAgo', [String(days)]);
  return new Date(ms).toLocaleDateString();
}

/** 시트 공통 머리(뒤로 + 제목) — 공지·기여가 같은 골격을 쓴다 */
function buildSheetHead(title: string, onBack?: () => void): HTMLDivElement {
  return h('div', { className: 'ey-sheet-head' },
    onBack
      ? h('button', {
        className: 'ey-secondary-btn ey-sheet-back',
        text: t('panels.sheet.back'),
        attrs: { type: 'button' },
        on: { click: () => onBack() },
      })
      : null,
    h('div', { className: 'ey-sheet-title', text: title }),
  );
}

// ── 설정 시트 (범주 + 검색) ──────────────────────────────────────

interface SettingRowBase {
  /**
   * 설정 키 — DOM id가 아니라 **검색 대상**이다. 라벨을 모른 채 'serverUrl'·'pitchF0'로
   * 찾는 사람이 실제로 있고(문서·제보 따라 하기), 라벨만 훑으면 그 경로가 막힌다.
   */
  key: string;
  label?: string;
  /** 툴팁 — 라벨에 붙고 검색 대상에도 들어간다(설명 문구로 찾는 경우) */
  title?: string;
  /** 라벨 오른쪽 표식(서버 상태 점 등). 호출부가 참조를 계속 들고 있어야 해서 받는다 */
  labelSuffix?: HTMLElement;
  /** 'col'이면 라벨 아래로 컨트롤을 내린다 — 긴 입력칸(서버 주소·API 키) */
  layout?: 'row' | 'col';
  /** 검색에 추가로 걸리게 할 낱말들 (예: '폰트 글씨 크기') */
  keywords?: string;
}

export interface SettingCheckboxRow extends SettingRowBase {
  kind: 'checkbox';
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  /** 체크박스 왼쪽에 함께 놓이는 슬라이더 — 멜로디·메트로놈은 예전부터 '볼륨+켜기' 한 줄이었다 */
  range?: { value: number; min?: number; max?: number; step?: number; onChange: (v: number) => void };
}

export interface SettingSelectRow extends SettingRowBase {
  kind: 'select';
  label: string;
  value: string;
  options: [string, string][];
  onChange: (v: string) => void;
}

export interface SettingRangeRow extends SettingRowBase {
  kind: 'range';
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  /** 슬라이더 옆 현재값 표시 — 없으면 숫자를 내지 않는다(볼륨류는 수치가 무의미하다) */
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

export interface SettingTextRow extends SettingRowBase {
  kind: 'text';
  label: string;
  value: string;
  placeholder?: string;
  password?: boolean;
  /** 저장 직전 다듬기 — 빈 문자열을 돌려주면 **저장하지 않고 되돌린다**(서버 주소를 비우는 사고 방지) */
  sanitize?: (v: string) => string;
  onChange: (v: string) => void;
}

export interface SettingNoteRow extends SettingRowBase {
  kind: 'note';
  text: string;
  tone?: 'normal' | 'bad';
}

export interface SettingButtonRow extends SettingRowBase {
  kind: 'button';
  label: string;
  /** 눌린 버튼을 넘겨준다 — '초기화했어요'처럼 그 자리에서 결과를 말해야 하는 경우가 있다 */
  onClick: (btn: HTMLButtonElement) => void;
}

/**
 * 호출부가 이미 만들어 둔 컨트롤을 그대로 꽂는 자리.
 *
 * 오디오 기기 목록처럼 **비동기로 채워지는 select**나 서버 상태에 따라 보였다 숨었다 하는
 * 버튼은 호출부가 참조를 들고 있어야 한다. 그것까지 선언형으로 감싸면 참조를 되돌려 주는
 * 또 다른 장치가 필요해지므로 그 경우만 예외로 남긴다.
 */
export interface SettingCustomRow extends SettingRowBase {
  kind: 'custom';
  control: HTMLElement;
}

export type SettingRow =
  | SettingCheckboxRow | SettingSelectRow | SettingRangeRow | SettingTextRow
  | SettingNoteRow | SettingButtonRow | SettingCustomRow;

export interface SettingsSection {
  id: string;
  title: string;
  /** 머리에 붙는 이모지 한 자 (선택) */
  icon?: string;
  rows: SettingRow[];
}

export interface SettingsSheetSpec {
  sections: SettingsSection[];
  onClose: () => void;
}

export interface SettingsSheetRefs {
  el: HTMLDivElement;
  /** 검색칸으로 포커스 */
  focusFilter: () => void;
}

/**
 * 섹션 펼침 상태 — **세션 기억**(모듈 수명 = 탭 수명)이다.
 *
 * 설정을 열 때마다 시트를 새로 그리므로 어딘가 남기지 않으면 방금 펼쳐 둔 자리가 매번
 * 접힌 채 돌아온다(설정은 고치고 확인하고 또 고치느라 연달아 연다). 그렇다고 storage에
 * 남길 값은 아니다: 취향이 아니라 "지금 뭘 만지는 중인가"라 다음 날까지 살아 있을 이유가
 * 없고, 토글 하나에 저장소 쓰기를 늘릴 값어치도 없다.
 */
const settingsSectionOpen = new Map<string, boolean>();

function settingRowHaystack(row: SettingRow): string {
  const parts = [row.key, row.label ?? '', row.title ?? '', row.keywords ?? ''];
  if (row.kind === 'note') parts.push(row.text);
  if (row.kind === 'select') parts.push(...row.options.map(([, label]) => label));
  return parts.join(' ').toLowerCase();
}

function buildSettingRange(
  spec: { value: number; min?: number; max?: number; step?: number; onChange: (v: number) => void },
): HTMLInputElement {
  const range = h('input', {
    className: 'ey-settings-range',
    attrs: {
      type: 'range',
      min: String(spec.min ?? 0),
      max: String(spec.max ?? 1),
      step: String(spec.step ?? 0.01),
      value: String(spec.value),
    },
  });
  // change(놓는 순간)로만 저장한다 — 드래그 중 input마다 저장하면 저장 체인이 수십 번 돈다
  range.addEventListener('change', () => spec.onChange(Number(range.value)));
  return range;
}

function buildSettingControl(row: SettingRow): HTMLElement {
  switch (row.kind) {
    case 'checkbox': {
      const box = h('input', { attrs: { type: 'checkbox' } });
      box.checked = row.value;
      box.addEventListener('change', () => row.onChange(box.checked));
      if (!row.range) return box;
      return h('span', { className: 'ey-settings-inline' }, buildSettingRange(row.range), box);
    }
    case 'select': {
      const select = h('select', { className: 'ey-select' });
      for (const [value, label] of row.options) select.append(h('option', { text: label, attrs: { value } }));
      select.value = row.value;
      select.addEventListener('change', () => row.onChange(select.value));
      return select;
    }
    case 'range': {
      const slider = buildSettingRange(row);
      const format = row.format;
      if (!format) return slider;
      // 값이 뜻을 갖는 슬라이더(너비 px·배율)는 숫자가 없으면 조준할 수 없다
      const readout = h('span', { className: 'ey-settings-range-val', text: format(row.value) });
      slider.addEventListener('input', () => { readout.textContent = format(Number(slider.value)); });
      return h('span', { className: 'ey-settings-inline' }, slider, readout);
    }
    case 'text': {
      const input = h('input', {
        className: 'ey-input',
        attrs: {
          placeholder: row.placeholder ?? '',
          ...(row.password ? { type: 'password' } : {}),
        },
      });
      input.value = row.value;
      input.addEventListener('change', () => {
        const value = row.sanitize ? row.sanitize(input.value) : input.value.trim();
        if (row.sanitize && !value) {
          input.value = row.value; // 다듬어서 빈 값이 되면 저장하지 않고 원래 값으로 되돌린다
          return;
        }
        row.onChange(value);
      });
      return input;
    }
    case 'button': {
      const btn = h('button', {
        className: 'ey-secondary-btn ey-settings-action',
        text: row.label,
        attrs: { type: 'button', ...(row.title ? { title: row.title } : {}) },
      });
      btn.addEventListener('click', () => row.onClick(btn));
      return btn;
    }
    case 'custom':
      return row.control;
    case 'note':
      return h('div', { className: `ey-settings-note${row.tone === 'bad' ? ' bad' : ''}`, text: row.text });
  }
}

/** 한 줄 그리기 — note와 라벨 없는 button·custom은 라벨 칸 없이 통째로 놓는다(권한 버튼 등) */
function buildSettingRowEl(row: SettingRow): HTMLElement {
  const control = buildSettingControl(row);
  if (row.kind === 'note') return control;
  if (row.kind === 'button' || !row.label) {
    return h('div', { className: 'ey-settings-row ey-settings-bare' }, control);
  }
  const label = h('label', { text: row.label, ...(row.title ? { attrs: { title: row.title } } : {}) });
  if (row.labelSuffix) label.append(row.labelSuffix);
  return h('div', {
    className: `ey-settings-row${row.layout === 'col' ? ' ey-settings-col' : ''}`,
  }, label, control);
}

/**
 * 범주별로 접히는 설정 시트 + 맨 위 검색칸.
 *
 * 예전에는 40줄 넘는 형제 div가 한 줄로 늘어서 있었다 — 찾는 설정이 어디쯤인지 알 방법이
 * 스크롤뿐이라, **이름을 아는 설정조차** 못 찾는 상태였다. 고치는 길은 둘이고 둘 다 있어야
 * 한다: 분류(짐작으로 도달)와 검색(이름만 알 때 직행).
 *
 * 기본 펼침은 "첫 섹션만"이다. 전부 펼치면 지금의 40줄 벽이 제목 몇 개 낀 채 그대로
 * 돌아와 분류가 아무것도 해결하지 못한다. 대신 발견성은 검색이 맡는다 — 검색어가 있으면
 * 걸린 섹션이 **자동으로 펼쳐지고** 안 걸린 섹션은 통째로 사라진다. 검색어를 지우면
 * 사용자가 직접 정해 둔 펼침 상태로 정확히 되돌아온다(검색 중 강제 펼침은 기억하지 않는다).
 */
export function buildSettingsSheet(spec: SettingsSheetSpec): SettingsSheetRefs {
  interface RowEntry { el: HTMLElement; haystack: string }
  interface SectionEntry { id: string; details: HTMLDetailsElement; count: HTMLElement; rows: RowEntry[] }

  const sections: SectionEntry[] = [];
  /** 검색 중에는 강제로 펼치므로 그 사이의 toggle을 사용자 의도로 기억하면 안 된다 */
  let filtering = false;

  const filter = h('input', {
    className: 'ey-input ey-settings-filter',
    attrs: { type: 'search', placeholder: t('panels.settings.filterPlaceholder') },
  });
  const noMatch = h('div', { className: 'ey-settings-empty' });
  noMatch.style.display = 'none';
  const expandAll = h('button', {
    className: 'ey-settings-expand-all',
    text: t('panels.settings.expandAll'),
    attrs: { type: 'button' },
  });
  const body = h('div', { className: 'ey-settings-sections' });

  spec.sections.forEach((section, index) => {
    const count = h('span', { className: 'ey-settings-section-count' });
    const summary = h('summary', { className: 'ey-settings-section-head' },
      section.icon ? h('span', { className: 'ey-settings-section-icon', text: section.icon }) : null,
      h('span', { className: 'ey-settings-section-title', text: section.title }),
      count,
    );
    const rowsWrap = h('div', { className: 'ey-settings-section-body' });
    const details = h('details', { className: 'ey-settings-section' }, summary, rowsWrap);
    details.open = settingsSectionOpen.get(section.id) ?? index === 0;
    // 기본값도 **반드시** 기억에 심는다 — 검색이 강제로 펼친 섹션을 검색어를 지울 때
    // 되돌리는 근거가 이 Map뿐이라, 비어 있으면 "펼쳐진 채로 남는" 상태가 굳는다
    settingsSectionOpen.set(section.id, details.open);

    const entry: SectionEntry = { id: section.id, details, count, rows: [] };
    details.addEventListener('toggle', () => {
      if (!filtering) settingsSectionOpen.set(entry.id, details.open);
    });
    for (const row of section.rows) {
      const el = buildSettingRowEl(row);
      // 범주 이름도 각 행의 검색 대상에 넣는다 — "가라오케"를 친 사람이 원하는 것은
      // 그 범주의 설정들이지 "결과 없음"이 아니다(행 라벨에는 범주명이 안 들어 있다)
      entry.rows.push({ el, haystack: `${settingRowHaystack(row)} ${section.title.toLowerCase()}` });
      rowsWrap.append(el);
    }
    sections.push(entry);
    body.append(details);
  });

  const applyFilter = () => {
    const raw = filter.value.trim();
    const query = raw.toLowerCase();
    filtering = query.length > 0;
    let total = 0;
    for (const section of sections) {
      let hit = 0;
      for (const row of section.rows) {
        const match = !query || row.haystack.includes(query);
        row.el.style.display = match ? '' : 'none';
        if (match) hit++;
      }
      total += hit;
      section.details.style.display = query && hit === 0 ? 'none' : '';
      section.count.textContent = query ? String(hit) : '';
      section.details.open = query ? hit > 0 : (settingsSectionOpen.get(section.id) ?? section.details.open);
    }
    noMatch.textContent = t('panels.settings.noMatch', [raw]);
    noMatch.style.display = query && total === 0 ? '' : 'none';
    expandAll.style.display = query ? 'none' : '';
  };
  filter.addEventListener('input', applyFilter);

  expandAll.addEventListener('click', () => {
    // 하나라도 접혀 있으면 "모두 펼치기", 전부 펼쳐져 있으면 "모두 접기"
    const open = sections.some(s => !s.details.open);
    for (const section of sections) {
      section.details.open = open;
      settingsSectionOpen.set(section.id, open);
    }
    expandAll.textContent = open ? t('panels.settings.collapseAll') : t('panels.settings.expandAll');
  });

  const el = h('div', { className: 'ey-settings' },
    h('div', { className: 'ey-settings-toolbar' }, filter, expandAll),
    noMatch,
    body,
    h('button', {
      className: 'ey-secondary-btn',
      text: t('overlay.settings.closeButton'),
      on: { click: () => spec.onClose() },
    }),
  );
  return { el, focusFilter: () => filter.focus() };
}

// ── 별점 · 제보 ──────────────────────────────────────────────────

export interface RatingPopOpts {
  /** 지금 보고 있는 싱크의 분석 깊이 — 제보 화면이 "더 좋아질 수 있다"고 말할 근거 */
  depth?: 'fast' | 'medium' | 'heavy' | null;
  /** 별 하나 = 곧바로 전송. 성공 여부를 돌려준다 */
  onSubmitRating: (rating: number) => Promise<boolean>;
  onSubmitReport: (category: string | undefined, comment: string | undefined) => Promise<boolean>;
  /** 분석 깊이 올리기 — 없으면 제보 화면의 '깊이 올리기' 버튼을 내지 않는다 */
  onDepthUpgrade?: () => void;
  /** 다 끝났으니 팝오버를 닫아 달라 — 표시 여부는 호스트가 관리한다 */
  onClose?: () => void;
}

/**
 * 별점 팝오버 — **별만 있고, 한 번 누르면 그대로 전송된다.**
 *
 * 예전 팝오버는 별·오류 유형·코멘트·전송 버튼이 한 화면에 있었다. 그래서 "괜찮았다"는
 * 한마디를 남기려던 사람도 유형을 고를지 말지 생각한 뒤 전송까지 눌러야 했고, 결과적으로
 * 가장 흔한 행동(그냥 별점)이 가장 번거로웠다. 지금은 반대다: 별점은 클릭 한 번으로 끝나고,
 * 자세한 제보는 **다른 버튼**으로 갈라져 확인 단계까지 거친다(오발송 방지).
 */
export function buildRatingPop(opts: RatingPopOpts): { el: HTMLDivElement } {
  const el = h('div', { className: 'ey-rating-pop' });

  const renderStars = () => {
    let sent = false;
    const stars: HTMLButtonElement[] = [];
    const paint = (upto: number) => stars.forEach((s, i) => s.classList.toggle('on', i < upto));
    const status = h('div', { className: 'ey-rating-status' });
    const starRow = h('div', { className: 'ey-rating-stars' });

    for (let i = 1; i <= 5; i++) {
      const star = h('button', {
        className: 'ey-rating-star',
        text: '★',
        attrs: { type: 'button', title: t('panels.rating.starTitle', [String(i)]) },
        on: {
          mouseenter: () => { if (!sent) paint(i); },
          mouseleave: () => { if (!sent) paint(0); },
          click: () => {
            if (sent) return;
            sent = true;
            paint(i);
            status.textContent = t('panels.rating.sending');
            void opts.onSubmitRating(i).then(ok => {
              status.textContent = ok ? t('panels.rating.thanks') : t('panels.rating.failed');
              if (ok) window.setTimeout(() => opts.onClose?.(), 1200);
              else sent = false; // 실패하면 다시 누를 수 있어야 한다
            });
          },
        },
      });
      stars.push(star);
      starRow.append(star);
    }

    el.replaceChildren(
      h('div', { className: 'ey-rating-title', text: t('panels.rating.title') }),
      starRow,
      h('div', { className: 'ey-rating-actions' },
        h('button', {
          className: 'ey-rating-report-btn',
          text: t('panels.rating.report'),
          attrs: { type: 'button', title: t('panels.rating.reportTitle') },
          on: {
            click: () => {
              const sheet = buildReportSheet({
                depth: opts.depth,
                onSubmit: opts.onSubmitReport,
                onDepthUpgrade: opts.onDepthUpgrade,
                onCancel: renderStars, // 제보를 그만두면 별점 화면으로 되돌아간다
                onDone: () => opts.onClose?.(),
              });
              el.replaceChildren(sheet.el);
            },
          },
        }),
      ),
      status,
    );
  };

  renderStars();
  return { el };
}

export interface ReportSheetOpts {
  depth?: 'fast' | 'medium' | 'heavy' | null;
  onSubmit: (category: string | undefined, comment: string | undefined) => Promise<boolean>;
  /** 제보를 그만둔다 — 호출부가 원래 화면으로 되돌린다 */
  onCancel: () => void;
  onDepthUpgrade?: () => void;
  /** 전송(또는 깊이 올리기)이 끝나고 잠시 뒤 */
  onDone?: () => void;
}

/**
 * 오류 제보 — 유형 + 코멘트 + **확인 단계**.
 *
 * 확인을 넣은 이유는 하나다: 제보는 되돌릴 수 없다. 예전에는 버튼 한 번에 그대로 나갔고
 * 오매칭 제보는 확인조차 없이 클릭 즉시 전송이었다 — 눌러 보고 무슨 버튼인지 알아보려던
 * 사람이 제보를 보내게 된다. 그래서 마지막에 "제보하시겠어요?"를 한 번 세운다. 취소하면
 * 입력이 그대로 남은 폼으로 돌아온다 — 확인을 무르는 대가로 다시 쓰게 만들지 않는다.
 *
 * 깊이가 fast·medium이면 **제보 전에 먼저** 말한다: 이 결과는 빠른 설정으로 만든 것이고
 * 깊이를 올리면 나아질 수 있다고. 품질 불만의 상당수가 거기서 끝나는데도 화면이 그 사실을
 * 알려 주지 않으면, 고칠 수 있는 문제가 제보로만 쌓인다.
 */
export function buildReportSheet(opts: ReportSheetOpts): { el: HTMLDivElement } {
  const el = h('div', { className: 'ey-report-sheet' });

  // 폼 컨트롤은 단계가 바뀌어도 **같은 노드를 다시 붙인다** — 확인 단계에서 취소해 돌아왔을 때
  // 입력이 남아 있어야 한다(새로 만들면 매번 빈 칸이 된다)
  const category = h('select', { className: 'ey-select' }, ...([
    ['', t('overlay.feedback.catNone')],
    ['timing', t('overlay.feedback.catTiming')],
    ['pronunciation', t('overlay.feedback.catPron')],
    ['lyrics', t('overlay.feedback.catLyrics')],
    ['other', t('overlay.feedback.catOther')],
  ] as [string, string][]).map(([value, label]) => h('option', { text: label, attrs: { value } })));
  const comment = h('input', {
    className: 'ey-input',
    attrs: { placeholder: t('panels.report.commentPh'), maxlength: '500' },
  });
  const status = h('div', { className: 'ey-report-status' });

  const renderForm = () => {
    el.replaceChildren(
      h('div', { className: 'ey-report-title', text: t('panels.report.title') }),
      h('div', { className: 'ey-report-row' }, category),
      h('div', { className: 'ey-report-row' }, comment),
      h('div', { className: 'ey-report-actions' },
        h('button', {
          className: 'ey-secondary-btn', text: t('panels.report.cancel'),
          attrs: { type: 'button' }, on: { click: () => opts.onCancel() },
        }),
        h('button', {
          className: 'ey-primary-btn ey-report-danger', text: t('panels.report.send'),
          attrs: { type: 'button' }, on: { click: () => renderConfirm() },
        }),
      ),
      status,
    );
  };

  const renderConfirm = () => {
    const send = h('button', {
      className: 'ey-primary-btn ey-report-danger',
      text: t('panels.report.confirmSend'),
      attrs: { type: 'button' },
    });
    send.addEventListener('click', () => {
      send.disabled = true;
      status.textContent = t('panels.rating.sending');
      void opts.onSubmit(category.value || undefined, comment.value.trim() || undefined).then(ok => {
        if (ok) {
          status.textContent = t('panels.report.thanks');
          window.setTimeout(() => opts.onDone?.(), 1400);
          return;
        }
        // 실패 — 입력을 살린 채 폼으로. status는 같은 노드라 다시 그려도 사유가 남는다
        renderForm();
        status.textContent = t('panels.report.failed');
      });
    });
    el.replaceChildren(
      h('div', { className: 'ey-report-confirm' },
        h('div', { className: 'ey-report-title', text: t('panels.report.confirmTitle') }),
        h('div', { className: 'ey-report-confirm-body', text: t('panels.report.confirmBody') }),
        h('div', { className: 'ey-report-actions' },
          h('button', {
            className: 'ey-secondary-btn', text: t('panels.report.cancel'),
            attrs: { type: 'button' }, on: { click: () => renderForm() },
          }),
          send,
        ),
      ),
      status,
    );
  };

  const renderDepthHint = (depth: 'fast' | 'medium') => {
    el.replaceChildren(
      h('div', { className: 'ey-report-hint' },
        h('div', { className: 'ey-report-hint-text', text: t('panels.report.depthHint', [depthLabel(depth)]) }),
        h('div', { className: 'ey-report-hint-actions' },
          opts.onDepthUpgrade
            ? h('button', {
              className: 'ey-primary-btn', text: t('panels.report.depthUpgrade'),
              attrs: { type: 'button' },
              on: { click: () => { opts.onDepthUpgrade?.(); opts.onDone?.(); } },
            })
            : null,
          h('button', {
            className: 'ey-secondary-btn', text: t('panels.report.depthAnyway'),
            attrs: { type: 'button' }, on: { click: () => renderForm() },
          }),
        ),
      ),
    );
  };

  if (opts.depth === 'fast' || opts.depth === 'medium') renderDepthHint(opts.depth);
  else renderForm();
  return { el };
}

export interface WrongLyricsConfirmOpts {
  /** 지금 붙어 있는 가사의 제목 — 무엇을 제보하는지 눈으로 확인시킨다 */
  matchedTitle?: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * "이 가사가 아니에요" 확인 팝오버.
 *
 * 이 버튼은 가사 바로 위에 상시 떠 있어 오클릭 거리가 가장 짧은데도 확인이 없었다 —
 * 누르는 즉시 제보가 나갔다. 흐름의 목적은 제보 자체가 아니라 **올바른 가사를 찾는 것**
 * 이므로, 확인을 한 단계 넣어도 사용자가 잃는 것은 없다.
 */
export function buildWrongLyricsConfirm(opts: WrongLyricsConfirmOpts): { el: HTMLDivElement } {
  const title = opts.matchedTitle?.trim();
  const el = h('div', { className: 'ey-confirm-pop' },
    h('div', { className: 'ey-confirm-text', text: t('panels.wrongLyrics.title') }),
    title ? h('div', { className: 'ey-confirm-target', text: t('panels.wrongLyrics.matched', [title]) }) : null,
    h('div', { className: 'ey-confirm-body', text: t('panels.wrongLyrics.body') }),
    h('div', { className: 'ey-confirm-actions' },
      h('button', {
        className: 'ey-secondary-btn', text: t('panels.wrongLyrics.cancel'),
        attrs: { type: 'button' }, on: { click: () => opts.onCancel() },
      }),
      h('button', {
        className: 'ey-primary-btn ey-report-danger', text: t('panels.wrongLyrics.confirm'),
        attrs: { type: 'button' }, on: { click: () => opts.onConfirm() },
      }),
    ),
  );
  return { el };
}

// ── 공지 시트 ────────────────────────────────────────────────────

/** 마지막으로 본 공지 id — 판정 규칙은 probeNotices 주석 참고 */
const NOTICES_SEEN_KEY = 'ey_notices_seen';

export interface NoticesProbe {
  /** 이 서버가 공지 기능을 갖고 있는가 — false면 호출부는 **진입점 자체를 숨긴다** */
  available: boolean;
  unread: boolean;
  count: number;
}

async function readSeenNoticeId(): Promise<string> {
  try {
    const stored = await chrome.storage.local.get(NOTICES_SEEN_KEY);
    const value = stored[NOTICES_SEEN_KEY] as unknown;
    // 서버 id는 정수다 — 숫자도 받아 문자열로 정규화한다. 예전 코드는 string만 인정해
    // 저장된 숫자를 영영 못 읽었고, 읽음 표시가 새로고침마다 초기화됐다(실사용 재현).
    // 숫자 허용은 그 시절 저장분(숫자)을 든 사용자 구제이기도 하다.
    if (typeof value === 'string') return value;
    if (typeof value === 'number') return String(value);
    return '';
  } catch {
    return '';
  }
}

async function markNoticesSeen(notices: ServerNotice[]): Promise<void> {
  if (notices.length === 0) return;
  try {
    await chrome.storage.local.set({ [NOTICES_SEEN_KEY]: String(notices[0].id) });
  } catch { /* 표시용 상태다 — 못 적으면 점이 한 번 더 뜰 뿐이다 */ }
}

/**
 * 공지가 있는지 + 안 읽은 것이 있는지.
 *
 * "안 읽음"은 **최신 공지 id 하나**를 마지막으로 본 것과 비교해 판정한다. id가 문자열이라
 * 크기 비교(더 큰 id = 더 새것)가 성립하지 않으므로 "맨 위 공지가 지난번 본 것과 다른가"만
 * 본다 — 서버가 최신순으로 준다는 전제 위에서만 맞다. 서버가 순서를 바꾸면 이미 읽은
 * 공지에 점이 한 번 더 뜰 수 있는데, 그건 점 하나를 잘못 켜는 비용이고 반대(새 공지를
 * 놓치는 것)보다 훨씬 싸다.
 *
 * 실패·구버전 서버는 available=false다 — 호출부는 **오류를 보여주지 말고** 진입점을 통째로
 * 숨긴다(없는 기능을 고장난 기능처럼 보이게 하지 않는다).
 */
export async function probeNotices(): Promise<NoticesProbe> {
  const res = await sendPanelMessage<NoticesResponse>({ type: 'NOTICES_GET' });
  const notices = res.data?.notices;
  if (res.error || !Array.isArray(notices)) return { available: false, unread: false, count: 0 };
  if (notices.length === 0) return { available: true, unread: false, count: 0 };
  const seen = await readSeenNoticeId();
  return { available: true, unread: String(notices[0].id) !== seen, count: notices.length };
}

export interface NoticesSheetOpts {
  onBack?: () => void;
  /** 목록을 '본 것'으로 표시한 직후 — 호스트가 안 읽음 점을 끄는 자리 */
  onSeen?: () => void;
}

/** 등급 이름 — 색만으로는 등급을 읽을 수 없는 사람이 있다 */
function noticeLevelLabel(level: ServerNotice['level']): string {
  if (level === 'critical') return t('panels.notices.levelCritical');
  if (level === 'warning') return t('panels.notices.levelWarning');
  return t('panels.notices.levelInfo');
}

/**
 * 서버 공지함 — 등급색 왼쪽 띠 + 제목 + 본문(줄바꿈 보존) + 상대 날짜.
 * 본문은 서버가 준 평문이라 textContent로만 넣는다(HTML을 심을 자리가 아니다).
 */
export function buildNoticesSheet(opts: NoticesSheetOpts = {}): { el: HTMLDivElement } {
  const list = h('div', { className: 'ey-notices-list' },
    h('div', { className: 'ey-state-sub', text: t('panels.notices.loading') }));
  const el = h('div', { className: 'ey-state ey-notices' },
    buildSheetHead(t('panels.notices.title'), opts.onBack),
    list,
  );

  void sendPanelMessage<NoticesResponse>({ type: 'NOTICES_GET' }).then(res => {
    const notices = res.data?.notices;
    if (res.error || !Array.isArray(notices)) {
      // 여기까지 들어온 사용자에게는 말 없는 빈 화면이 더 나쁘다. 다만 서버 오류 문구를
      // 그대로 옮기지는 않는다 — 대부분 '이 서버엔 그 기능이 없다'는 뜻이다
      list.replaceChildren(h('div', { className: 'ey-state-sub', text: t('panels.notices.unavailable') }));
      return;
    }
    if (notices.length === 0) {
      list.replaceChildren(h('div', { className: 'ey-state-sub', text: t('panels.notices.empty') }));
      return;
    }
    list.replaceChildren(...notices.map(notice => {
      const at = Date.parse(notice.created_at);
      return h('div', { className: `ey-notice-item ey-notice-${notice.level}` },
        h('div', { className: 'ey-notice-head' },
          h('span', { className: 'ey-notice-level', text: noticeLevelLabel(notice.level) }),
          h('span', { className: 'ey-notice-title', text: notice.title }),
          Number.isFinite(at) ? h('span', { className: 'ey-notice-date', text: relativeDate(at) }) : null,
        ),
        // 줄바꿈은 CSS(white-space: pre-wrap)가 살린다 — 서버가 넣은 개행이 곧 문단 구분이다
        h('div', { className: 'ey-notice-body', text: notice.body }),
      );
    }));
    void markNoticesSeen(notices).then(() => opts.onSeen?.());
  });

  return { el };
}

/** next_reset_at(UTC ISO)을 사용자 로컬 시각 HH:MM으로 — debug-panel.ts formatHHMM과
 *  같은 표기 규칙(로컬 타임존, 24시간제 아님 — 브라우저 로케일에 맡긴다). */
function formatResetTime(isoUtc: string): string {
  const d = new Date(isoUtc);
  if (Number.isNaN(d.getTime())) return isoUtc;
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ── 기여 · 쿼터 시트 ─────────────────────────────────────────────

export interface ContributionSheetOpts {
  /** 지금 보고 있는 영상 — 남은 한도는 영상 기준 조회라 없으면 한도 블록을 건너뛴다 */
  videoId?: string | null;
  /** 목록 한 줄을 눌렀을 때. 실제 이동 방법(새 탭·같은 탭)은 호스트가 정한다 */
  onOpenVideo?: (videoId: string) => void;
  onBack?: () => void;
}

/** STATS_VIEWS 한 번에 보낼 수 있는 최대 건수 — 서버 계약 */
const VIEWS_CHUNK = 100;

async function readContributions(): Promise<ContribEntry[]> {
  try {
    const stored = await chrome.storage.local.get(CONTRIB_STORAGE_KEY);
    const list = stored[CONTRIB_STORAGE_KEY] as ContribEntry[] | undefined;
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

/**
 * "내가 만든 N곡을 M명이 봤어요" + 오늘 남은 횟수 + 기여 목록.
 *
 * 세 조각의 출처가 서로 다르다는 것이 이 화면의 전부다: 기여 목록은 **이 브라우저에만**
 * 있고(서버엔 계정이 없어 '누가 만들었나'가 없다), 조회수와 남은 한도는 서버에만 있다.
 * 그래서 목록은 즉시 그리고 서버 값은 도착하는 대로 채운다 — 서버가 없거나 낡아 못 주면
 * 그 블록만 사라지고 나머지는 그대로 쓸 수 있다.
 *
 * 같은 영상을 여러 번 재생성하면 항목이 여럿 쌓인다 — "N곡"은 **영상 단위로** 세고(중복
 * 제거) 조회수도 영상당 한 번만 더한다. 아니면 재생성이 기여를 부풀린다.
 */
export function buildContributionSheet(opts: ContributionSheetOpts = {}): { el: HTMLDivElement } {
  const quota = h('div', { className: 'ey-contrib-quota' });
  quota.style.display = 'none'; // 서버가 한도를 강제하는 배포에서만 나타난다
  const headline = h('div', { className: 'ey-contrib-headline' });
  const list = h('div', { className: 'ey-contrib-list' },
    h('div', { className: 'ey-state-sub', text: t('panels.contrib.loading') }));

  const el = h('div', { className: 'ey-state ey-contrib' },
    buildSheetHead(t('panels.contrib.title'), opts.onBack),
    quota,
    headline,
    list,
    // 흐릿하게 두지 않는다 — 로컬 전용이라는 사실이 이 화면의 약속 그 자체다
    h('div', { className: 'ey-contrib-privacy', text: t('panels.contrib.privacy') }),
  );

  void (async () => {
    const entries = await readContributions();
    // 최신이 뒤에 쌓이므로(ContribEntry 규약) 뒤에서부터 훑어 영상당 최신 한 건만 남긴다
    const latest = new Map<string, ContribEntry>();
    for (let i = entries.length - 1; i >= 0; i--) {
      const entry = entries[i];
      if (!latest.has(entry.videoId)) latest.set(entry.videoId, entry);
    }
    const unique = [...latest.values()].sort((a, b) => b.completedAt - a.completedAt);

    if (unique.length === 0) {
      list.replaceChildren(h('div', { className: 'ey-state-sub', text: t('panels.contrib.empty') }));
    } else {
      // 조회수가 오기 전에도 곡 수는 이미 안다 — 먼저 말하고 나중에 사람 수를 덧붙인다
      headline.textContent = t('panels.contrib.headlineNoViews', [String(unique.length)]);
      const metaEls = new Map<string, HTMLElement>();
      list.replaceChildren(...unique.map(entry => {
        const meta = h('span', { className: 'ey-contrib-item-meta', text: relativeDate(entry.completedAt) });
        const row = h('button', {
          className: 'ey-contrib-item',
          attrs: { type: 'button', title: t('panels.contrib.openTitle') },
          on: { click: () => opts.onOpenVideo?.(entry.videoId) },
        },
          h('span', { className: 'ey-contrib-item-title', text: entry.title || entry.videoId }),
          meta,
          entry.depth
            ? h('span', { className: `ey-contrib-depth ${entry.depth}`, text: depthLabel(entry.depth) })
            : null,
        );
        if (!opts.onOpenVideo) row.disabled = true; // 이동할 방법이 없으면 눌리는 척하지 않는다
        metaEls.set(entry.videoId, meta);
        return row;
      }));

      // 조회수 — 100건씩 끊어 묻는다(서버 계약). 일부 청크만 실패해도 받은 만큼은 보여준다
      const ids = unique.map(e => e.videoId);
      const views: Record<string, number> = {};
      let gotAny = false;
      for (let i = 0; i < ids.length; i += VIEWS_CHUNK) {
        const res = await sendPanelMessage<ViewStatsResponse>({
          type: 'STATS_VIEWS', payload: { videoIds: ids.slice(i, i + VIEWS_CHUNK) },
        });
        const chunk = res.data?.views;
        if (!chunk) continue;
        gotAny = true;
        Object.assign(views, chunk);
      }
      if (gotAny) {
        const total = ids.reduce((sum, id) => sum + (views[id] ?? 0), 0);
        headline.textContent = t('panels.contrib.headline', [String(unique.length), String(total)]);
        for (const [videoId, meta] of metaEls) {
          const count = views[videoId];
          if (count === undefined) continue;
          meta.textContent = `${meta.textContent} · ${t('panels.contrib.rowViews', [String(count)])}`;
        }
      }
    }

    // 남은 한도 — 영상이 있어야 물을 수 있다. 강제하지 않는 배포도 블록은 띄우고
    // "무제한 (사용 N회)"로 표시한다 — 숨기면 사용자는 기능이 고장난 줄 안다(운영자
    // 지시 2026-08-04: "무제한이어도 무제한이라고 뜨게 해야지").
    if (!opts.videoId) return;
    const res = await sendPanelMessage<LimitsResponse>({
      type: 'LIMITS_GET', payload: { videoId: opts.videoId },
    });
    const limits = res.data;
    if (!limits) return;
    // 각 버킷은 행 하나(라벨+남은/전체) + 있으면 회복 안내 한 줄(작게)을 낸다 — 회복 안내는
    // next_reset_at이 null(이 한도를 세션 내 아직 안 씀)이면 통째로 생략한다(팀 지시).
    const bucket = (label: string, data: LimitBucket): HTMLElement[] => {
      const row = h('div', { className: 'ey-contrib-quota-row' },
        h('span', { text: label }),
        h('span', {
          className: `ey-contrib-quota-val${limits.enforced && data.remaining === 0 ? ' out' : ''}`,
          text: limits.enforced
            ? t('panels.contrib.quotaValue', [String(data.remaining), String(data.limit)])
            : t('panels.contrib.quotaUnlimited', [String(data.used)]),
        }),
      );
      const nextResetAt = data.next_reset_at ?? null;
      if (!nextResetAt) return [row];
      const note = h('div', {
        className: 'ey-settings-note',
        text: t('panels.contrib.quotaNextReset', [formatResetTime(nextResetAt)]),
      });
      return [row, note];
    };
    quota.replaceChildren(
      h('div', { className: 'ey-contrib-quota-title', text: t('panels.contrib.quotaTitle') }),
      ...bucket(t('panels.contrib.quotaGenerate'), limits.generate),
      // link·upgrade는 구서버엔 없는 additive 필드(LimitsResponse 참고) — 응답에 없으면
      // (undefined) 그 줄 자체를 생략한다, 빈 값으로 그리지 않는다.
      ...(limits.link ? bucket(t('panels.contrib.quotaLink'), limits.link) : []),
      // upgrade는 이제 generate와 분리된 자기만의 카운터(daily_upgrade_limit, types.ts
      // 참고) — 2026-08-04 운영자 정정으로 "공유돼요" 안내는 더 이상 사실이 아니라 제거.
      ...(limits.upgrade ? bucket(t('panels.contrib.quotaUpgrade'), limits.upgrade) : []),
      ...bucket(t('panels.contrib.quotaDestructive'), limits.destructive),
      h('div', {
        className: 'ey-settings-note',
        text: t('panels.contrib.quotaWindow', [String(limits.window_hours)]),
      }),
    );
    quota.style.display = '';
  })();

  return { el };
}
