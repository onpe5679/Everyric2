import type {
  BgRequest, EveryricSegment, LyricLine, MessageResponse, SyncDebugMeta, SyncPreviousVersion,
  SyncVersionDetail, SyncVersionSummary, SyncVersionsResponse,
} from '../types';
import { t } from '../lib/i18n';
import { h } from './dom';

/**
 * 곡 전체 디버그 패널 — 라인 집중 뷰(pip.ts renderTimingLanes)의 자매 기능이다.
 * 그쪽이 "지금 재생 중인 한 줄"을 확대해 보여준다면, 이 패널은 "곡 전체를 한눈에" 훑어
 * 원문 vs heard(CTC가 실제로 들은 것)를 전수 대비하는 용도다. 패널은 UI만 만들고 상태를
 * 갖지 않는다 — 시크는 호출부(overlay.ts)가 콜백으로 주입한다.
 */

/** 3색 정렬 신뢰도 등급 — pip.ts confBucketColor·overlay.css .ey-conf-*와 같은 경계값(재사용) */
function confGrade(conf: number | undefined): { cls: string; label: string } | null {
  if (conf == null) return null;
  if (conf < 1e-4) return { cls: 'ey-conf-low', label: t('debugPanel.confLow') };
  if (conf < 2e-2) return { cls: 'ey-conf-mid', label: t('debugPanel.confMid') };
  return { cls: 'ey-conf-ok', label: t('debugPanel.confOk') };
}

/** updateDebug의 시각 표기(초 2자리)와 같은 형식으로 통일 */
function fmtTime(t: number | null): string {
  return t === null ? '-' : `${t.toFixed(2)}s`;
}

/**
 * 엔진 정체 요약 한 줄 — 어느 스택(engine_version)·어느 깊이(routing.route)가 이 싱크를
 * 만들었는지. depth=/lang= 토큰은 언어 중립 디버그 표기라 i18n하지 않는다. lang 뒤의
 * `*`는 라벨이 비어 가사 문자 계열로 판정했다는 표시(서버 language_source=script_census).
 */
function engineSummary(meta: SyncDebugMeta | null | undefined): string | null {
  if (!meta) return null;
  const parts: string[] = [];
  if (meta.engine_version !== undefined) {
    parts.push(t('debugPanel.engine', [meta.engine_version ?? t('debugPanel.engineLegacy')]));
  }
  if (meta.engine_variant) parts.push(meta.engine_variant);
  if (meta.alignment_text) parts.push(meta.alignment_text);
  const r = meta.routing;
  if (r?.route) parts.push(`depth=${r.route}`);
  if (r?.language) parts.push(`lang=${r.language}${r.language_source === 'script_census' ? '*' : ''}`);
  return parts.length > 0 ? parts.join(' · ') : null;
}

/** 곡 단위 자막 스캐폴드 요약 한 줄 — 없으면(구서버·미배선·해당 없음) null로 생략된다 */
function scaffoldSummary(meta: SyncDebugMeta | null | undefined): string | null {
  const sc = meta?.caption_scaffold;
  if (!sc) return null;
  if (sc.applied) {
    const src = sc.sources ?? {};
    const total = (src.caption ?? 0) + (src.interp ?? 0) + (src.kept ?? 0);
    const matchPct = total > 0 ? Math.round(((src.caption ?? 0) / total) * 100) : 0;
    return t('debugPanel.scaffoldApplied', [
      String(sc.moved ?? 0), String(matchPct),
      String(src.caption ?? 0), String(src.interp ?? 0), String(src.kept ?? 0),
    ]);
  }
  // not_collapsed(정상 곡이라 애초에 시도조차 안 함)는 소음이라 생략 — pip.ts 디버그 오버레이와 같은 판단
  if (sc.skipped && sc.skipped !== 'not_collapsed') {
    return t('debugPanel.scaffoldSkipped', [sc.skipped]);
  }
  return null;
}

export interface DebugPanelRefs {
  el: HTMLDivElement;
}

/**
 * 고스트 비교 Δ 계산 — 직전 세대든 저장된 특정 버전이든 같은 로직으로 적용한다. 줄 텍스트가
 * 같은 라인만 대응시킨다(재처리 = 같은 가사의 재정렬일 때만 A/B가 성립). 반환값은 대응된
 * 줄 수 — 호출부가 각자의 문구(출처마다 다르다)로 상태 줄을 조립하는 데 쓴다. 이 함수 자체는
 * "누구와 비교 중인지"를 몰라도 된다 — 그래서 "이전 세대와 비교"·"깊이·버전 비교" 두 버튼이
 * 이 하나를 공유한다.
 */
function applyGhostDeltas(
  lines: LyricLine[], deltaEls: HTMLSpanElement[], timestamps: EveryricSegment[],
): number {
  let matched = 0;
  lines.forEach((line, i) => {
    const old = timestamps[i];
    const deltaEl = deltaEls[i];
    if (!deltaEl) return;
    if (!old || old.text !== line.text || line.time === null) {
      deltaEl.textContent = 'Δ—';
      deltaEl.classList.remove('big');
      return;
    }
    matched++;
    const d = line.time - old.start;
    deltaEl.textContent = `Δ${d >= 0 ? '+' : ''}${d.toFixed(2)}s`;
    // 0.15s 이내는 흐리게(사실상 동일), 그 밖은 또렷하게 — 큰 이동만 눈에 띄게
    deltaEl.classList.toggle('big', Math.abs(d) > 0.15);
  });
  return matched;
}

function clearGhostDeltas(deltaEls: HTMLSpanElement[]): void {
  deltaEls.forEach(el => { el.textContent = ''; el.classList.remove('big'); });
}

/** 서버가 준 원본 문자열(타임존 표기 없는 UTC)을 Date로 — content.ts formatSyncCreated와
 *  같은 UTC 판정 규칙(Z·오프셋이 없으면 UTC로 간주)을 쓴다. */
function parseServerUtc(raw: string): Date | null {
  const text = raw.trim();
  const hasZone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(text);
  const at = new Date(hasZone ? text : `${text.replace(' ', 'T')}Z`);
  return Number.isNaN(at.getTime()) ? null : at;
}

/** 로컬 HH:MM만 — 라인 목록 등 같은 날짜 안에서만 의미가 있는 자리에 쓴다. */
function formatHHMM(raw: string | null | undefined): string {
  if (!raw) return '-';
  const at = parseServerUtc(raw);
  if (!at) return '-';
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(at.getHours())}:${pad(at.getMinutes())}`;
}

/** 버전 목록 항목의 생성 시각 — HH:MM만으론 여러 날에 걸친 재생성을 구분할 수 없다
 *  (운영자 지적: "디버그 비교 기능 직관적이지 않다" — 목록 항목이 뭔지 한눈에 안 들어옴).
 *  월/일을 앞에 붙여 날짜 모호성을 없앤다. */
function formatVersionWhen(raw: string | null | undefined): string {
  if (!raw) return '-';
  const at = parseServerUtc(raw);
  if (!at) return '-';
  return `${at.getMonth() + 1}/${at.getDate()} ${formatHHMM(raw)}`;
}

/** 버전 목록의 깊이 배지 — 서버 라벨(fast/medium/heavy)을 안내 없이 그대로 노출하면
 *  뜻을 알 수 없다(같은 운영자 지적). 모르는 값(null/undefined, 구세대 버전)은 '미상'으로
 *  — 지어내지 않는다. */
function depthBadgeLabel(depth: 'fast' | 'medium' | 'heavy' | null | undefined): string {
  if (depth === 'fast') return t('debugPanel.depth.fast');
  if (depth === 'medium') return t('debugPanel.depth.medium');
  if (depth === 'heavy') return t('debugPanel.depth.heavy');
  return t('debugPanel.depth.unknown');
}

/**
 * content.ts의 sendToBackground와 같은 목적의 최소판 — 이 모듈은 SYNC_VERSIONS·
 * SYNC_VERSION_GET을 overlay 콜백을 거치지 않고 직접 부른다(팀 배선: 이 두 요청은
 * videoId 하나만 있으면 끝나서 overlay.ts·content.ts에 새 콜백을 늘릴 이유가 없다).
 * orphan 탭(확장 리로드 후 남은 content script) 가드만 있으면 충분하다 — 실패는
 * 패널의 상태 줄이 대신 보여준다(content.ts처럼 영구 알림을 띄우지 않는다).
 */
async function sendMessageDirect<T>(message: BgRequest): Promise<MessageResponse<T>> {
  if (!chrome.runtime?.id) {
    return { error: 'extension context invalidated' };
  }
  try {
    return await chrome.runtime.sendMessage(message) as MessageResponse<T>;
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * @param onSeek 이미 SEEK_INTO_LINE_SEC 같은 보정이 필요하면 호출부가 콜백 안에서 적용한다 —
 *   이 함수는 line.time을 그대로 넘긴다.
 * @param loadPrevious 있으면 "이전 세대와 비교" 버튼을 그린다(기존과 동일한 동작).
 * @param videoId SYNC_VERSIONS·SYNC_VERSION_GET을 직접 부르는 데 필요하다 — 없으면(호출부
 *   미배선) "깊이·버전 비교" 버튼 자체를 생략한다. "이전 세대와 비교"는 이 값과 무관하게
 *   기존과 동일하게 동작한다.
 */
export function buildDebugPanel(
  lines: LyricLine[],
  debugMeta: SyncDebugMeta | null | undefined,
  onSeek: (time: number) => void,
  loadPrevious?: () => Promise<SyncPreviousVersion | null>,
  videoId?: string | null,
): DebugPanelRefs {
  const el = h('div', { className: 'ey-debug-panel' });

  const engine = engineSummary(debugMeta);
  if (engine) {
    el.append(h('div', {
      className: 'ey-debug-panel-summary',
      text: engine,
      // lang 뒤 `*`의 의미는 툴팁으로 — 요약 한 줄에 다 적으면 소음이다
      title: t('debugPanel.engineSummaryTitle'),
    }));
  }

  const summary = scaffoldSummary(debugMeta);
  if (summary) {
    el.append(h('div', { className: 'ey-debug-panel-summary', text: summary }));
  }

  if (lines.length === 0) {
    el.append(h('div', { className: 'ey-debug-panel-empty', text: t('debugPanel.empty') }));
    return { el };
  }

  // 고스트 비교 Δ 자리 — 두 비교 경로(이전 세대·특정 버전) 모두 같은 배열에 쓴다
  const deltaEls: HTMLSpanElement[] = [];

  // A/B 고스트 비교 바 — 서버가 보관한 다른 세대(직전 세대 또는 저장된 특정 버전)의 줄
  // 시각과 지금 화면의 줄 시각을 라인 단위 Δ로 대비한다. 줄 텍스트가 같은 라인만 비교하고
  // (재생성 = 같은 가사의 재정렬일 때만 A/B가 성립), 다른 줄은 Δ— 로 남긴다.
  //
  // "이전 세대와 비교"(loadPrevious)와 "깊이·버전 비교"(videoId)는 activeKey 하나로
  // "지금 뭘 비교 중인가"를 함께 추적한다 — 같은 항목을 다시 누르면 비교가 해제되고,
  // 다른 항목을 누르면 비교 대상이 바뀐다(Δ는 항상 하나의 출처만 표시).
  if (loadPrevious || videoId) {
    const status = h('span', { className: 'ey-debug-compare-status' });
    let activeKey: string | null = null; // 'prev' | 버전 id | null
    let prevBtn: HTMLButtonElement | null = null;
    let versionsListEl: HTMLDivElement | null = null;
    const versionRowEls = new Map<string, HTMLButtonElement>();

    const paintActive = (): void => {
      if (prevBtn) prevBtn.classList.toggle('active', activeKey === 'prev');
      versionRowEls.forEach((rowEl, id) => {
        const active = activeKey === id;
        // .ey-debug-row에는 .active 스타일이 없어(그 클래스는 .ey-btn 전용) 인라인으로 칠한다
        rowEl.style.background = active ? 'var(--ey-accent-soft)' : '';
        rowEl.style.color = active ? 'var(--ey-accent)' : '';
      });
    };

    const clearComparison = (): void => {
      activeKey = null;
      clearGhostDeltas(deltaEls);
      status.textContent = '';
      paintActive();
    };

    const bar = h('div', { className: 'ey-debug-compare-bar' });

    if (loadPrevious) {
      const btn = h('button', {
        className: 'ey-btn',
        attrs: { type: 'button' },
        text: t('debugPanel.comparePrev'),
        on: {
          click: () => {
            if (activeKey === 'prev') { clearComparison(); return; }
            btn.disabled = true;
            status.textContent = '…';
            void loadPrevious().then(prev => {
              btn.disabled = false; // 재시도·재비교 모두 가능하게 항상 되살린다
              if (!prev?.found || !prev.timestamps || prev.timestamps.length === 0) {
                activeKey = null;
                status.textContent = t('debugPanel.comparePrevNone');
                paintActive();
                return;
              }
              activeKey = 'prev';
              const matched = applyGhostDeltas(lines, deltaEls, prev.timestamps);
              status.textContent = t('debugPanel.comparePrevLoaded', [
                prev.engine_version ?? t('debugPanel.engineLegacy'),
                String(matched), String(lines.length),
              ]);
              paintActive();
            }).catch(() => {
              btn.disabled = false;
              activeKey = null;
              status.textContent = t('debugPanel.comparePrevNone');
              paintActive();
            });
          },
        },
      }) as HTMLButtonElement;
      prevBtn = btn;
      bar.append(btn);
    }

    if (videoId) {
      const vid = videoId; // 클로저 안에서 string으로 좁혀 매 사용마다 null 체크를 피한다
      const list = h('div', { className: 'ey-debug-panel-list' });
      list.style.display = 'none';
      versionsListEl = list;
      let versionsLoaded: SyncVersionSummary[] | null = null; // 재조회 없이 펼침/접힘만 토글하기 위한 캐시

      const renderVersionRows = (versions: SyncVersionSummary[]): void => {
        versionRowEls.clear();
        list.replaceChildren();
        // 이 기능이 "미리보기·되돌리기"로 오해되지 않게 항상 보이는 안내를 먼저 둔다
        // (툴팁 하나에만 의존하면 안 읽힌다 — 운영자 지적: "디버그 비교 기능 직관적이지 않다").
        list.append(h('div', { className: 'ey-debug-panel-empty', text: t('debugPanel.versionsHint') }));
        if (versions.length === 0) {
          list.append(h('div', {
            className: 'ey-debug-panel-empty', text: t('debugPanel.compareVersionsEmpty'),
          }));
          return;
        }
        for (const v of versions) {
          // 지금 화면이 이 버전과 같은 스택으로 만들어졌는지 — engine_version 일치로만
          // 판정한다(서버가 버전별 고유 id를 debugMeta에 안 실어 주므로 이게 가진 최선의
          // 신호다. 같은 스택으로 여러 번 재생성했으면 여러 항목이 동시에 "지금 보는
          // 버전"으로 표시될 수 있다 — 완전 정확한 식별은 아니라는 한계를 인지하고 쓴다).
          const isCurrent = Boolean(debugMeta?.engine_version) && debugMeta?.engine_version === v.engine_version;
          const qualityText = v.quality_score != null
            ? t('debugPanel.qualityLabel', [v.quality_score.toFixed(3)])
            : null;
          const textCol = h('div', { className: 'ey-debug-row-text' },
            h('div', { className: 'ey-debug-row-orig', text: v.engine_version ?? t('debugPanel.engineLegacy') }));
          if (qualityText) textCol.append(h('div', { className: 'ey-debug-row-heard', text: qualityText }));
          if (isCurrent) textCol.append(h('div', { className: 'ey-debug-row-labels', text: t('debugPanel.currentVersionTag') }));
          const rowEl = h('button', {
            className: 'ey-debug-row',
            attrs: { type: 'button' },
            title: t('debugPanel.compareVersionRowTitle'),
          },
            h('span', { className: 'ey-debug-row-time', text: formatVersionWhen(v.created_at) }),
            h('span', {
              className: 'ey-debug-row-chip', text: depthBadgeLabel(v.depth),
              title: t('debugPanel.depthBadgeTitle'),
            }),
            textCol,
          ) as HTMLButtonElement;
          rowEl.addEventListener('click', () => {
            if (activeKey === v.id) { clearComparison(); return; }
            rowEl.disabled = true;
            status.textContent = '…';
            void sendMessageDirect<SyncVersionDetail>({
              type: 'SYNC_VERSION_GET', payload: { videoId: vid, resultId: v.id },
            }).then(res => {
              rowEl.disabled = false;
              const detail = res.data;
              if (!detail?.timestamps || detail.timestamps.length === 0) {
                activeKey = null;
                status.textContent = t('debugPanel.compareVersionNone');
                paintActive();
                return;
              }
              activeKey = v.id;
              const matched = applyGhostDeltas(lines, deltaEls, detail.timestamps);
              status.textContent = t('debugPanel.compareVersionLoaded', [
                detail.engine_version ?? v.engine_version ?? t('debugPanel.engineLegacy'),
                detail.depth ?? v.depth ?? '?',
                String(matched), String(lines.length),
              ]);
              paintActive();
            }).catch(() => {
              rowEl.disabled = false;
              activeKey = null;
              status.textContent = t('debugPanel.compareVersionNone');
              paintActive();
            });
          });
          versionRowEls.set(v.id, rowEl);
          list.append(rowEl);
        }
        paintActive();
      };

      const versionsBtn = h('button', {
        className: 'ey-btn',
        attrs: { type: 'button' },
        text: t('debugPanel.compareVersions'),
        on: {
          click: () => {
            if (versionsLoaded) {
              // 이미 불러온 목록 — 펼침/접힘만 토글한다(재조회 없음)
              const show = list.style.display === 'none';
              list.style.display = show ? '' : 'none';
              versionsBtn.classList.toggle('active', show);
              return;
            }
            versionsBtn.disabled = true;
            void sendMessageDirect<SyncVersionsResponse>({
              type: 'SYNC_VERSIONS', payload: { videoId: vid },
            }).then(res => {
              versionsBtn.disabled = false;
              const versions = res.data?.versions ?? [];
              versionsLoaded = versions;
              renderVersionRows(versions);
              list.style.display = '';
              versionsBtn.classList.add('active');
              if (!res.data) status.textContent = t('debugPanel.compareVersionsFailed');
            }).catch(() => {
              versionsBtn.disabled = false;
              versionsLoaded = [];
              renderVersionRows([]);
              list.style.display = '';
              status.textContent = t('debugPanel.compareVersionsFailed');
            });
          },
        },
      }) as HTMLButtonElement;
      bar.append(versionsBtn);
    }

    bar.append(status);
    el.append(bar);
    if (versionsListEl) el.append(versionsListEl);
  }

  const list = h('div', { className: 'ey-debug-panel-list' });
  for (const line of lines) {
    const dbg = line.debug;
    const grade = confGrade(line.confidence);

    const chip = h('span', {
      className: `ey-debug-row-chip${grade ? ` ${grade.cls}` : ''}`,
      text: grade ? grade.label : '—',
    });

    const textCol = h('div', { className: 'ey-debug-row-text' },
      h('div', { className: 'ey-debug-row-orig', text: line.text }));
    // heard(CTC가 실제로 들은 것) — 원문과 구분되게 흐린 색으로, 있을 때만
    if (dbg?.heard) {
      textCol.append(h('div', { className: 'ey-debug-row-heard', text: t('debugPanel.heardPrefix', [dbg.heard]) }));
    }
    // fixes 라벨 + 심판 개입(⚖) — 한 줄에 이어 붙인다(있는 것만)
    const labels: string[] = [];
    if (dbg?.fixes && dbg.fixes.length > 0) labels.push(dbg.fixes.join('·'));
    const ref = dbg?.referee;
    if (ref?.chosen && ref.chosen !== ref.default) {
      labels.push(`⚖ ${ref.default ?? '?'}→${ref.chosen}`);
    }
    if (labels.length > 0) {
      textCol.append(h('div', { className: 'ey-debug-row-labels', text: labels.join(' · ') }));
    }

    // 고스트 비교 Δ 자리 — 비교를 실행하기 전에는 비어 있다(레이아웃만 확보)
    const deltaEl = h('span', { className: 'ey-debug-row-delta' });
    deltaEls.push(deltaEl);

    const row = h('button', {
      className: 'ey-debug-row',
      attrs: { type: 'button' },
      title: t('debugPanel.seekTitle'),
      on: {
        click: () => {
          if (line.time !== null) onSeek(line.time);
        },
      },
    }, h('span', { className: 'ey-debug-row-time', text: fmtTime(line.time) }), deltaEl, chip, textCol);

    list.append(row);
  }
  el.append(list);
  return { el };
}
