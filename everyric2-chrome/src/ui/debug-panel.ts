import type { LyricLine, SyncDebugMeta, SyncPreviousVersion } from '../types';
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
 * @param onSeek 이미 SEEK_INTO_LINE_SEC 같은 보정이 필요하면 호출부가 콜백 안에서 적용한다 —
 *   이 함수는 line.time을 그대로 넘긴다.
 */
export function buildDebugPanel(
  lines: LyricLine[],
  debugMeta: SyncDebugMeta | null | undefined,
  onSeek: (time: number) => void,
  loadPrevious?: () => Promise<SyncPreviousVersion | null>,
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

  // A/B 고스트 비교 — 서버가 보관한 직전 세대(재처리로 덮어써지기 전)의 줄 시각과
  // 지금 화면의 줄 시각을 라인 단위 Δ로 대비한다. 줄 텍스트가 같은 라인만 비교하고
  // (재생성 = 같은 가사의 재정렬일 때만 A/B가 성립), 다른 줄은 Δ— 로 남긴다.
  const deltaEls: HTMLSpanElement[] = [];
  if (loadPrevious) {
    const status = h('span', { className: 'ey-debug-compare-status' });
    const btn = h('button', {
      className: 'ey-btn',
      attrs: { type: 'button' },
      text: t('debugPanel.comparePrev'),
      on: {
        click: () => {
          btn.disabled = true;
          status.textContent = '…';
          void loadPrevious().then(prev => {
            if (!prev?.found || !prev.timestamps || prev.timestamps.length === 0) {
              status.textContent = t('debugPanel.comparePrevNone');
              return;
            }
            let matched = 0;
            lines.forEach((line, i) => {
              const old = prev.timestamps?.[i];
              const deltaEl = deltaEls[i];
              if (!deltaEl) return;
              if (!old || old.text !== line.text || line.time === null) {
                deltaEl.textContent = 'Δ—';
                return;
              }
              matched++;
              const d = line.time - old.start;
              deltaEl.textContent = `Δ${d >= 0 ? '+' : ''}${d.toFixed(2)}s`;
              // 0.15s 이내는 흐리게(사실상 동일), 그 밖은 또렷하게 — 큰 이동만 눈에 띄게
              deltaEl.classList.toggle('big', Math.abs(d) > 0.15);
            });
            status.textContent = t('debugPanel.comparePrevLoaded', [
              prev.engine_version ?? t('debugPanel.engineLegacy'),
              String(matched), String(lines.length),
            ]);
          }).catch(() => {
            status.textContent = t('debugPanel.comparePrevNone');
            btn.disabled = false;
          });
        },
      },
    }) as HTMLButtonElement;
    el.append(h('div', { className: 'ey-debug-compare-bar' }, btn, status));
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
