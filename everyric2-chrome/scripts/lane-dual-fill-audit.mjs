// 이중표시(발음 줄) 채움 타이밍 실측 — 로컬 DB 싱크 전수.
//
// 운영자 제보: "가라오케 발음 이중표시 시 발음 하이라이트 타이밍이 원문이나 발음 전사본과
// 다르게 표시되는 것 같은데 기분탓인가". 기분탓인지 실결함인지 수치로 가른다.
//
// 재는 것:
//  1) 폴백 발동률 — renderPronFallback의 채움이 «음절 세그 타이밍»(pronCharProgress)을
//     타는 줄 vs «라인 구간 선형 보간»으로 떨어지는 줄. 사유별로 분해한다.
//  2) 채움 오차 — 각 줄을 25Hz로 훑으며 «실제 코드가 그리는 채움 경계»와 «음절 타이밍이
//     가리키는 올바른 경계»의 차이를 글자 수로 잰다(캔버스 measureText 실폭 기준).
//  3) 노트 vs 이중표시 — 같은 시각에 노트 하이라이트가 가리키는 음절과 이중표시 채움
//     경계가 가리키는 음절이 같은가.
//
// **판정 코드는 재구현하지 않는다** — src/ui/pitch-lane.ts를 통째로 임시 복사해 내부
// 함수만 export로 노출하고 esbuild로 번들해 실행한다(프로드와 같은 코드).
// 폭 계산은 실제 캔버스가 필요하므로 headless chromium 페이지 안에서 돈다.
//
// 실행: node scripts/lane-dual-fill-audit.mjs [--script hangul] [--songs 999] [--json out.json]
import { chromium } from 'playwright';
import { DatabaseSync } from 'node:sqlite';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdtempSync, readFileSync, writeFileSync, appendFileSync } from 'fs';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DB = process.env.EVERYRIC_DB ?? resolve(ROOT, '../everyric2.db');

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};
const SCRIPTS = arg('script', 'hangul').split(',');
const SONG_LIMIT = Number(arg('songs', '999'));
const JSON_OUT = arg('json', null);

// ── 1) 프로드 코드 번들 (내부 함수 노출용 임시 복사본) ─────────────────────────
const USE_HEAD = process.argv.includes('--head');
const workDir = mkdtempSync(join(tmpdir(), 'lane-audit-'));
cpSync(join(ROOT, 'src'), join(workDir, 'src'), { recursive: true });
cpSync(join(ROOT, '_locales'), join(workDir, '_locales'), { recursive: true });
// --head: pitch-lane.ts만 커밋된 HEAD 판으로 갈아끼운다 — 같은 코퍼스·같은 지표로
// 수리 전/후를 한 명령으로 대조하기 위한 것이다(다른 파일은 작업본 그대로).
if (USE_HEAD) {
  writeFileSync(join(workDir, 'src/ui/pitch-lane.ts'),
    execFileSync('git', ['show', 'HEAD:everyric2-chrome/src/ui/pitch-lane.ts'],
      { cwd: resolve(ROOT, '..'), encoding: 'utf8', maxBuffer: 1 << 24 }));
}
appendFileSync(join(workDir, 'src/ui/pitch-lane.ts'), `
// ── 하네스 전용 노출(임시 복사본에만 붙는다 — 원본은 안 건드린다) ──
export { pronSegmentsFor, laneLineText, pronCharProgress, originalTextSegments, collectPitchData };
export { segmentsToLines } from '../lib/lyrics-parser';
export { isLatinDominant, resolvedPronSegments } from '../lib/lang';
`);
const esbuild = await import('esbuild');
const built = await esbuild.build({
  entryPoints: [join(workDir, 'src/ui/pitch-lane.ts')],
  bundle: true, format: 'iife', globalName: 'LANE', platform: 'browser',
  loader: { '.json': 'json' }, write: false, logLevel: 'error',
});
const bundle = built.outputFiles[0].text;

// ── 1b) 셀프테스트: pronCharProgress의 경계 동작 (--selftest) ─────────────────
// 코퍼스에 안 나오는 형태(내용 글자가 낀 불일치·꼬리 구두점·간주 틈)까지 못 박아 둔다.
if (process.argv.includes('--selftest')) {
  const LANE = new Function(`${bundle}; return LANE;`)();
  const seg = (text, start, end) => ({ text, start, end });
  let bad = 0;
  const eq = (label, got, want) => {
    const ok = Math.abs((got ?? NaN) - (want ?? NaN)) < 1e-9 || (got === null && want === null);
    console.log(`${ok ? 'PASS' : 'FAIL'}: ${label} = ${got}${ok ? '' : ` (기대 ${want})`}`);
    if (!ok) bad++;
  };
  const S = [seg('a', 0, 1), seg('b', 1, 2)];
  eq('공백 끼움 매핑 — b 한가운데', LANE.pronCharProgress(S, 'a b', 1.5), 2.5);
  eq('공백 끼움 매핑 — a 다 부름(공백은 안 채움)', LANE.pronCharProgress(S, 'a b', 1.0), 1);
  eq('내용 글자가 끼면 포기', LANE.pronCharProgress(S, 'axb', 1.5), null);
  eq('세그가 표시 문자열에 없으면 포기', LANE.pronCharProgress(S, 'zz', 1.5), null);
  eq('꼬리 구두점 — 다 부르면 끝까지 채움', LANE.pronCharProgress(S, 'a b!', 9), 4);
  eq('간주 틈 — 다음 음절 전엔 앞 음절 끝에 머문다',
    LANE.pronCharProgress([seg('a', 0, 1), seg('b', 5, 6)], 'a b', 3), 1);
  eq('이어붙임 그대로인 줄은 예전과 동일', LANE.pronCharProgress(S, 'ab', 1.5), 1.5);
  eq('빈 세그 텍스트도 통과', LANE.pronCharProgress([seg('', 0, 1), ...S], 'a b', 1.5), 2.5);
  console.log(bad === 0 ? 'SELFTEST: PASS' : 'SELFTEST: FAIL');
  if (bad > 0) process.exit(1);
}

// ── 2) 코퍼스: 로컬 DB 싱크(비디오별 최신) ────────────────────────────────────
const db = new DatabaseSync(DB, { readOnly: true });
const rows = db.prepare(
  'select video_id, language, title, timestamps, created_at from sync_results order by created_at desc',
).all();
const seen = new Set();
const corpus = [];
for (const r of rows) {
  if (seen.has(r.video_id)) continue;
  seen.add(r.video_id);
  let segments;
  try { segments = JSON.parse(r.timestamps)?.segments; } catch { continue; }
  if (!Array.isArray(segments) || segments.length === 0) continue;
  const hasNotes = segments.some(s => (s.notes?.length ?? 0) > 0
    || (s.words ?? []).some(w => (w.notes?.length ?? 0) > 0));
  if (!hasNotes) continue;
  corpus.push({ videoId: r.video_id, lang: r.language, segments });
  if (corpus.length >= SONG_LIMIT) break;
}
db.close();
console.log(`[${USE_HEAD ? 'HEAD' : '수리본'}] 코퍼스: ${corpus.length}곡 (노트 보유 싱크, 비디오별 최신)`);

// ── 3) 브라우저 안에서 실측 ───────────────────────────────────────────────────
/** 페이지에서 도는 측정기 — 인자: (곡 세그먼트, 표기) */
const MEASURE = (payload) => {
  const { segments, script, fontPx, cw } = payload;
  const L = window.LANE;
  const canvas = document.createElement('canvas');
  canvas.width = 1200; canvas.height = 60;
  const ctx = canvas.getContext('2d');
  ctx.font = `${fontPx}px system-ui, sans-serif`;
  const clamp01 = v => Math.max(0, Math.min(1, v));

  const lines = L.segmentsToLines(segments).lines;
  const data = L.collectPitchData(lines, script);
  const out = { lines: [], samples: [] };

  for (const page of data.pages) {
    const line = page.line;
    const pron = L.laneLineText(line, script);
    const rec = {
      text: line.text, pron: pron ?? '', reason: null, mapped: null,
      n: 0, errSum: 0, errMax: 0, errs: [],
      noteCmp: 0, noteBadDisp: 0, noteBadTruth: 0,
    };
    if (!pron) { rec.reason = 'no-display'; out.lines.push(rec); continue; }
    const segs = L.laneTextSegments(line, script);
    const pronSegs = L.pronSegmentsFor(line, script);
    const concat = (segs ?? []).map(s => s.text).join('');

    // ── 폴백 사유 분류. «세그 경로를 타는가»는 추측하지 않고 **실제 판정 함수**에 묻는다.
    //    seg-path-mapped = 이어 붙임이 표시 문자열과 다른데도 대응에 성공한 줄(수리로 구제된 줄).
    const takesSeg = segs && segs.length > 0
      && L.pronCharProgress(segs, pron, page.start + (page.end - page.start) / 2) !== null;
    if (!segs || segs.length === 0) rec.reason = 'no-segs';
    else if (takesSeg) rec.reason = concat === pron ? 'seg-path' : 'seg-path-mapped';
    else if (!pronSegs) rec.reason = 'fallback-original';
    else if (L.isLatinDominant(line.text)) rec.reason = 'latin-space-join';
    else rec.reason = 'other-mismatch';

    if (rec.reason !== 'seg-path' && segs && segs.length > 0) {
      const ws = s => s.replace(/\s+/gu, '');
      const wp = s => s.normalize('NFKC').toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '');
      rec.diff = ws(concat) === ws(pron) ? 'whitespace-only'
        : wp(concat) === wp(pron) ? 'punctuation-or-case'
        : 'text-differs';
      rec.sample = { concat: concat.slice(0, 60), pron: pron.slice(0, 60) };
    }

    // ── 세그 → 표시 문자열 문자 구간 대응(정답 매핑). 순서 보존 최초 일치.
    let offs = null;
    if (segs && segs.length > 0) {
      offs = []; let cur = 0; let ok = true;
      for (const s of segs) {
        if (s.text.length === 0) { offs.push([cur, cur]); continue; }
        const idx = pron.indexOf(s.text, cur);
        if (idx < 0) { ok = false; break; }
        offs.push([idx, idx + s.text.length]);
        cur = idx + s.text.length;
      }
      if (!ok) offs = null;
    }
    rec.mapped = offs !== null;
    if (!offs) { out.lines.push(rec); continue; }

    // ── 폭 테이블(누적 prefix 폭) — 픽셀↔글자 환산용
    const W = [0];
    for (let k = 1; k <= pron.length; k++) W.push(ctx.measureText(pron.slice(0, k)).width);
    const fullW = W[pron.length];
    const maxW = cw - 16;
    const textW = Math.min(fullW, maxW);
    const squeeze = fullW > 0 ? textW / fullW : 1;
    const invert = px => {
      if (px <= 0) return 0;
      if (px >= fullW) return pron.length;
      let lo = 0, hi = pron.length;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (W[m] <= px) lo = m; else hi = m; }
      const d = W[lo + 1] - W[lo];
      return lo + (d > 0 ? (px - W[lo]) / d : 0);
    };
    const truthAt = t => {
      let last = 0;
      for (let i = 0; i < segs.length; i++) {
        const s = segs[i], a = offs[i][0], b = offs[i][1];
        if (t >= s.end) { last = b; continue; }
        if (t <= s.start) return last;
        const d = s.end - s.start;
        return a + (b - a) * clamp01(d > 0 ? (t - s.start) / d : 1);
      }
      return last;
    };
    const segIdxAtChar = c => {
      for (let i = 0; i < segs.length; i++) if (c < offs[i][1] - 1e-9) return i;
      return segs.length - 1;
    };
    // 경계(정확히 음절이 끝난 지점)는 앞뒤 어느 음절을 가리켜도 시각적으로 같다 —
    // 두 후보를 모두 허용해야 «채움이 실제로 다른 음절을 가리키는» 경우만 남는다.
    const segTextsAt = c => {
      const a = segs[segIdxAtChar(c)]?.text ?? '';
      const b = segs[segIdxAtChar(Math.max(0, c - 1e-6))]?.text ?? '';
      return a === b ? [a] : [a, b];
    };

    // ── 25Hz 샘플링
    const span = Math.max(0.001, page.end - page.start);
    const steps = Math.min(600, Math.max(4, Math.round(span * 25)));
    for (let k = 0; k <= steps; k++) {
      const t = page.start + (span * k) / steps;
      const chars = L.pronCharProgress(segs, pron, t);
      let sungW;
      if (chars == null) sungW = textW * clamp01((t - page.start) / span);
      else {
        const whole = Math.floor(chars);
        const head = ctx.measureText(pron.slice(0, whole)).width;
        const curW = whole < pron.length ? ctx.measureText(pron[whole]).width : 0;
        sungW = (head + curW * (chars - whole)) * squeeze;
      }
      const dispChars = invert(sungW / (squeeze || 1));
      const tc = truthAt(t);
      const err = Math.abs(dispChars - tc);
      rec.n++; rec.errSum += err; rec.errMax = Math.max(rec.errMax, err);
      rec.errs.push(err);

      // 노트 하이라이트가 가리키는 음절 vs 이중표시 경계가 가리키는 음절
      const note = data.notes.find(n => n.start <= t && t < n.end && n.pron);
      if (note) {
        rec.noteCmp++;
        const dSeg = segTextsAt(dispChars);
        const tSeg = segTextsAt(tc);
        if (!dSeg.some(s => s && note.pron.includes(s))) rec.noteBadDisp++;
        if (!tSeg.some(s => s && note.pron.includes(s))) rec.noteBadTruth++;
      }
    }
    out.lines.push(rec);
  }
  return out;
};

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setContent('<html><body></body></html>');
await page.addScriptTag({ content: bundle });

const pct = (a, b) => (b > 0 ? ((a / b) * 100).toFixed(1) + '%' : '-');
const quant = (arr, q) => (arr.length === 0 ? 0 : arr[Math.min(arr.length - 1, Math.floor(arr.length * q))]);
const report = {};

for (const script of SCRIPTS) {
  const agg = {
    songs: 0, lines: 0, byReason: {}, mapped: 0, unmapped: 0,
    errs: [], errsSegPath: [], errsFallback: [],
    noteCmp: 0, noteBadDisp: 0, noteBadTruth: 0, noteByReason: {},
    worst: [], samplesByReason: {},
  };
  for (const song of corpus) {
    const r = await page.evaluate(MEASURE,
      { segments: song.segments, script, fontPx: 14, cw: 480 });
    agg.songs++;
    for (const rec of r.lines) {
      if (rec.reason === 'no-display') continue;
      agg.lines++;
      agg.byReason[rec.reason] = (agg.byReason[rec.reason] ?? 0) + 1;
      if (rec.diff) {
        const key = `${rec.reason}/${rec.diff}`;
        agg.samplesByReason[key] = agg.samplesByReason[key] ?? { count: 0, ex: [] };
        agg.samplesByReason[key].count++;
        if (agg.samplesByReason[key].ex.length < 3) {
          agg.samplesByReason[key].ex.push({ song: song.videoId, ...rec.sample });
        }
      }
      if (rec.mapped === false || rec.mapped === null) { agg.unmapped++; continue; }
      agg.mapped++;
      for (const e of rec.errs) {
        agg.errs.push(e);
        (rec.reason.startsWith('seg-path') ? agg.errsSegPath : agg.errsFallback).push(e);
      }
      agg.noteCmp += rec.noteCmp;
      agg.noteBadDisp += rec.noteBadDisp;
      agg.noteBadTruth += rec.noteBadTruth;
      const nb = agg.noteByReason[rec.reason] ?? (agg.noteByReason[rec.reason] = { cmp: 0, disp: 0, truth: 0 });
      nb.cmp += rec.noteCmp; nb.disp += rec.noteBadDisp; nb.truth += rec.noteBadTruth;
      if (rec.errMax > 1) {
        agg.worst.push({ song: song.videoId, reason: rec.reason, max: +rec.errMax.toFixed(2),
          text: rec.text.slice(0, 30), pron: rec.pron.slice(0, 30) });
      }
    }
  }
  for (const a of [agg.errs, agg.errsSegPath, agg.errsFallback]) a.sort((x, y) => x - y);
  agg.worst.sort((a, b) => b.max - a.max);
  agg.worst = agg.worst.slice(0, 10);

  const segLines = (agg.byReason['seg-path'] ?? 0) + (agg.byReason['seg-path-mapped'] ?? 0);
  console.log(`\n════ 표기 script=${script} ════`);
  console.log(`곡 ${agg.songs} / 표시되는 줄 ${agg.lines}`);
  console.log(`  세그 타이밍 경로: ${segLines} (${pct(segLines, agg.lines)})`);
  console.log(`  선형 보간 폴백 : ${agg.lines - segLines} (${pct(agg.lines - segLines, agg.lines)})`);
  for (const [k, v] of Object.entries(agg.byReason).sort((a, b) => b[1] - a[1])) {
    if (k === 'seg-path') continue;
    console.log(`     - ${k}: ${v} (${pct(v, agg.lines)})`);
  }
  for (const [k, v] of Object.entries(agg.samplesByReason)) {
    console.log(`     · ${k}: ${v.count}줄, 예시=${JSON.stringify(v.ex[0])}`);
  }
  const fmt = (a) => a.length === 0 ? 'n/a'
    : `중앙값 ${quant(a, 0.5).toFixed(2)}자 · p90 ${quant(a, 0.9).toFixed(2)}자 · 최대 ${a[a.length - 1].toFixed(2)}자 · ≥1자 ${pct(a.filter(v => v >= 1).length, a.length)}`;
  console.log(`  채움 오차(전체 ${agg.errs.length}샘플): ${fmt(agg.errs)}`);
  console.log(`  ├ 세그 경로 줄  (${agg.errsSegPath.length}): ${fmt(agg.errsSegPath)}`);
  console.log(`  └ 폴백 줄      (${agg.errsFallback.length}): ${fmt(agg.errsFallback)}`);
  console.log(`  노트↔이중표시 지시 음절 불일치: 현재 ${pct(agg.noteBadDisp, agg.noteCmp)} / 정답매핑 기준 ${pct(agg.noteBadTruth, agg.noteCmp)} (비교 ${agg.noteCmp}샘플)`);
  for (const [k, v] of Object.entries(agg.noteByReason).sort((a, b) => b[1].cmp - a[1].cmp)) {
    console.log(`     - ${k}: 현재 ${pct(v.disp, v.cmp)} / 정답 ${pct(v.truth, v.cmp)} (${v.cmp}샘플)`);
  }
  console.log(`  정답 매핑 실패(측정 제외): ${agg.unmapped}줄`);
  if (agg.worst.length) console.log(`  최악 줄: ${JSON.stringify(agg.worst.slice(0, 5))}`);
  report[script] = {
    songs: agg.songs, lines: agg.lines, byReason: agg.byReason,
    unmapped: agg.unmapped, samples: agg.samplesByReason,
    err: { med: quant(agg.errs, 0.5), p90: quant(agg.errs, 0.9), max: agg.errs[agg.errs.length - 1] ?? 0,
      ge1: agg.errs.filter(v => v >= 1).length / Math.max(1, agg.errs.length) },
    errSegPath: { med: quant(agg.errsSegPath, 0.5), p90: quant(agg.errsSegPath, 0.9) },
    errFallback: { med: quant(agg.errsFallback, 0.5), p90: quant(agg.errsFallback, 0.9) },
    note: { cmp: agg.noteCmp, badDisp: agg.noteBadDisp, badTruth: agg.noteBadTruth },
    worst: agg.worst,
  };
}

await browser.close();
if (JSON_OUT) { writeFileSync(JSON_OUT, JSON.stringify(report, null, 2)); console.log(`\nJSON: ${JSON_OUT}`); }
