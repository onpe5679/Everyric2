// 이중표시 채움 경계 **픽셀 실측** A/B — HEAD 코드 vs 작업본 코드를 같은 페이지에서
// 같은 곡·같은 시각으로 렌더하고, 발음 줄의 accent 채움이 실제로 어디서 끊기는지 잰다.
//
// lane-dual-fill-audit.mjs가 «판정 함수»를 재는 반면 이쪽은 «화면에 그려진 결과»를 잰다 —
// 함수 수준의 개선이 렌더 경로까지 실제로 도달했는지 확인하는 것이 목적이다.
//
// 재는 법: 오선 아래(staffBottom 이하)에서 accent 색 픽셀의 오른쪽 끝 = 채움 경계.
// 같은 y 밴드의 전체 잉크 좌우 끝으로 나눠 «채워진 비율»을 얻고, 음절 타이밍이 가리키는
// 정답 비율(measureText 실폭 기준)과 비교한다. 비율이라 폰트 크기에 의존하지 않는다.
//
// 실행: node scripts/lane-dual-fill-pixel-ab.mjs <videoId> [--script hangul] [--lines 4]
import { chromium } from 'playwright';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { appendFileSync, cpSync, mkdtempSync, writeFileSync } from 'fs';
import { execFileSync } from 'child_process';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const DB = process.env.EVERYRIC_DB ?? resolve(ROOT, '../everyric2.db');
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const VIDEO = process.argv[2];
const SCRIPT = arg('script', 'hangul');
const MAX_LINES = Number(arg('lines', '4'));
if (!VIDEO) { console.log('사용법: node scripts/lane-dual-fill-pixel-ab.mjs <videoId> [--script hangul]'); process.exit(2); }

const esbuild = await import('esbuild');
/** src 스냅샷 하나를 번들한다. head=true면 pitch-lane.ts만 HEAD 판으로 갈아끼운다. */
async function bundleOf(globalName, head) {
  const dir = mkdtempSync(join(tmpdir(), `lane-ab-${globalName}-`));
  cpSync(join(ROOT, 'src'), join(dir, 'src'), { recursive: true });
  cpSync(join(ROOT, '_locales'), join(dir, '_locales'), { recursive: true });
  if (head) {
    const src = execFileSync('git', ['show', 'HEAD:everyric2-chrome/src/ui/pitch-lane.ts'],
      { cwd: resolve(ROOT, '..'), encoding: 'utf8', maxBuffer: 1 << 24 });
    writeFileSync(join(dir, 'src/ui/pitch-lane.ts'), src);
  }
  // 하네스 전용 노출 — 임시 복사본에만 붙는다
  appendFileSync(join(dir, 'src/ui/pitch-lane.ts'),
    `\nexport { laneLineText, pronCharProgress, pronSegmentsFor };\nexport { segmentsToLines } from '../lib/lyrics-parser';\n`);
  const built = await esbuild.build({
    entryPoints: [join(dir, 'src/ui/pitch-lane.ts')], bundle: true, format: 'iife',
    globalName, platform: 'browser', loader: { '.json': 'json' }, write: false, logLevel: 'error',
  });
  return built.outputFiles[0].text;
}
const [bundleA, bundleB] = await Promise.all([bundleOf('LANE_A', true), bundleOf('LANE_B', false)]);

const db = new DatabaseSync(DB, { readOnly: true });
const row = db.prepare('select timestamps, language from sync_results where video_id = ? order by created_at desc limit 1').get(VIDEO);
db.close();
if (!row) { console.log(`FAIL: DB에 ${VIDEO} 싱크가 없다`); process.exit(1); }
const segments = JSON.parse(row.timestamps).segments;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
await page.setContent(`<html><body style="margin:0;background:#0b0b0d">
  <canvas id="a" style="width:480px;height:260px;display:block"></canvas>
  <canvas id="b" style="width:480px;height:260px;display:block"></canvas>
</body></html>`);
await page.addScriptTag({ content: bundleA });
await page.addScriptTag({ content: bundleB });

const result = await page.evaluate(async ({ segments: raw, script, maxLines }) => {
  // 원문 가사 줄(노트 아래 시간축 토큰)은 이미 부른 단어를 accent로 칠한다 — 발음 줄
  // 채움과 같은 색이라 픽셀로 구분되지 않는다. words의 노트를 라인 노트로 옮기고 words를
  // 비워 그 줄만 없앤다(노트·발음 세그는 그대로라 이 측정의 대상은 바뀌지 않는다).
  const segments = raw.map(s => ({
    ...s,
    notes: [...(s.notes ?? []), ...(s.words ?? []).flatMap(w => w.notes ?? [])],
    words: undefined,
  }));
  const mk = (L, id) => {
    const canvas = document.getElementById(id);
    const r = new L.PitchLaneRenderer({
      enabled: true, pronPosition: 'bottom', pronScript: script, scrollMode: 'page',
      fontScale: 1, windowMeasures: 4, countdown: false, showConfidence: false, showF0: false,
    });
    r.attach(canvas, document.documentElement);
    r.setLines(L.segmentsToLines(segments).lines);
    return { r, canvas };
  };
  const A = mk(window.LANE_A, 'a');
  const B = mk(window.LANE_B, 'b');
  const L = window.LANE_B;
  const lines = L.segmentsToLines(segments).lines;

  // 정답 비율(글자 폭 기준) — 폰트 크기에 무관한 비율이라 오프스크린 20px로 계산한다
  const probe = document.createElement('canvas').getContext('2d');
  probe.font = '20px system-ui, sans-serif';
  const truthFraction = (line, t) => {
    const pron = L.laneLineText(line, script);
    const segs = L.laneTextSegments(line, script);
    if (!pron || !segs) return null;
    let cur = 0; const offs = [];
    for (const s of segs) {
      if (!s.text.length) { offs.push([cur, cur]); continue; }
      const at = pron.indexOf(s.text, cur);
      if (at < 0) return null;
      offs.push([at, at + s.text.length]); cur = at + s.text.length;
    }
    let c = 0;
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      if (t >= s.end) { c = offs[i][1]; continue; }
      if (t <= s.start) break;
      const d = s.end - s.start;
      c = offs[i][0] + (offs[i][1] - offs[i][0]) * Math.max(0, Math.min(1, d > 0 ? (t - s.start) / d : 1));
      break;
    }
    const whole = Math.floor(c);
    const head = probe.measureText(pron.slice(0, whole)).width;
    const curW = whole < pron.length ? probe.measureText(pron[whole]).width : 0;
    return (head + curW * (c - whole)) / probe.measureText(pron).width;
  };

  /** 발음 줄의 accent 채움 오른쪽 끝 / 그 줄 전체 잉크 좌우 끝 → 채워진 비율.
   *  오선 아래에서 accent가 나오는 곳은 발음 줄 채움뿐이다(원문 가사 줄은 words를
   *  비워 없앴고, 다음 줄 미리보기·번역은 dim으로만 그려진다). accent가 걸친 y에서
   *  잉크가 이어지는 만큼 위아래로 밴드를 넓혀 그 줄의 좌우 끝을 잡는다. */
  const measure = ({ r, canvas }) => {
    const view = r.viewport();
    const g = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const y0 = Math.round((view?.staffBottom ?? canvas.height * 0.6) * dpr);
    const h = canvas.height - y0;
    const w = canvas.width;
    const d = g.getImageData(0, y0, w, h).data;
    const inkAt = (y, x) => d[(y * w + x) * 4 + 3] >= 60;
    const rowHasInk = y => { for (let x = 0; x < w; x++) if (inkAt(y, x)) return true; return false; };
    let ax = -1, ayMin = 1e9, ayMax = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        if (d[i + 3] < 60) continue;
        // accent(#ffb02e)는 파랑이 낮고 빨강이 높다 — 흰 글자(#f1f1f2)와 확실히 갈린다
        if (d[i] > 190 && d[i + 2] < 130) { if (x > ax) ax = x; if (y < ayMin) ayMin = y; if (y > ayMax) ayMax = y; }
      }
    }
    if (ax < 0) return null;
    let top = ayMin, bot = ayMax;
    for (let k = 0; k < 14 && top > 0 && rowHasInk(top - 1); k++) top--;
    for (let k = 0; k < 14 && bot < h - 1 && rowHasInk(bot + 1); k++) bot++;
    let lx = 1e9, rx = -1;
    for (let y = top; y <= bot; y++) {
      for (let x = 0; x < w; x++) {
        if (!inkAt(y, x)) continue;
        if (x < lx) lx = x; if (x > rx) rx = x;
      }
    }
    if (rx <= lx) return null;
    return { fill: (ax - lx) / (rx - lx), boundaryPx: ax / dpr, leftPx: lx / dpr, rightPx: rx / dpr,
      bandPx: [(top + y0) / dpr, (bot + y0) / dpr] };
  };

  const out = [];
  const control = [];
  for (let i = 0; i < lines.length && (out.length < maxLines * 2 || control.length < maxLines * 2); i++) {
    const line = lines[i];
    const pron = L.laneLineText(line, script);
    const segs = L.laneTextSegments(line, script);
    if (!pron || !segs || segs.length < 4) continue;
    // words를 비운 것이 세그에 영향을 주지 않도록 «서버 발음 세그가 있는 줄»만 본다
    if (!L.pronSegmentsFor(line, script)) continue;
    const concat = segs.map(s => s.text).join('');
    // 수리 대상 = «이어 붙임 ≠ 표시»인데 대응은 되는 줄, 대조군 = 원래부터 세그 경로인 줄.
    // 대조군의 잔차가 이 지표 자체의 편향(잉크 끝 vs 글자 폭)이라 판정 문턱이 된다.
    const isTarget = concat !== pron;
    if (isTarget && window.LANE_B.pronCharProgress(segs, pron, segs[0].start) === null) continue;
    if (isTarget ? out.length >= maxLines * 2 : control.length >= maxLines * 2) continue;
    // 두 시각을 잰다: 음절 «시작 직후»(경계가 공백 자리에 놓인다 — 잉크 끝과 글자 폭 끝이
    // 갈리는 지표 편향이 최대)와 음절 «한가운데»(경계가 글자 안이라 편향이 최소).
    const s = segs[Math.floor(segs.length * 0.7)];
    for (const [when, t] of [['음절시작', s.start + 0.01], ['음절중간', (s.start + s.end) / 2]]) {
      A.r.setIndex(i); B.r.setIndex(i);
      A.r.render(t, false); B.r.render(t, false);
      const rec = { i, when, t: +t.toFixed(3), text: line.text.slice(0, 34), pron: pron.slice(0, 40),
        truth: truthFraction(line, t), head: measure(A), fixed: measure(B) };
      (isTarget ? out : control).push(rec);
    }
  }
  // 스크린샷에는 «수리 대상 줄»의 프레임을 남긴다 — 마지막 렌더가 대조군이면 두 캔버스가
  // 똑같이 나와 증거 구실을 못 한다. 첫 대상 줄의 음절 중간 시각으로 다시 그려 둔다.
  const shot = out.find(r => r.when === '음절중간') ?? out[0];
  if (shot) {
    A.r.setIndex(shot.i); B.r.setIndex(shot.i);
    A.r.render(shot.t, false); B.r.render(shot.t, false);
  }
  return { targets: out, control, shot: shot ? { i: shot.i, t: shot.t, pron: shot.pron } : null };
}, { segments, script: SCRIPT, maxLines: MAX_LINES });

let failed = false;
const err = (r, which) => (r[which] && r.truth != null ? Math.abs(r[which].fill - r.truth) : null);
console.log(`\n픽셀 A/B — ${VIDEO} (script=${SCRIPT})\n`);
console.log('─ 대조군(원래부터 세그 경로인 줄) — 이 잔차가 지표 자체의 편향이다');
const ctlErr = [];
for (const r of result.control) {
  const ef = err(r, 'fixed');
  if (ef != null) ctlErr.push(ef);
  console.log(`L${r.i} [${r.when}] "${r.pron}" 정답 ${r.truth?.toFixed(3)} | HEAD ${r.head?.fill?.toFixed(3)} | 수리 ${r.fixed?.fill?.toFixed(3)} (잔차 ${ef?.toFixed(3)})`);
  if (err(r, 'head') != null && ef != null && Math.abs(ef - err(r, 'head')) > 0.01) {
    failed = true; console.log('   FAIL: 회귀 — 원래 세그 경로였던 줄의 채움이 바뀌었다');
  }
}
const noise = ctlErr.length ? Math.max(...ctlErr) : 0.05;
const limit = noise + 0.02;
console.log(`\n─ 수리 대상 줄 (판정 문턱 = 대조군 최대 잔차 ${noise.toFixed(3)} + 0.02 = ${limit.toFixed(3)})`);
for (const r of result.targets) {
  const eh = err(r, 'head'), ef = err(r, 'fixed');
  console.log(`L${r.i} [${r.when}] t=${r.t}s "${r.text}" → "${r.pron}"`);
  console.log(`   정답 채움 비율 ${r.truth?.toFixed(3)} | HEAD ${r.head?.fill?.toFixed(3)} (오차 ${eh?.toFixed(3)}) | 수리 ${r.fixed?.fill?.toFixed(3)} (오차 ${ef?.toFixed(3)})`);
  console.log(`   경계 x: HEAD ${r.head?.boundaryPx?.toFixed(1)}px → 수리 ${r.fixed?.boundaryPx?.toFixed(1)}px (줄 폭 ${r.fixed?.leftPx?.toFixed(0)}~${r.fixed?.rightPx?.toFixed(0)}px)`);
  // «음절시작» 시각은 채움 경계가 공백 위에 놓인다 — 공백은 잉크가 없으므로 픽셀로 잰
  // 경계(직전 글자의 잉크 끝)가 글자 폭 기준 정답보다 구조적으로 모자라게 나온다.
  // 편향 없는 판정은 «음절중간»으로 하고, 시작 시각은 HEAD 대비 개선만 본다.
  const cap = r.when === '음절중간' ? limit : 0.10;
  if (ef == null || ef > cap) { failed = true; console.log(`   FAIL: 수리 후에도 정답 비율과 ${cap.toFixed(3)} 넘게 차이`); }
  if (eh != null && ef != null && ef > eh) { failed = true; console.log('   FAIL: 수리가 오히려 나빠졌다'); }
}
await page.locator('#a').screenshot({ path: resolve(ROOT, 'lane-dual-fill-head.png') });
await page.locator('#b').screenshot({ path: resolve(ROOT, 'lane-dual-fill-fixed.png') });
console.log(`\nscreenshot: lane-dual-fill-head.png / lane-dual-fill-fixed.png`);
console.log(result.targets.length === 0 ? 'PIXEL A/B: 대상 줄 없음(SKIP)' : failed ? 'PIXEL A/B: FAIL' : 'PIXEL A/B: PASS');
await browser.close();
process.exitCode = failed ? 1 : 0;
