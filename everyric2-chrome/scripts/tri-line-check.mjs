// src/lib/tri-line.ts의 다형식 가사 파서 단독 검증. 테스트 러너가 없는 프로젝트라
// node 단독 실행으로 대체한다. tri-line.ts는 확장자 없는 상대 import(./lyrics-clean)를
// 쓰는 TS 관례라 Node ESM 로더가 직접 resolve하지 못한다 — vite가 이미 의존하는
// esbuild로 번들링해 임시 파일로 만든 뒤 그걸 import한다 (실제 빌드 경로와 같은 변환기).
// 실행: node scripts/tri-line-check.mjs
import { build } from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const tmpDir = mkdtempSync(join(tmpdir(), 'tri-line-check-'));
const outfile = join(tmpDir, 'tri-line.bundle.mjs');
await build({
  entryPoints: [resolve(__dirname, '../src/lib/tri-line.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile,
});
const { parseTriLineLyrics } = await import(`file://${outfile.replace(/\\/g, '/')}`);
rmSync(tmpDir, { recursive: true, force: true });

let pass = 0;
let fail = 0;

function assertEqual(name, actual, expected) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    pass++;
    console.log(`PASS: ${name}`);
  } else {
    fail++;
    console.log(`FAIL: ${name}\n  actual:   ${a}\n  expected: ${e}`);
  }
}

function assertNull(name, actual) {
  assertEqual(name, actual, null);
}

// ── 1) 기존: 원문/독음/번역 3줄 반복, 빈 줄 없이 이어짐 ──────────────
assertEqual(
  '3줄 교차 · 빈 줄 없음',
  parseTriLineLyrics('紫の気配\n보라시키\n보랏빛 기척\n幼き春の余光\n오사나키하루노요코오\n어렸던 봄의 잔향'),
  [
    { text: '紫の気配', pronunciation: '보라시키', translation: '보랏빛 기척' },
    { text: '幼き春の余光', pronunciation: '오사나키하루노요코오', translation: '어렸던 봄의 잔향' },
  ],
);

// ── 2) 기존: 원문/독음/번역 3줄, 블록마다 빈 줄로 구분 (블록 길이 === 3) ──
assertEqual(
  '3줄 교차 · 블록마다 빈 줄 구분',
  parseTriLineLyrics('紫の気配\n보라시키\n보랏빛 기척\n\n幼き春の余光\n오사나키하루노요코오\n어렸던 봄의 잔향'),
  [
    { text: '紫の気配', pronunciation: '보라시키', translation: '보랏빛 기척' },
    { text: '幼き春の余光', pronunciation: '오사나키하루노요코오', translation: '어렸던 봄의 잔향' },
  ],
);

// ── 3) 신규: 원문/번역 2줄 교차, 빈 줄 없이 이어짐 ──────────────────
assertEqual(
  '2줄 교차 · 빈 줄 없음',
  parseTriLineLyrics('紫の気配\n보랏빛 기척\n幼き春の余光\n어렸던 봄의 잔향'),
  [
    { text: '紫の気配', translation: '보랏빛 기척' },
    { text: '幼き春の余光', translation: '어렸던 봄의 잔향' },
  ],
);

// ── 4) 운영자가 준 실제 예시: 첫 그룹은 빈 줄로 구분, 나머지 두 그룹은 붙어 있음 ──
assertEqual(
  '2줄 교차 · 실제 예시(첫 그룹만 빈 줄 구분, 나머지는 붙음)',
  parseTriLineLyrics(
    '紫の気配\n보랏빛 기척\n\n幼き春の余光\n어렸던 봄의 잔향\n悲しイオの味\n슬픈 이오논 맛',
  ),
  [
    { text: '紫の気配', translation: '보랏빛 기척' },
    { text: '幼き春の余光', translation: '어렸던 봄의 잔향' },
    { text: '悲しイオの味', translation: '슬픈 이오논 맛' },
  ],
);

// ── 5) 신규: 원문(영어)/번역 2줄 교차 ────────────────────────────
assertEqual(
  '2줄 교차 · 라틴(영어) 원문',
  parseTriLineLyrics(
    'I am falling into the night\n밤 속으로 떨어지고 있어\nHold my hand and never let go\n내 손을 잡고 놓지 마',
  ),
  [
    { text: 'I am falling into the night', translation: '밤 속으로 떨어지고 있어' },
    { text: 'Hold my hand and never let go', translation: '내 손을 잡고 놓지 마' },
  ],
);

// ── 6) 안전 폴백: 원문만 있는 일반 텍스트는 절대 오판하지 않는다 ────────
assertNull(
  '안전 폴백 · 원문만(일어)',
  parseTriLineLyrics('紫の気配\n幼き春の余光\n悲しイオの味\n忘れられぬ夜\n遠い記憶\n静かな雨'),
);
assertNull(
  '안전 폴백 · 원문만(영어)',
  parseTriLineLyrics('I am falling into the night\nHold my hand and never let go\nDon\'t look back now\nWe are running out of time'),
);
assertNull(
  '안전 폴백 · 한국어 가사만(번역 없이)',
  parseTriLineLyrics('보랏빛 기척\n어렸던 봄의 잔향\n슬픈 이오논 맛\n잊혀지지 않는 밤\n먼 기억\n조용한 비'),
);

// ── 7) 안전 폴백: 세트가 하나뿐이면(반복 아님) 인정하지 않는다 ───────────
assertNull(
  '안전 폴백 · 세트 1개(2줄)',
  parseTriLineLyrics('紫の気配\n보랏빛 기척'),
);
assertNull(
  '안전 폴백 · 세트 1개(3줄)',
  parseTriLineLyrics('紫の気配\n보라시키\n보랏빛 기척'),
);

// ── 8) 파트 표기가 섞여도 걸러내고 정상 인식 ────────────────────────
assertEqual(
  '파트 표기 혼입 · 3줄 교차',
  parseTriLineLyrics('[Verse 1]\n紫の気配\n보라시키\n보랏빛 기척\n幼き春の余光\n오사나키하루노요코오\n어렸던 봄의 잔향'),
  [
    { text: '紫の気配', pronunciation: '보라시키', translation: '보랏빛 기척' },
    { text: '幼き春の余光', pronunciation: '오사나키하루노요코오', translation: '어렸던 봄의 잔향' },
  ],
);

// ── 9) 짧은 3줄 교차(2세트)가 2줄 형식으로 오판되지 않는다 ─────────────
assertEqual(
  '오판 방지 · 짧은 3줄 교차가 2줄로 안 새어나감',
  parseTriLineLyrics('紫の気配\n보라시키\n보랏빛 기척\n幼き春の余光\n오사나키하루노요코오\n어렸던 봄의 잔향'),
  [
    { text: '紫の気配', pronunciation: '보라시키', translation: '보랏빛 기척' },
    { text: '幼き春の余光', pronunciation: '오사나키하루노요코오', translation: '어렸던 봄의 잔향' },
  ],
);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
