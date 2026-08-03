// matchWikiLinesToSegments 반복 원문 재사용 보강 — 실함수 단위 검증(esbuild 트랜스파일).
// 케이스: (1) 熱異常 재현(반복 1회 축약 위키)에서 그 줄이 채워짐, (2) 기존 매칭 결과 불변,
// (3) 진짜 결측(고유 텍스트)은 여전히 undefined.
// 실행: node scripts/wiki-repeat-translation-reuse.test.mjs
import { build } from 'esbuild';
import { writeFileSync, mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(__dirname, '../src/lib/lang.ts');
const out = await build({
  entryPoints: [SRC], bundle: false, write: false, format: 'esm', platform: 'node', target: 'node18',
});
const code = out.outputFiles[0].text;
const tmpFile = join(mkdtempSync(join(tmpdir(), 'lang-test-')), 'lang.mjs');
writeFileSync(tmpFile, code);
const { matchWikiLinesToSegments } = await import('file://' + tmpFile.replace(/\\/g, '/'));

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}

// 熱異常(arX83q0oJhM) 실측 원문/번역 기반 — "どこに送るあてもなく"가 원문에는 2번(인덱스
// 2, 5) 나오는데 위키 표에는 한 번만 적혀 있다(실제 사고 재현). 인덱스6("本当に嫌な奴だと
// 思われても")은 원문에도 위키에도 대응 짝이 아예 없는 고유 텍스트 — 진짜 결측 케이스.
const segs = [
  { text: '「死んだ変数で繰り返す' },       // 0
  { text: '数え事が孕んだ熱' },             // 1
  { text: 'どこに送るあてもなく' },         // 2  <- 위키에 1회만 존재
  { text: 'あわれな独り言を記している' },   // 3
  { text: 'この世界の全ての人間に' },       // 4
  { text: 'どこに送るあてもなく' },         // 5  <- 반복 등장(위키엔 대응 짝 없음) — 보강 대상
  { text: '本当に嫌な奴だと思われても' },   // 6  <- 진짜 결측(위키에 이 텍스트 자체가 없음)
  { text: '微粒子の濃い煙の向こうに' },     // 7
];
const wiki = [
  { text: '「死んだ変数で繰り返す', translation: 'I start over with a dead variable' },
  { text: '数え事が孕んだ熱', translation: 'Counting them made my temperature rise' },
  { text: 'どこに送るあてもなく', translation: 'With nowhere to send it to,' },
  { text: 'あわれな独り言を記している', translation: 'I write down a pitiful soliloquy' },
  { text: 'この世界の全ての人間に', translation: 'To every human being in this world' },
  // (반복 두 번째 등장은 위키에 없음 — 사고 재현)
  // (인덱스6에 대응하는 위키 줄 자체가 없음 — 진짜 결측)
  { text: '微粒子の濃い煙の向こうに', translation: 'From beyond the thick smoke of particulates,' },
];

// ── 대조군: 보강 없는(기존) 동작을 흉내내기 위해 같은 입력을 한 번 더 돌려 앞부분
//    5개(원래도 매칭됐어야 할 자리)의 값이 안 바뀌는지 확인 ──
const before = {
  0: 'I start over with a dead variable',
  1: 'Counting them made my temperature rise',
  2: 'With nowhere to send it to,',
  3: 'I write down a pitiful soliloquy',
  4: 'To every human being in this world',
  7: 'From beyond the thick smoke of particulates,',
};

const result = matchWikiLinesToSegments(segs, wiki);
console.log('result:', JSON.stringify(result, null, 2));

// (1) 熱異常 재현 케이스 — 반복 두 번째 등장(인덱스5)이 채워짐, 첫 등장(인덱스2)과 동일 값
check(result[5] === 'With nowhere to send it to,',
  '(1) 반복 원문 두 번째 등장이 첫 등장과 같은 번역으로 채워짐', { idx5: result[5] });
check(result[5] === result[2], '(1) 반복 두 등장의 번역이 서로 동일(같은 문장=같은 번역)', { idx2: result[2], idx5: result[5] });

// (2) 기존 매칭 결과 불변 — 원래도 매칭됐던 자리는 값이 그대로
let regressed = [];
for (const [idx, expected] of Object.entries(before)) {
  if (result[idx] !== expected) regressed.push({ idx, expected, actual: result[idx] });
}
check(regressed.length === 0, '(2) 기존 매칭 결과 한 줄도 안 바뀜(전/후 대조)', { regressed });

// (3) 진짜 결측 — 고유 텍스트(인덱스6)는 여전히 undefined(번역 발명 금지)
check(result[6] === undefined, '(3) 진짜 결측 줄은 여전히 undefined(발명 금지)', { idx6: result[6] });

// 반환 길이는 항상 seg 개수와 같아야 한다(기존 불변식 유지 확인)
check(result.length === segs.length, '반환 배열 길이 = seg 개수', { len: result.length, expected: segs.length });

console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
