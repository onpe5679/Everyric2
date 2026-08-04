// 로케일 키 전수 대조 — 소스가 부르는 키가 카탈로그에 **실제로 있는가**.
//
// 왜 필요한가: t()는 못 찾은 키를 조용히 폴백한다(ko 카탈로그 → 그것도 없으면 키 문자열
// 자체). 즉 오타나 «지운 키를 계속 부르는 자리»가 있어도 빌드는 통과하고, 화면에는 영어도
// 한국어도 아닌 `overlay.pip.keepPanl` 같은 원문이 그대로 뜨거나 엉뚱한 언어가 섞인다.
// 키가 계속 늘어나는 배치에서는 이 어긋남이 사람 눈에만 걸린다 — 그래서 기계로 센다.
//
// 재는 것:
//   1) [오류] 소스가 부르는 키가 ko/en/ja 어디에든 없다
//   2) [오류] 3언어 키 집합이 어긋난다(한쪽에만 있는 키)
//   3) [정보] 정적으로 못 읽는 t() 호출 — 키가 변수/템플릿이라 사람이 봐야 하는 자리
//   4) [정보] 카탈로그에는 있는데 아무도 안 부르는 키(고아) — 지울 후보이지 오류는 아니다
//
// 실행: node scripts/i18n-key-check.mjs
import { readFileSync, readdirSync, statSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve, join, relative } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const LANGS = ['ko', 'en', 'ja'];

let failed = false;
const fail = (msg) => { console.log(`FAIL: ${msg}`); failed = true; };
const pass = (msg) => console.log(`PASS: ${msg}`);

// ── 카탈로그 ──────────────────────────────────────────────────────
const catalogs = {};
for (const lang of LANGS) {
  catalogs[lang] = JSON.parse(readFileSync(join(ROOT, '_locales', lang, 'messages.json'), 'utf8'));
}
const keySets = Object.fromEntries(LANGS.map(l => [l, new Set(Object.keys(catalogs[l]))]));
const allKeys = new Set(LANGS.flatMap(l => [...keySets[l]]));

// ── 소스 수집 ─────────────────────────────────────────────────────
function walk(dir, exts, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, exts, out);
    else if (exts.some(e => name.endsWith(e))) out.push(p);
  }
  return out;
}
const tsFiles = walk(join(ROOT, 'src'), ['.ts']);
const htmlFiles = walk(join(ROOT, 'src'), ['.html']);

/** i18n 키는 점 표기로 부르고 카탈로그에는 밑줄로 들어간다(lib/i18n.ts norm) */
const norm = (k) => k.replace(/\./g, '_');

/** 참조된 키 → 어디서 불렀는지(사람이 고칠 수 있게 위치를 남긴다) */
const refs = new Map();
const addRef = (key, where) => {
  if (!refs.has(key)) refs.set(key, []);
  refs.get(key).push(where);
};
const dynamicSites = [];

for (const file of tsFiles) {
  const src = readFileSync(file, 'utf8');
  const rel = relative(ROOT, file).replace(/\\/g, '/');
  const lineOf = (idx) => src.slice(0, idx).split('\n').length;

  // t('키') / t("키") — 앞에 식별자나 점이 붙은 것(format(, .at( 등)은 제외한다
  for (const m of src.matchAll(/(?<![\w$.])t\(\s*(['"])([^'"]+)\1/g)) {
    addRef(m[2], `${rel}:${lineOf(m.index)}`);
  }
  // 키가 리터럴이 아닌 호출 — 정적으로는 못 읽는다. 오류가 아니라 «사람이 봐야 하는 자리»다.
  // 주석과 t() 자신의 정의는 뺀다 — 목록에 섞이면 «확인할 자리»가 아닌 것이 끼어 신뢰를 잃는다.
  for (const m of src.matchAll(/(?<![\w$.])t\(\s*(?!['"])(?![)])(.{0,40})/g)) {
    const line = src.slice(src.lastIndexOf('\n', m.index) + 1, src.indexOf('\n', m.index));
    const head = line.trim();
    if (head.startsWith('*') || head.startsWith('//') || head.includes('export function t(')) continue;
    dynamicSites.push(`${rel}:${lineOf(m.index)}  ${head.slice(0, 60)}`);
  }
}

// options.html의 data-i18n / data-i18n-html — options.ts가 이 값을 그대로 t()에 넣는다
for (const file of htmlFiles) {
  const src = readFileSync(file, 'utf8');
  const rel = relative(ROOT, file).replace(/\\/g, '/');
  const lineOf = (idx) => src.slice(0, idx).split('\n').length;
  for (const m of src.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) {
    addRef(m[1], `${rel}:${lineOf(m.index)}`);
  }
}

// manifest의 __MSG_키__ (지금은 없지만 나중에 생기면 자동으로 잡힌다)
const manifestPath = join(ROOT, 'manifest.json');
try {
  const mf = readFileSync(manifestPath, 'utf8');
  for (const m of mf.matchAll(/__MSG_([A-Za-z0-9_]+)__/g)) addRef(m[1].replace(/_/g, '.'), 'manifest.json');
} catch { /* manifest가 없는 배치 구조면 건너뛴다 */ }

// ── 1) 참조했는데 없는 키 ─────────────────────────────────────────
const missing = [];
for (const [key, where] of refs) {
  const k = norm(key);
  const absent = LANGS.filter(l => !keySets[l].has(k));
  if (absent.length > 0) missing.push({ key, k, absent, where });
}
if (missing.length === 0) {
  pass(`소스가 부르는 키 ${refs.size}개가 3언어 카탈로그에 모두 있다`);
} else {
  for (const m of missing) {
    fail(`없는 키 참조: '${m.key}' (${m.absent.join('/')}에 없음) ← ${m.where.join(', ')}`);
  }
}

// ── 2) 3언어 키 집합 일치 ─────────────────────────────────────────
const onlyIn = {};
for (const lang of LANGS) {
  const others = LANGS.filter(l => l !== lang);
  const only = [...keySets[lang]].filter(k => others.some(o => !keySets[o].has(k)));
  if (only.length) onlyIn[lang] = only;
}
if (Object.keys(onlyIn).length === 0) {
  pass(`3언어 키 집합이 일치한다 (각 ${keySets.ko.size}개)`);
} else {
  for (const [lang, keys] of Object.entries(onlyIn)) {
    fail(`${lang}에만 있는 키 ${keys.length}개: ${keys.slice(0, 8).join(', ')}${keys.length > 8 ? ' …' : ''}`);
  }
}

// ── 3) 정적으로 못 읽는 호출 ──────────────────────────────────────
console.log(`\nINFO: 키가 리터럴이 아닌 t() 호출 ${dynamicSites.length}곳 — 사람이 확인할 자리`);
for (const s of dynamicSites) console.log(`  ${s}`);

// ── 4) 고아 키 ────────────────────────────────────────────────────
//
// 부르는 곳을 «t() 호출»로만 세면 표에 담아 두고 나중에 t()에 넣는 키(예: 진행 단계
// 라벨 표)가 전부 고아로 잡힌다. 그래서 소스 어딘가에 그 키 문자열이 **리터럴로**
// 등장하기만 해도 쓰이는 것으로 본다 — 고아 목록은 «지울 후보»지 오류가 아니므로
// 이 정도 느슨함이 맞다.
// 주의: **밑줄→점 역변환으로 찾으면 안 된다.** norm은 점만 밑줄로 바꾸므로 세그먼트
// 이름 자체에 밑줄이 있는 키(content_genChip_stage_align ← 'content.genChip.stage_align')는
// 역변환이 원래 문자열을 복원하지 못한다. 그래서 반대로 **소스 리터럴을 정규화해서** 비교한다.
const allSource = [...tsFiles, ...htmlFiles].map(f => readFileSync(f, 'utf8')).join('\n');
const literalKeys = new Set();
for (const m of allSource.matchAll(/(['"])([A-Za-z][A-Za-z0-9_.]*\.[A-Za-z][A-Za-z0-9_.]*)\1/g)) {
  literalKeys.add(norm(m[2]));
}
const referencedNorm = new Set([...refs.keys()].map(norm));
const orphans = [...allKeys].filter(k => !referencedNorm.has(k) && !literalKeys.has(k));
console.log(`\nINFO: 아무 데서도 안 부르는 키 ${orphans.length}개`);
for (const k of orphans.slice(0, 40)) console.log(`  ${k}`);
if (orphans.length > 40) console.log(`  … 외 ${orphans.length - 40}개`);

console.log(failed ? '\nI18N KEY CHECK: FAIL' : '\nI18N KEY CHECK: PASS');
process.exitCode = failed ? 1 : 0;
