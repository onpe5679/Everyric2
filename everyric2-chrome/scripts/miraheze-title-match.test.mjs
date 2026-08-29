// Miraheze 제목 검증 회귀 — esbuild로 실제 TypeScript 소스를 실행한다.
// 네트워크는 가짜 fetch로 가로채 search/parse 결과와 일본어 가사 표를 독립적으로 검증한다.
import { build } from 'esbuild';
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join, resolve } from 'path';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const entry = resolve(__dirname, '../src/lib/miraheze.ts');
const result = await build({
  entryPoints: [entry],
  bundle: false,
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node18',
  loader: { '.ts': 'ts' },
});
const outDir = mkdtempSync(join(tmpdir(), 'miraheze-title-match-'));
const outFile = join(outDir, 'miraheze.mjs');
writeFileSync(outFile, result.outputFiles[0].text);
const { mirahezeLookup, titleMatchesCandidate } = await import(pathToFileURL(outFile).href);

const LYRICS_HTML = [
  '<table class="lyrics-table" id="lyrics-1">',
  '<tr class="lyrics-table-header">',
  '<th class="lyrics-jp">Japanese</th><th class="lyrics-romaji">Romaji</th>',
  '<th class="lyrics-en">English</th></tr>',
  '<tr><td>正しいダミー歌詞</td><td>tadashii damii kashi</td><td>verified dummy lyric</td></tr>',
  '</table>',
].join('');

let scenario = { searchHits: {}, pages: {} };
let parseRequests = 0;
globalThis.fetch = async rawUrl => {
  const url = new URL(String(rawUrl));
  const action = url.searchParams.get('action');
  if (action === 'query') {
    const query = url.searchParams.get('srsearch') ?? '';
    return response({ query: { search: scenario.searchHits[query] ?? [] } });
  }
  if (action === 'parse') {
    parseRequests++;
    const pageid = Number(url.searchParams.get('pageid'));
    const page = scenario.pages[pageid];
    return response(page ? { parse: { title: page.title, text: { '*': page.html } } } : {});
  }
  return response({}, false);
};

function response(body, ok = true) {
  return { ok, json: async () => body };
}

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail === undefined ? '' : ` = ${JSON.stringify(detail)}`}`);
  if (!ok) failed = true;
}

function useScenario(searchHits, pages) {
  scenario = { searchHits, pages };
  parseRequests = 0;
}

// 순수 제목 계약: 표기 정규화는 하되 일반 문자가 이어진 부분열은 거절한다.
for (const [candidate, pageTitle, expected] of [
  ['Ｒｏｋｉ', 'roki (song)', true],
  ['ロキ', 'ロキ (Roki)', true],
  ['フラジール', 'フラジール/nulut', true],
  ['Wonder', 'Wonderland', false],
  ['Wonder', 'Tenshi', false],
  ['Whole Blue World', 'トコヨトキヨ (Tokoyo Tokiyo)', false],
]) {
  check(
    titleMatchesCandidate(candidate, pageTitle) === expected,
    `title contract ${candidate} -> ${pageTitle}`,
    expected,
  );
}

// 실사용 오채택: 검색 1위의 다른 곡이 정상 일본어 가사 표를 가져도 채택하지 않는다.
for (const [query, wrongTitle] of [
  ['Wonder', 'Tenshi'],
  ['Whole Blue World', 'トコヨトキヨ (Tokoyo Tokiyo)'],
]) {
  useScenario(
    { [query]: [{ pageid: 1, title: wrongTitle }] },
    { 1: { title: wrongTitle, html: LYRICS_HTML } },
  );
  const got = await mirahezeLookup(query);
  check(got === null, `${query} must not adopt ${wrongTitle}`, got?.pageTitle);
  check(parseRequests === 0, `${query} rejects the unverified hit before parse`, parseRequests);
}

// 실검증 정상 경로: 검색 1위가 아니어도 제목 접두가 맞는 ロキ/シアンブルー 곡 페이지는 유지한다.
for (const [query, pageTitle] of [
  ['ロキ', 'ロキ (Roki)'],
  ['シアンブルー', 'シアンブルー (Cyan Blue)'],
]) {
  useScenario(
    { [query]: [{ pageid: 1, title: 'Unrelated album' }, { pageid: 2, title: pageTitle }] },
    { 2: { title: pageTitle, html: LYRICS_HTML } },
  );
  const got = await mirahezeLookup(query);
  check(got?.pageTitle === pageTitle, `${query} keeps its verified page`, got?.pageTitle);
  check(got?.lines?.length === 1, `${query} keeps the Japanese lyrics table`, got?.lines?.length);
}

// search 제목은 맞아도 parse가 돌려준 정규 제목이 다르면 채택하지 않는다.
useScenario(
  { Wonder: [{ pageid: 1, title: 'Wonder (Song)' }] },
  { 1: { title: 'Tenshi', html: LYRICS_HTML } },
);
const redirected = await mirahezeLookup('Wonder');
check(redirected === null, 'canonical parse title is revalidated', redirected?.pageTitle);

// 동명이곡 접두가 둘 이상이면 검색 순위 첫 항목을 자동 채택하지 않는다.
useScenario(
  {
    Scream: [
      { pageid: 1, title: 'Scream/Naoki' },
      { pageid: 2, title: 'Scream/Umetora' },
    ],
  },
  {
    1: { title: 'Scream/Naoki', html: LYRICS_HTML },
    2: { title: 'Scream/Umetora', html: LYRICS_HTML },
  },
);
const ambiguous = await mirahezeLookup('Scream');
check(ambiguous === null, 'ambiguous prefix titles fail closed', ambiguous?.pageTitle);
check(parseRequests === 2, 'ambiguous lyric pages are both inspected before rejection', parseRequests);

// 제목 접두가 같은 앨범/목록 페이지는 가사 표가 없으므로 진짜 곡을 모호하게 만들지 않는다.
useScenario(
  {
    ロキ: [
      { pageid: 1, title: 'ロキ (Roki) Album' },
      { pageid: 2, title: 'ロキ (Roki)' },
    ],
  },
  {
    1: { title: 'ロキ (Roki) Album', html: '<div>album page</div>' },
    2: { title: 'ロキ (Roki)', html: LYRICS_HTML },
  },
);
const songNotAlbum = await mirahezeLookup('ロキ');
check(songNotAlbum?.pageTitle === 'ロキ (Roki)', 'non-lyric album does not create false ambiguity', songNotAlbum?.pageTitle);

console.log(failed ? '\nMIRAHEZE TITLE MATCH TEST: FAIL' : '\nMIRAHEZE TITLE MATCH TEST: PASS');
process.exitCode = failed ? 1 : 0;
