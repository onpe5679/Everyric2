// 피드백 세대 귀속 계약 — 실제 everyric-api.ts를 번들해 sync_id 문자열과 명시적 null이
// JSON 본문에서 보존되는지 검증한다. 네트워크는 가짜 fetch로 가로챈다.
import { build } from 'esbuild';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, join, resolve } from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const result = await build({
  entryPoints: [resolve(__dirname, '../src/lib/everyric-api.ts')],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node18',
  loader: { '.ts': 'ts' },
});
const outDir = mkdtempSync(join(tmpdir(), 'feedback-sync-id-'));
const outFile = join(outDir, 'everyric-api.mjs');
writeFileSync(outFile, result.outputFiles[0].text);
const {
  generateSyncFromCaption, lookupSync, submitFeedback, vocaroMatch,
} = await import(pathToFileURL(outFile).href);

const bodies = [];
const urls = [];
globalThis.fetch = async (url, init = {}) => {
  urls.push(String(url));
  if (init.body) bodies.push(JSON.parse(init.body));
  return { ok: true, status: 200, json: async () => ({ ok: true }) };
};

const server = { serverUrl: 'https://everyric.test' };
await submitFeedback(server, {
  video_id: 'FEEDFEEDFB1',
  sync_id: '11111111-2222-3333-4444-555555555555',
  rating: 1,
  depth: 'fast',
});
await submitFeedback(server, {
  video_id: 'FEEDFEEDFB1',
  sync_id: null,
  rating: 2,
  category: 'lyrics',
});
await submitFeedback(server, {
  video_id: 'FEEDFEEDFB1',
  rating: 3,
});
await generateSyncFromCaption(server, {
  video_id: 'FEEDFEEDFB1',
  title: 'POLARIS',
  artist: 'Patterns',
});
await lookupSync(server, 'FEEDFEEDFB1', {
  title: 'POLARIS',
  artist: 'Patterns',
  titleEvidence: 'channel_reversed',
});
await vocaroMatch(server, '노래 입니다', undefined, { matchMode: 'search' });

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label} = ${JSON.stringify(detail)}`);
  if (!ok) failed = true;
}

check(
  bodies[0]?.sync_id === '11111111-2222-3333-4444-555555555555',
  '표시 중 sync_id가 그대로 직렬화됨',
  bodies[0],
);
check(
  Object.hasOwn(bodies[1] ?? {}, 'sync_id') && bodies[1].sync_id === null,
  '비 Everyric 화면의 명시적 null이 생략되지 않음',
  bodies[1],
);
check(
  !Object.hasOwn(bodies[2] ?? {}, 'sync_id'),
  '구형 호출의 undefined sync_id는 키 자체가 생략됨',
  bodies[2],
);
check(
  bodies[3]?.title === 'POLARIS' && bodies[3]?.artist === 'Patterns',
  '자막 생성도 교정된 canonical title/artist를 보존',
  bodies[3],
);
check(
  urls.some(url => url.includes('title_evidence=channel_reversed')),
  '검증된 제목 교정 근거가 조회 백필 경계까지 직렬화됨',
  urls,
);
check(
  urls.some(url => url.includes('/api/vocaro/match?') && url.includes('mode=search')),
  '느슨한 매칭 모드는 수동 검색 요청에만 명시됨',
  urls,
);

console.log(failed ? '\nFEEDBACK SYNC ID TEST: FAIL' : '\nFEEDBACK SYNC ID TEST: PASS');
process.exitCode = failed ? 1 : 0;
