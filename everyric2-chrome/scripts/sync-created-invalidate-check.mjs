// existsCache 무효화 수리 검증(감사 A3 후속) — SYNC_CREATED 메시지가 실제로 background.ts
// existsCache를 지우는지 확인한다. 실제 DB/서버 상태에 의존하면 외부 요인(WAL 가시성 등
// 이 세션에서 원인 미상으로 재현된 지연)에 흔들릴 수 있어, 대신 라우트 가로채기로
// "서버가 방금 답을 바꿨다"를 완전히 통제한다 — 검증 대상은 오직 캐시 무효화 로직이다.
// taskkill 없음(팀리드 공지) — 이 스크립트 전용 고정 user-data-dir.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdirSync } from 'fs';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const SERVER = 'http://127.0.0.1:8000';
const VID = 'ZZROUTEFAKE';

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const profileDir = 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad\\ey-sync-created-profile';
mkdirSync(profileDir, { recursive: true });

const ctx = await chromium.launchPersistentContext(profileDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1200, height: 800 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required',
  ],
});

// 서버의 진짜 응답과 무관하게, ZZROUTEFAKE에 대해서만 우리가 답을 통제한다.
// (이 컨텍스트로 나가는 모든 요청에 적용 — 배경 스크립트의 fetch도 여기서 잡힌다)
let routeServerSaysExists = false;
let routeCallCount = 0;
await ctx.route('**/api/sync/exists', async (route) => {
  const req = route.request();
  const body = JSON.parse(req.postData() ?? '{}');
  if (Array.isArray(body.video_ids) && body.video_ids.includes(VID)) {
    routeCallCount++;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ exists: { [VID]: routeServerSaysExists } }),
    });
    return;
  }
  await route.continue();
});

const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url, uiLanguage: 'ko' } });
}, SERVER);

// SW 자기 자신에게 sendMessage하면 "Receiving end does not exist"로 실패한다(실측) —
// 실제 확장 페이지(옵션 페이지)에서 보낸다. content script와 동일한 경로.
const msgPage = await ctx.newPage();
await msgPage.goto(`chrome-extension://${extId}/src/options.html`, { waitUntil: 'domcontentloaded' });

async function syncExists(videoId) {
  const res = await msgPage.evaluate((vid) => chrome.runtime.sendMessage({ type: 'SYNC_EXISTS', payload: { videoIds: [vid] } }), videoId);
  return res?.data?.[videoId] ?? null;
}
async function syncCreated(videoId) {
  return msgPage.evaluate((vid) => chrome.runtime.sendMessage({ type: 'SYNC_CREATED', payload: { videoId: vid } }), videoId);
}

// 1) 서버(우리 라우트)가 "없음"이라고 답하는 상태에서 조회 — 미스 캐시(2분 TTL) 생성
routeServerSaysExists = false;
const r1 = await syncExists(VID);
info('1차 조회(서버=없음)', { result: r1, routeCallCount });
check(r1 === false, '1차: 없음으로 응답', r1);
const callsAfter1 = routeCallCount;

// 2) 서버 쪽(라우트) 답을 "있음"으로 바꾼다 — 실제 생성 완료를 흉내
routeServerSaysExists = true;

// 3) SYNC_CREATED 없이 재조회 — 캐시가 살아 있으면 여전히 false + 네트워크 호출 없음(캐시 히트)
const r2 = await syncExists(VID);
info('2차 조회(무효화 없이, 서버는 이미 있음으로 바뀜)', { result: r2, routeCallCount });
check(r2 === false, '2차: 캐시가 살아 있어 여전히 없음으로 응답(버그 재현)', r2);
check(routeCallCount === callsAfter1, '2차: 캐시 히트라 서버에 새 요청이 안 나감(네트워크 호출 수 불변)', { callsAfter1, now: routeCallCount });

// 4) SYNC_CREATED 발사 — 캐시 무효화
const createdRes = await syncCreated(VID);
info('SYNC_CREATED 응답', createdRes);
check(createdRes?.data?.ok === true, 'SYNC_CREATED가 정상 응답(ok:true)', createdRes);

// 5) 재조회 — 이번엔 캐시가 비어 서버(라우트)에 다시 묻고 true를 받아야 한다
const r3 = await syncExists(VID);
info('3차 조회(SYNC_CREATED 이후)', { result: r3, routeCallCount });
check(r3 === true, '3차: 무효화 이후 서버 재조회로 있음으로 갱신됨(수리 확인)', r3);
check(routeCallCount === callsAfter1 + 1, '3차: 무효화 이후 실제로 새 네트워크 요청이 나감', { callsAfter1, now: routeCallCount });

await ctx.close().catch(() => {});
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
