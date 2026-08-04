/**
 * 실브라우저 하네스 공용 런처 — 여러 에이전트가 동시에 하네스를 돌려도 서로 안 밟게 하는 규약.
 *
 * 규약의 근거(a28540e 실측):
 *  - `taskkill /F /IM chrome.exe`는 이미지 이름으로 죽여 **남의 크롬까지 날린다** → 절대 안 쓴다.
 *    대신 매 실행이 자기 프로필·자기 창만 쓰고 끝나면 스스로 닫는다.
 *  - dist 스냅샷 경로를 **고정**해야 한다. 언팩 확장 ID는 로드 경로로 정해지므로, 경로가
 *    매번 다르면 확장 ID가 바뀌어 프로필에 저장된 host permission이 무효가 되고 권한 버블이
 *    되살아난다. 경로는 고정하고 내용만 갱신(cpSync 덮어쓰기)한다.
 *  - 스냅샷을 뜨는 또 다른 이유: 다른 에이전트가 리포를 빌드하는 중이면 dist/manifest.json이
 *    갈리는 순간이 있고, 그때 크롬은 "매니페스트 없음" 모달을 띄우며 연결을 영영 막는다.
 *    복사 직후 manifest를 파싱해 깨진 스냅샷이면 즉시 죽는다.
 *
 * 경로는 하네스 이름별로 갈라 둔다 — 두 하네스를 동시에 돌려도 프로필이 겹치지 않는다.
 * 환경변수로 재정의할 수 있다(EVERYRIC_E2E_DIST_DIR / EVERYRIC_E2E_PROFILE_DIR).
 */
import { chromium } from 'playwright';
import { cpSync, mkdirSync, readFileSync, rmSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join, resolve } from 'path';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(__dirname, '../..');
export const LOCAL_SERVER = 'http://127.0.0.1:8000';
export const PROD_SERVER = 'https://everyric.moref.co';

/** 로컬 주소만 optional_host_permissions라 승인 절차가 필요하다(프로드는 필수 권한) */
export const isLocalServer = (url) => /127\.0\.0\.1|localhost/.test(url);

const e2eRoot = () => join(tmpdir(), 'everyric-e2e-profiles');

/**
 * 확장을 물린 크롬을 띄운다.
 * @param {object} o
 * @param {string} o.name        하네스 이름 — 고정 dist·프로필 경로의 키가 된다
 * @param {string} [o.serverUrl] 기본 LOCAL_SERVER. 로컬이면 host permission을 확보한다
 * @param {number} [o.width]     뷰포트
 * @param {number} [o.height]
 * @param {boolean} [o.freshProfile] 프로필을 지우고 시작할지(권한 버블이 다시 뜬다)
 */
export async function launchE2E({
  name, serverUrl = LOCAL_SERVER, width = 1600, height = 1000, freshProfile = false,
} = {}) {
  if (!name) throw new Error('launchE2E: name이 필요하다(고정 경로 키)');
  // 원본 dist는 보통 리포의 dist/지만, 다른 에이전트가 쓰는 공유 dist를 덮지 않으려고
  // 격리 빌드(npx vite build --outDir …)를 가리켜야 할 때가 있다.
  const distSrc = resolve(REPO, process.env.EVERYRIC_E2E_DIST_SRC ?? 'dist');
  if (!existsSync(join(distSrc, 'manifest.json'))) {
    throw new Error(`dist가 없다(${distSrc}) — 먼저 npm run build 또는 npx vite build`);
  }
  const distDir = process.env.EVERYRIC_E2E_DIST_DIR ?? join(e2eRoot(), `${name}-dist`);
  const profileDir = process.env.EVERYRIC_E2E_PROFILE_DIR ?? join(e2eRoot(), `${name}-profile`);
  mkdirSync(distDir, { recursive: true });
  cpSync(distSrc, distDir, { recursive: true });
  const manifest = JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8')); // 깨졌으면 여기서 즉사
  if (freshProfile && existsSync(profileDir)) rmSync(profileDir, { recursive: true, force: true });
  mkdirSync(profileDir, { recursive: true });

  const ctx = await chromium.launchPersistentContext(profileDir, {
    ignoreDefaultArgs: ['--disable-extensions'],
    headless: false,
    viewport: { width, height },
    args: [
      `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
      '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=20,20',
    ],
  });
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
  const extId = new URL(sw.url()).host;
  if (isLocalServer(serverUrl)) {
    await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, serverUrl);
  }
  console.log(`[e2e] ${name}: 확장 ${manifest.version} / id ${extId.slice(0, 12)}… / 서버 ${serverUrl}`);
  return { ctx, sw, extId, version: manifest.version, distDir, profileDir };
}

/** 설정 저장소 읽기/쓰기 — 서비스워커를 거친다(콘텐츠 스크립트보다 이른 시점에도 된다) */
export function settingsIO(sw) {
  return {
    patch: (p) => sw.evaluate(async (v) => {
      const cur = (await chrome.storage.local.get('settings')).settings ?? {};
      await chrome.storage.local.set({ settings: { ...cur, ...v } });
    }, p),
    replace: (v) => sw.evaluate(async (s) => { await chrome.storage.local.set({ settings: s }); }, v),
    read: () => sw.evaluate(async () => (await chrome.storage.local.get('settings')).settings ?? {}),
  };
}

/** PASS/FAIL을 찍고 실패만 모아 둔다 — 각 하네스의 종료 코드 근거 */
export function makeCheck() {
  const failures = [];
  const check = (ok, label, detail) => {
    console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
    if (!ok) failures.push(label + (detail !== undefined ? ' = ' + JSON.stringify(detail) : ''));
    return ok;
  };
  return { check, failures };
}

/** 스크린샷 경로 — 다른 하네스와 같은 관례로 리포 루트에 남긴다 */
export function shotPath(name) {
  return resolve(REPO, `${name}.png`);
}
