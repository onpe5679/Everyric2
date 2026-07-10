// 사용자 시나리오 재현: vocaro를 거친 적 없는 "새 프로필"로 로키 영상을 열었을 때
//   - 서버 싱크에 저장된 발음/사람 번역이 그대로 표시되는지
//   - 번역 표시 ON이어도 [NO API KEY]가 나타나지 않는지 (사람 번역 우선)
//   - PiP 음정 바(신규 UI)가 그려지고, pitchGuide 설정으로 껐다 켤 수 있는지
// 사전 조건: 실서버(:8000) + 해당 곡 싱크(발음/번역/notes 포함)가 DB에 있어야 한다.
// 실행: node scripts/fresh-profile-check.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-fresh-'));

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

const ctx = await chromium.launchPersistentContext(userDataDir, {
  channel: process.env.EVERYRIC_E2E_CHANNEL ?? 'msedge',
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
    '--window-position=40,40',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  // 사용자 설정 재현: 번역 표시 ON (문제가 발생했던 조건)
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), { showTranslation: true, translationLanguage: 'ko' });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line:not(.ey-line-plain)').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.waitForTimeout(2500); // 번역 로드가 시도된다면 그 뒤 상태를 봐야 한다

  const panel = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const text = root?.textContent ?? '';
    return {
      lines: root?.querySelectorAll('.ey-line').length ?? 0,
      pron: root?.querySelectorAll('.ey-line-pron').length ?? 0,
      tr: root?.querySelectorAll('.ey-line-tr').length ?? 0,
      noApiKey: text.includes('[NO API KEY]'),
      firstPron: root?.querySelector('.ey-line-pron')?.textContent?.slice(0, 40) ?? '',
    };
  });
  check(panel.pron > 40, `새 프로필에서 발음 표기 (서버 저장분, ${panel.pron}줄)`, panel);
  check(panel.tr > 40, `새 프로필에서 사람 번역 (${panel.tr}줄)`);
  check(!panel.noApiKey, '[NO API KEY] 없음');

  // 재생 보장 후 PiP 열기
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1500);
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(3000);

  const pip = await page.evaluate(() => {
    const w = window.documentPictureInPicture?.window;
    let pitch = { present: false, visible: false, drawnPx: 0 };
    const c = w?.document.querySelector('.ey-pip-pitch');
    if (c) {
      pitch.present = true;
      pitch.visible = w.getComputedStyle(c).display !== 'none';
      try {
        const data = c.getContext('2d').getImageData(0, 0, c.width || 1, c.height || 1).data;
        for (let i = 3; i < data.length; i += 4) if (data[i] > 0) pitch.drawnPx++;
      } catch { /* ignore */ }
    }
    return {
      open: !!w,
      pron: w?.document.querySelector('.ey-pip-pron')?.textContent?.slice(0, 40) ?? '',
      pitch,
    };
  });
  check(pip.open, 'PiP 열림');
  check(pip.pitch.present && pip.pitch.visible && pip.pitch.drawnPx > 50, '음정 바 (신규 UI) 렌더링', pip.pitch);
  if (pip.pron) console.log('PASS: PiP 발음 =', JSON.stringify(pip.pron));

  const pipPage = ctx.pages().find(p => p !== page);
  if (pipPage) {
    await pipPage.screenshot({ path: resolve(__dirname, '../fresh-pip.png') });
    console.log('screenshot: fresh-pip.png');
  }
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(1000);

  // pitchGuide OFF → 리로드 → PiP에서 음정 바 숨김 확인
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }),
    { showTranslation: true, translationLanguage: 'ko', pitchGuide: false });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1000);
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(2500);
  const off = await page.evaluate(() => {
    const w = window.documentPictureInPicture?.window;
    const c = w?.document.querySelector('.ey-pip-pitch');
    return {
      open: !!w,
      pitchVisible: c ? w.getComputedStyle(c).display !== 'none' : false,
    };
  });
  check(off.open && !off.pitchVisible, 'pitchGuide OFF → 음정 바 숨김', off);

  console.log(failed ? 'FRESH PROFILE CHECK: FAIL' : 'FRESH PROFILE CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('FRESH PROFILE CHECK: ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
