// 대량 렌더 E2E — 어젯밤 인제스트된 인기 보카로곡이 실브라우저에서 ko/en 자동 로딩되는지.
//
// **프로드 대상, 읽기 전용.** 익명(키 없음)으로 돈다 — 조회는 무제한이고, en 패스는
// en 레이어가 이미 있는 곡만 골라 LLM 번역 생성(쓰기)이 트리거되지 않게 한다.
//
// 곡 목록은 2026-07-28 야간 체인 chunk 0-1의 성공 곡에서 뽑았다(레이어 실측 포함).
// 검증: 싱크 라인 렌더 + 발음 표기(ko=한글, en=로마자) + 번역(ko=한글, en=라틴).
//
// 실행: EVERYRIC_E2E_HEADLESS=1 node scripts/bulk-render-e2e.mjs
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const SERVER = process.env.EVERYRIC_E2E_SERVER ?? 'https://everyric.moref.co';

// (video_id, slug) — ko 패스 대상. 전부 어젯밤 생성 + ko 레이어 실측.
const KO_SONGS = [
  ['Mj38FoEYVGA', 'become-suspended-animation'],
  ['q9q-3jJgKpU', 'tears-of-garnet'],
  ['zKCCkbZOuvE', 'rondo-of-able-world'],
  ['uCR6PQI2s3I', 'case-mask-island-kamikakushi'],
  ['L0roQoTpMSo', 'virtual-citizen'],
  ['vKW0CLkA4P4', 'gossip-music-party-night'],
  ['t1cC3pdFODA', 'have-a-nightmare'],
  ['qXkkhP0d_iM', 'cryptid-of-autumn'],
  ['_B3ed2Z2qgs', 'flowers-mountains-and-funerals'],
  ['q7XUSu5-SFw', 'gadget-cheat'],
  ['CSJTVDxHPGY', 'fake-human-no-40'],
  ['9bX3NykfVIY', 'maked-up-story'],
];
// en 레이어 실측 보유곡만 — 레이어 없는 곡을 en으로 열면 LLM 생성(쓰기)이 나간다
const EN_SONGS = [
  ['Mj38FoEYVGA', 'become-suspended-animation'],
  ['q9q-3jJgKpU', 'tears-of-garnet'],
  ['zKCCkbZOuvE', 'rondo-of-able-world'],
  ['uCR6PQI2s3I', 'case-mask-island-kamikakushi'],
  ['L0roQoTpMSo', 'virtual-citizen'],
  ['t1cC3pdFODA', 'have-a-nightmare'],
  ['qXkkhP0d_iM', 'cryptid-of-autumn'],
  ['_B3ed2Z2qgs', 'flowers-mountains-and-funerals'],
  ['CSJTVDxHPGY', 'fake-human-no-40'],
  ['7snBeht9jYU', 'disappearance-from-home-boy-lost-girl'],
];

const HAS_HANGUL = /[가-힣]/;
const HAS_LATIN = /[a-zA-Z]{3,}/;

// EVERYRIC_E2E_VIDS="vid:slug,vid:slug" — 지정 시 그 목록만 ko 패스로 돌린다 (재현·회귀용)
const OVERRIDE = (process.env.EVERYRIC_E2E_VIDS ?? '')
  .split(',').map(s => s.trim()).filter(Boolean)
  .map(s => { const [vid, slug] = s.split(':'); return [vid, slug ?? vid]; });

const channel = process.env.EVERYRIC_E2E_CHANNEL
  ?? (process.env.EVERYRIC_E2E_HEADLESS === '1' ? 'chromium' : '');
const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'ey-bulk-')), {
  ignoreDefaultArgs: ['--disable-extensions'],
  ...(channel ? { channel } : {}),
  headless: process.env.EVERYRIC_E2E_HEADLESS === '1',
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
  ],
});

const KNOWN_EXT_ID = 'pkhjekjjpccigehfkljffncdhnhfmoob';
async function acquireServiceWorker() {
  for (let i = 0; i < 15; i++) {
    const sw = ctx.serviceWorkers()[0];
    if (sw) return sw;
    await new Promise(r => setTimeout(r, 1000));
  }
  const waker = await ctx.newPage();
  await waker.goto(`chrome-extension://${KNOWN_EXT_ID}/src/options.html`, { timeout: 15000 }).catch(() => {});
  for (let i = 0; i < 15; i++) {
    const sw = ctx.serviceWorkers()[0];
    if (sw) { await waker.close(); return sw; }
    await new Promise(r => setTimeout(r, 1000));
  }
  await waker.close();
  return ctx.waitForEvent('serviceworker', { timeout: 15000 });
}

async function setSettings(sw, patch) {
  await sw.evaluate(async p => {
    const { settings } = await chrome.storage.local.get('settings');
    await chrome.storage.local.set({ settings: { ...(settings ?? {}), ...p } });
  }, patch);
}

const readLines = () => {
  const root = document.getElementById('everyric-root')?.shadowRoot;
  if (!root) return null;
  const lines = root.querySelectorAll('.ey-line:not(.ey-line-plain)').length;
  if (lines === 0) return null;
  const prons = Array.from(root.querySelectorAll('.ey-line-pron')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
  const trs = Array.from(root.querySelectorAll('.ey-line-tr')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
  return {
    lines,
    pron: prons.length,
    pronSample: prons.find(p => p.length > 3) ?? '',
    tr: trs.length,
    trSample: trs.find(t => t.length > 3) ?? '',
  };
};

let pass = 0, fail = 0;
const failures = [];

async function testSong(page, vid, slug, lang) {
  const label = `${lang} ${slug} (${vid})`;
  try {
    await page.goto(`https://www.youtube.com/watch?v=${vid}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
    const r = await page.waitForFunction(readLines, null, { timeout: 75000, polling: 1000 }).then(h => h.jsonValue());
    const problems = [];
    if (!(r.lines > 5)) problems.push(`lines=${r.lines}`);
    if (!(r.pron > 0)) problems.push(`pron=${r.pron}`);
    if (lang === 'ko' && !HAS_HANGUL.test(r.pronSample)) problems.push(`pron not hangul: ${r.pronSample.slice(0, 30)}`);
    if (lang === 'en' && HAS_HANGUL.test(r.pronSample)) problems.push(`pron has hangul (romaji 기대): ${r.pronSample.slice(0, 30)}`);
    if (!(r.tr > 0)) problems.push(`tr=${r.tr}`);
    if (lang === 'ko' && !HAS_HANGUL.test(r.trSample)) problems.push(`tr not hangul: ${r.trSample.slice(0, 30)}`);
    if (lang === 'en' && !HAS_LATIN.test(r.trSample)) problems.push(`tr not latin: ${r.trSample.slice(0, 30)}`);
    if (problems.length === 0) {
      pass++;
      console.log(`PASS ${label} lines=${r.lines} pron=${r.pron} tr=${r.tr} | ${r.pronSample.slice(0, 24)} | ${r.trSample.slice(0, 24)}`);
    } else {
      fail++;
      failures.push(label);
      console.log(`FAIL ${label} — ${problems.join('; ')}`);
    }
  } catch (e) {
    fail++;
    failures.push(label);
    const st = await page.evaluate(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      return (root?.textContent ?? '(no root)').replace(/\s+/g, ' ').slice(0, 120);
    }).catch(() => '(page dead)');
    console.log(`FAIL ${label} — ${String(e).slice(0, 100)} | panel: ${st}`);
  }
}

try {
  const sw = await acquireServiceWorker();
  console.log('extension loaded:', sw.url().slice(0, 60));
  const extId = new URL(sw.url()).host;
  // SERVER가 프로드(기본값)면 무동작. EVERYRIC_E2E_SERVER로 로컬을 가리키면 실제
  // 옵션 페이지 "허용" 흐름을 재현해 optional_host_permissions를 부여한다.
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
  await setSettings(sw, {
    serverUrl: SERVER, apiKey: '',
    showTranslation: true, showPronunciation: true,
    translationLanguage: 'ko', uiLanguage: 'auto',
  });
  const page = ctx.pages()[0] ?? await ctx.newPage();

  const koList = OVERRIDE.length > 0 ? OVERRIDE : KO_SONGS;
  console.log(`── ko pass: ${koList.length} songs ──`);
  for (const [vid, slug] of koList) await testSong(page, vid, slug, 'ko');

  if (OVERRIDE.length === 0) {
    await setSettings(sw, { translationLanguage: 'en' });
    console.log(`── en pass: ${EN_SONGS.length} songs ──`);
    for (const [vid, slug] of EN_SONGS) await testSong(page, vid, slug, 'en');
  }

  console.log(`\nRESULT: pass=${pass} fail=${fail}${failures.length ? ' | failed: ' + failures.join(', ') : ''}`);
  await ctx.close();
  process.exit(fail === 0 ? 0 : 1);
} catch (e) {
  console.log('HARNESS_FAIL', String(e).slice(0, 300));
  await ctx.close();
  process.exit(1);
}
