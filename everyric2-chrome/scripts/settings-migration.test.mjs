// pipKeepPanel 1회성 마이그레이션 검증 — 브라우저 없이 chrome.storage를 대역으로 세운다.
//
// 검증해야 하는 것이 «시간에 걸친 성질»(한 번만 돈다 / 되돌린 선택을 존중한다)이라
// 스토리지 대역으로 재로드를 흉내 내는 편이 실브라우저보다 정확하고 빠르다.
// 쓰기 횟수까지 세서 «dist 재로드마다 반복되지 않음»을 직접 증명한다.
//
// 실행: node scripts/settings-migration.test.mjs
import { createRequire } from 'node:module';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repo = resolve(__dirname, '..');

// settings.ts만 번들해서 실제 코드를 그대로 돌린다(로직 복제 금지 — 복제하면 시험 대상이 달라진다)
const work = mkdtempSync(join(tmpdir(), 'everyric-settings-test-'));
const entry = join(work, 'entry.ts');
writeFileSync(entry, `export * from '${resolve(repo, 'src/lib/settings').replace(/\\/g, '/')}';\n`);
const bundle = join(work, 'settings.mjs');
// .bin 심은 윈도우에서 .cmd라 셸을 타야 한다 — 플랫폼 분기 대신 JS API를 직접 쓴다
const esbuild = createRequire(resolve(repo, 'package.json'))('esbuild');
await esbuild.build({
  entryPoints: [entry], bundle: true, format: 'esm', outfile: bundle, platform: 'neutral',
});

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** chrome.storage.local 대역 — 쓰기 횟수를 센다 */
function installChrome(initial) {
  const store = initial ? structuredClone(initial) : {};
  const stats = { writes: 0 };
  globalThis.chrome = {
    storage: {
      local: {
        async get(key) {
          if (key === null || key === undefined) return structuredClone(store);
          if (typeof key === 'string') return key in store ? { [key]: structuredClone(store[key]) } : {};
          return {};
        },
        async set(obj) {
          stats.writes++;
          Object.assign(store, structuredClone(obj));
        },
      },
    },
  };
  return { store, stats };
}

const MARKER = 'pipKeepPanel-default-false-2026-08';

// 모듈 상태(lastKnown 등)를 실제 «재로드»처럼 초기화하려면 매번 새로 import해야 한다.
// 윈도우 절대경로는 그대로 넣으면 'c:'가 URL 스킴으로 읽혀 거부된다 — file:// URL로 준다.
let importSeq = 0;
const bundleUrl = pathToFileURL(bundle).href;
const freshImport = () => import(`${bundleUrl}?v=${++importSeq}`);

// ── 1) 기존 사용자: 옛 기본값 true를 물고 있다 → 딱 한 번 내려간다 ──
{
  const { store, stats } = installChrome({
    settings: { pipKeepPanel: true, serverUrl: 'https://x', fontSize: 'large' },
  });
  const m = await freshImport();
  const s1 = await m.getSettings();
  check(s1.pipKeepPanel === false, '기존 사용자(true)의 pipKeepPanel이 내려간다', s1.pipKeepPanel);
  check(Array.isArray(s1.settingsMigrations) && s1.settingsMigrations.includes(MARKER),
    '마커가 반환값에 있다', s1.settingsMigrations);
  check(store.settings?.settingsMigrations?.includes(MARKER) === true,
    '마커가 **디스크에 저장**된다', store.settings?.settingsMigrations);
  check(store.settings?.pipKeepPanel === false, '내려간 값도 디스크에 저장된다', store.settings?.pipKeepPanel);
  check(store.settings?.fontSize === 'large', '다른 설정은 보존된다', store.settings?.fontSize);
  const writesAfterFirst = stats.writes;
  check(writesAfterFirst >= 1, '마이그레이션이 쓰기를 1회 이상 냈다', writesAfterFirst);

  // 같은 프로세스에서 여러 번 더 읽어도 추가 쓰기가 없어야 한다
  await m.getSettings();
  await m.getSettings();
  check(stats.writes === writesAfterFirst,
    '같은 세션에서 재조회해도 추가 쓰기 없음', { before: writesAfterFirst, after: stats.writes });
}

// ── 2) 재로드: 마커가 있으면 다시 돌지 않는다 ──────────────────────
{
  const { store, stats } = installChrome({
    settings: { pipKeepPanel: false, settingsMigrations: [MARKER], fontSize: 'small' },
  });
  const m = await freshImport();
  await m.getSettings();
  await m.getSettings();
  check(stats.writes === 0, 'dist 재로드(마커 있음)에서 쓰기가 아예 없다 — 반복 실행 없음', stats.writes);
  check(store.settings.fontSize === 'small', '저장본이 그대로다', store.settings.fontSize);
}

// ── 3) 사용자가 다시 켠 선택은 존중된다 ───────────────────────────
{
  const { store, stats } = installChrome({
    settings: { pipKeepPanel: true, settingsMigrations: [MARKER] },
  });
  const m = await freshImport();
  const s = await m.getSettings();
  check(s.pipKeepPanel === true, '마커가 있으면 사용자가 켠 true를 안 건드린다', s.pipKeepPanel);
  check(stats.writes === 0, '그때 쓰기도 없다', stats.writes);
  check(store.settings.pipKeepPanel === true, '디스크 값도 그대로', store.settings.pipKeepPanel);
}

// ── 4) 마이그레이션 직후 사용자가 다시 켜면, 재로드해도 유지된다 ──
{
  const { store } = installChrome({ settings: { pipKeepPanel: true } });
  const m1 = await freshImport();
  await m1.getSettings();                       // 1회 내려감 + 마커 저장
  await m1.saveSettings({ pipKeepPanel: true }); // 사용자가 설정에서 다시 켬
  check(store.settings.pipKeepPanel === true, '사용자가 다시 켠 값이 저장된다', store.settings.pipKeepPanel);

  const m2 = await freshImport();                // 재로드
  const s = await m2.getSettings();
  check(s.pipKeepPanel === true,
    '**되돌린 선택이 재로드 후에도 유지된다**(재마이그레이션 없음)', s.pipKeepPanel);
}

// ── 5) 새 설치: 기본이 false이고 마커가 남아 다시 안 돈다 ──────────
{
  const { store, stats } = installChrome(null);
  const m = await freshImport();
  const s = await m.getSettings();
  check(s.pipKeepPanel === false, '새 설치 기본값이 false', s.pipKeepPanel);
  check(store.settings?.settingsMigrations?.includes(MARKER) === true,
    '새 설치에도 마커가 남아 나중에 안 돈다', store.settings?.settingsMigrations);
  const w = stats.writes;
  const m2 = await freshImport();
  await m2.getSettings();
  check(stats.writes === w, '새 설치도 재로드에서 추가 쓰기 없음', { before: w, after: stats.writes });
}

// ── 6) 레인 발음 위치 어휘 마이그레이션(값 기반)은 그대로 산다 ────
{
  installChrome({ settings: { pitchPronPosition: 'note', settingsMigrations: [MARKER] } });
  const m = await freshImport();
  const s = await m.getSettings();
  check(s.pitchPronPosition === 'off', "'note' → 'off' 어휘 마이그레이션 유지", s.pitchPronPosition);
}
{
  installChrome({ settings: { pitchPronPosition: 'both', settingsMigrations: [MARKER] } });
  const m = await freshImport();
  const s = await m.getSettings();
  check(s.pitchPronPosition === 'bottom', "구 'both' → 'bottom' 유지", s.pitchPronPosition);
}

rmSync(work, { recursive: true, force: true });
console.log(failed ? '\nSETTINGS MIGRATION TEST: FAIL' : '\nSETTINGS MIGRATION TEST: PASS');
process.exitCode = failed ? 1 : 0;
