// song-detector.ts 제목 분해 회귀·사고 재현 검증 — esbuild로 실제 소스를 트랜스파일해
// 그대로 실행한다(로직 복제 아님). splitArtistTitle/cleanTitle을 detectSong의 mediaSession
// 경로와 동일한 순서(splitArtistTitle(cleanTitle(raw)))로 체이닝해 검증한다.
import { build } from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
const entry = resolve(__dirname, '../src/lib/song-detector.ts');

const result = await build({
  entryPoints: [entry],
  bundle: false, // 이 모듈은 '../types'만 import(type-only)하므로 번들 불필요
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node18',
  loader: { '.ts': 'ts' },
});
const outDir = mkdtempSync(join(tmpdir(), 'song-detector-'));
const outFile = join(outDir, 'song-detector.mjs');
writeFileSync(outFile, result.outputFiles[0].text);
const mod = await import('file://' + outFile.replace(/\\/g, '/'));
const { cleanTitle, splitArtistTitle } = mod;

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}

// mediaSession 경로가 실제로 하는 체이닝과 동일하게: splitArtistTitle(cleanTitle(raw))
function detect(raw) {
  return splitArtistTitle(cleanTitle(raw));
}

// ── 케이스 표 ──────────────────────────────────────────────────
const cases = [
  {
    label: '사고 재현: PrXtrTgMDEg(한국어 커버, 일본식 제목)',
    raw: '【韓国語で】네모네모(NEMONEMO) / YENA 歌ってみた｜Kotoha',
    expectArtist: 'YENA',
    // title은 "NEMONEMO"를 부분 문자열로 포함하면 유효 근사로 인정(팀 지시)
    titleContains: 'NEMONEMO',
  },
  {
    label: '표준 "아티스트 - 곡명" (하이픈, 기존 회귀 없어야 함)',
    raw: 'IU - Blueming',
    expectArtist: 'IU',
    expectTitle: 'Blueming',
  },
  {
    label: 'em dash "아티스트 — 곡명" (기존 회귀 없어야 함)',
    raw: 'NewJeans — Super Shy',
    expectArtist: 'NewJeans',
    expectTitle: 'Super Shy',
  },
  {
    label: '구분자 없는 단순 제목 — 분해 안 되고 artist=null(호출부가 meta.artist/channel로 폴백)',
    raw: 'Butter (Official MV)',
    expectArtist: null,
    expectTitle: 'Butter', // (Official MV) 소음은 cleanTitle이 이미 제거
  },
  {
    label: '슬래시 있지만 커버 지시어 없음 — 잘못 쪼개면 안 됨(보수적 게이트 확인)',
    raw: 'Track A / Track B',
    expectArtist: null,
    expectTitle: 'Track A / Track B',
  },
  {
    label: '한국어 "커버" 지시어',
    raw: '좋은 날 / 아이유 커버',
    expectArtist: '아이유',
    expectTitle: '좋은 날',
  },
  {
    label: '반각 파이프(공백 없음) + cover 지시어',
    raw: 'Song Title / Artist Name Cover|SomeChannel',
    expectArtist: 'Artist Name',
    expectTitle: 'Song Title',
  },
  {
    label: '기존 파이프 규칙(공백 있는 " | ") — 회귀 없이 그대로 동작(슬래시 없어 커버 규칙 비발동)',
    raw: 'Legacy Artist | Legacy Title',
    expectArtist: 'Legacy Artist',
    expectTitle: 'Legacy Title',
  },
  {
    label: '채널=아티스트 공식 영상, 대괄호만 있는 제목 — 분해 안 됨(정상)',
    raw: '(MV) Dynamite',
    expectArtist: null,
    expectTitle: 'Dynamite',
  },
  {
    label: '앞쪽이 비어 슬래시로 시작하는 기형 제목 — null 처리로 안전 폴백',
    raw: '/ Artist 커버',
    expectArtist: null, // slash.index<=0 이라 splitCoverTitle 발동 안 함, 일반 규칙도 매치 없음
    expectTitle: '/ Artist 커버',
  },
];

for (const c of cases) {
  const got = detect(c.raw);
  if ('expectTitle' in c) {
    check(got.title === c.expectTitle, `${c.label} — title`, { raw: c.raw, got: got.title, want: c.expectTitle });
  }
  if ('titleContains' in c) {
    check(got.title.includes(c.titleContains), `${c.label} — title(부분포함)`, { raw: c.raw, got: got.title, want_substring: c.titleContains });
  }
  check(got.artist === c.expectArtist, `${c.label} — artist`, { raw: c.raw, got: got.artist, want: c.expectArtist });
}

console.log(failed ? '\nSONG-DETECTOR SPLIT TEST: FAIL' : '\nSONG-DETECTOR SPLIT TEST: PASS');
process.exitCode = failed ? 1 : 0;
