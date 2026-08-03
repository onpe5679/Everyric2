// 사고 재현 종단 검증 — song-detector.ts의 실제 분해 결과를 lib/lrclib.ts의 실제
// fetchFromLrclib에 그대로 먹여 진짜 lrclib.net API가 히트하는지 확인한다(둘 다 esbuild로
// 트랜스파일한 실제 소스 — 로직 복제 아님). 공개 API라 가볍게 실행 가능.
import { build } from 'esbuild';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function transpile(relPath) {
  const entry = resolve(__dirname, relPath);
  const result = await build({
    entryPoints: [entry], bundle: false, write: false,
    format: 'esm', platform: 'node', target: 'node18', loader: { '.ts': 'ts' },
  });
  const outDir = mkdtempSync(join(tmpdir(), 'ey-live-'));
  const outFile = join(outDir, 'mod.mjs');
  writeFileSync(outFile, result.outputFiles[0].text);
  return import('file://' + outFile.replace(/\\/g, '/'));
}

const { cleanTitle, splitArtistTitle } = await transpile('../src/lib/song-detector.ts');
const { fetchFromLrclib } = await transpile('../src/lib/lrclib.ts');

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}

// 사고 영상(PrXtrTgMDEg)의 실제 mediaSession 제목을 그대로 재현 — detectSong의 mediaSession
// 분기와 동일한 체이닝(splitArtistTitle(cleanTitle(raw)))으로 title/artist를 뽑는다.
const rawTitle = '【韓国語で】네모네모(NEMONEMO) / YENA 歌ってみた｜Kotoha';
const split = splitArtistTitle(cleanTitle(rawTitle));
console.log('분해 결과:', JSON.stringify(split));

const song = {
  title: split.title,
  artist: split.artist, // 수리 전이었다면 여기가 채널명 "Kotoha"였을 것
  videoId: 'PrXtrTgMDEg',
  duration: 0, // mediaSession 단계에선 video duration을 아직 모를 수 있다 — 0으로 재현
};

const track = await fetchFromLrclib(song);
check(track !== null, '분해된 title/artist로 fetchFromLrclib이 실제 곡을 찾음', track && {
  trackName: track.trackName, artistName: track.artistName, hasLyrics: Boolean(track.plainLyrics || track.syncedLyrics),
});
if (track) {
  check(/nemonemo/i.test(track.trackName ?? ''), '찾은 트랙명이 NEMONEMO 계열', track.trackName);
  check(/yena/i.test(track.artistName ?? ''), '찾은 아티스트가 YENA 계열', track.artistName);
}

// 대조군 — 수리 전 상태(채널명을 artist로 오인)였다면 어떻게 되는지도 같이 확인해 둔다
// (팀 보고용 비교 자료 — 이 검증 자체가 PASS/FAIL을 좌우하진 않는다)
const brokenSong = { title: rawTitle, artist: 'Kotoha', videoId: 'PrXtrTgMDEg', duration: 0 };
const brokenTrack = await fetchFromLrclib(brokenSong);
console.log('INFO: 수리 전 상태(미분해 title + 채널명 artist) 재현 결과 =', brokenTrack ? JSON.stringify({ trackName: brokenTrack.trackName }) : 'null(미조달 — 사고 그대로 재현됨)');

console.log(failed ? '\nLRCLIB LIVE VERIFY: FAIL' : '\nLRCLIB LIVE VERIFY: PASS');
process.exitCode = failed ? 1 : 0;
