// 신/구 서버 응답과 vocaroRef 캐시가 자동 채택 경계를 우회하지 않는지 실제 TS 소스로 검증.
import { build } from 'esbuild';
import { mkdtempSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, join, resolve } from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const result = await build({
  entryPoints: [resolve(__dirname, '../src/lib/vocaro.ts')],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  target: 'node18',
  loader: { '.ts': 'ts' },
});
const outDir = mkdtempSync(join(tmpdir(), 'vocaro-compat-'));
const outFile = join(outDir, 'vocaro.mjs');
writeFileSync(outFile, result.outputFiles[0].text);
const {
  cachedVocaroSlug,
  makeVocaroRef,
  vocaroCacheIdentityKey,
  vocaroMatchResponseIsSafe,
  vocaroPageMatchesSafeMatch,
  vocaroResultMatchesEvidence,
} = await import(pathToFileURL(outFile).href);

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label} = ${JSON.stringify(detail)}`);
  if (!ok) failed = true;
}

const screamEvidence = { title: 'S.C.R.E.A.M', titleCandidates: ['S.C.R.E.A.M'] };
check(
  !vocaroMatchResponseIsSafe({ found: true, slug: 'scream', ja: 'SCREAM' }, screamEvidence),
  'matcher_version 없는 구서버 found는 자동 채택하지 않음',
  null,
);
check(
  !vocaroMatchResponseIsSafe({
    found: true, slug: 'scream', ja: 'SCREAM', matcher_version: 'identity-1', status: 'ambiguous',
  }, screamEvidence),
  'found+ambiguous 불가능 상태를 거절',
  null,
);
check(
  !vocaroMatchResponseIsSafe({
    found: true, slug: 'song', ja: 'Song', matcher_version: 'identity-1', status: 'matched',
  }, { title: 'Song (Remix)', titleCandidates: ['Song (Remix)'] }),
  '임의 Roman 괄호를 제거해 다른 곡으로 내리지 않음',
  null,
);
check(
  vocaroMatchResponseIsSafe({
    found: true, slug: 'roki', ja: 'ロキ', matcher_version: 'identity-1', status: 'matched',
  }, { title: 'ロキ (Roki)', titleCandidates: ['ロキ (Roki)'] }),
  '같은 항목의 실제 Roman slug 별칭만 허용',
  null,
);
check(
  !vocaroMatchResponseIsSafe({
    found: true, slug: 'scream', ja: 'SCREAM', matcher_version: 'identity-1', status: 'matched',
  }, screamEvidence),
  '기호가 다른 제목은 신서버 응답이어도 재검증에서 거절',
  null,
);
check(
  vocaroMatchResponseIsSafe({
    found: true, slug: 'roki', ja: 'ロキ', matcher_version: 'identity-1', status: 'matched',
  }, { title: 'ロキ', titleCandidates: ['ロキ'] }),
  '정상 identity 응답은 유지',
  null,
);
check(
  vocaroMatchResponseIsSafe({
    found: true,
    slug: 'cyan-blue',
    ja: 'シアンブルー',
    matcher_version: 'identity-1',
    status: 'matched',
    evidence_source: 'youtube_oembed',
    resolved_title: 'シアンブルー / ポリスピカデリー feat. 初音ミク',
    resolved_channel: 'Hatsune Miku',
  }, {
    title: 'シアンブルー / ポリスピカデリー feat. 初音ミク',
    titleCandidates: ['シアンブルー / ポリスピカデリー feat. 初音ミク'],
  }),
  '실제 H7PR video id 복구의 slash+feat 원제를 수용',
  null,
);
const koreanMatch = {
  found: true,
  slug: 'roki',
  ko: '로키',
  ja: 'ロキ',
  matcher_version: 'identity-1',
  status: 'matched',
};
check(
  vocaroPageMatchesSafeMatch(
    { pageUrl: 'x', pageTitle: 'ロキ', slug: 'roki', lines: [] },
    koreanMatch,
    { title: '로키', titleCandidates: ['로키'] },
  ),
  '검증된 한국어 별칭 뒤 일본어 원제 페이지를 유지',
  null,
);
check(
  !vocaroMatchResponseIsSafe({
    found: true,
    slug: 'wrong',
    ja: '완전히 다른 곡',
    matched_query: '완전히 다른 곡',
    matcher_version: 'identity-1',
    status: 'matched',
  }, { title: '로키', titleCandidates: ['로키'] }),
  '서버가 돌려준 matched_query 자체는 클라이언트 증거가 아님',
  null,
);
check(
  vocaroMatchResponseIsSafe({
    found: true,
    slug: 'polaris',
    ja: 'POLARIS',
    matcher_version: 'identity-1',
    status: 'matched',
    evidence_source: 'youtube_oembed',
    resolved_title: 'POLARIS - Patterns ft. @rino',
    resolved_channel: 'Patterns',
  }, { title: 'Patterns ft. @rino', titleCandidates: ['Patterns ft. @rino'] }),
  'video id 복구 제목은 채널로 방향을 검증한 경우만 수용',
  null,
);

const wrongPage = { pageUrl: 'x', pageTitle: '완전히 다른 곡', slug: 'wrong-song', lines: [] };
check(
  !vocaroResultMatchesEvidence(wrongPage, { title: '로키' }),
  '만료 인덱스가 가리킨 다른 페이지를 거절',
  null,
);
const wrongProducer = { pageUrl: 'x', pageTitle: 'Scream', slug: 'scream-umetora', lines: [] };
check(
  !vocaroResultMatchesEvidence(
    wrongProducer, { title: 'Scream', artist: 'Naoki', channel: 'Naoki' }, true,
  ),
  'legacy 자동 폴백의 다른 producer를 거절',
  null,
);
check(
  !vocaroResultMatchesEvidence(
    { pageUrl: 'x', pageTitle: 'Song', slug: 'song-eleven', lines: [] },
    { title: 'Song', artist: 'Eve', channel: 'Eve' },
    true,
  ),
  'producer 부분문자열 Eve는 Eleven과 일치하지 않음',
  null,
);
check(
  vocaroResultMatchesEvidence(
    { pageUrl: 'x', pageTitle: 'Song', slug: 'song-eve', lines: [] },
    { title: 'Song', artist: 'Eve', channel: 'Eve' },
    true,
  ),
  '정확한 producer suffix는 유지',
  null,
);

const key = vocaroCacheIdentityKey({
  title: 'ロキ', titleCandidates: ['ロキ'], artist: 'みきとP', channel: 'みきとP',
});
check(cachedVocaroSlug('roki', key) === null, '구형 문자열 ref 무효화', null);
check(cachedVocaroSlug({ slug: 'roki', t: 1 }, key) === null, '구형 객체 ref 무효화', null);
const now = 2_000_000_000_000;
const fresh = makeVocaroRef('roki', key, now);
check(cachedVocaroSlug(fresh, key, now) === 'roki', 'identity provenance가 맞는 ref만 재사용', fresh);
check(cachedVocaroSlug(fresh, vocaroCacheIdentityKey({ title: '다른 곡' }), now) === null, '다른 제목의 ref 무효화', null);
const otherChannelKey = vocaroCacheIdentityKey({
  title: 'ロキ', titleCandidates: ['ロキ'], artist: '다른 사람', channel: '다른 채널',
});
check(cachedVocaroSlug(fresh, otherChannelKey, now) === null, '같은 제목이어도 artist/channel 변경 시 ref 무효화', null);
const otherVersionKey = vocaroCacheIdentityKey({
  title: 'ロキ', titleCandidates: ['ロキ'], rawTitle: 'ロキ (Remix)', artist: 'みきとP', channel: 'みきとP',
});
check(cachedVocaroSlug(fresh, otherVersionKey, now) === null, '같은 정리 제목이어도 raw 버전 힌트 변경 시 ref 무효화', null);
check(cachedVocaroSlug(fresh, key, now + 24 * 60 * 60 * 1000 + 1) === null, '24시간 지난 ref 무효화', null);

console.log(failed ? '\nVOCARO COMPAT TEST: FAIL' : '\nVOCARO COMPAT TEST: PASS');
process.exitCode = failed ? 1 : 0;
