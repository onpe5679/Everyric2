/**
 * 하네스 기본 실곡 해결자 — **하드코딩된 기본값이 죽었을 때만** DB에서 대체를 고른다.
 *
 * 왜 필요한가(2026-08-04 실측): 여러 하네스가 `Xg-qfsKN2_E`를 기본 영상으로 박아 두었는데
 * 로컬 서버에는 그 싱크가 없다(프로드에는 45줄로 살아 있다). 인자 없이 돌리면 «가사를 찾지
 * 못했어요» 화면에서 전부 실패하는데, 그건 제품 결함이 아니라 **죽은 기본값**이다.
 *
 * 설계 원칙 두 가지:
 *  1) **선호값을 먼저 존중한다.** 그 곡이 DB에 있으면 그대로 쓴다 — 하네스의 기존 판정이
 *     그 곡의 성질(간주 길이 등)에 기대고 있을 수 있어, 멀쩡한데 굳이 바꾸지 않는다.
 *  2) 없을 때만 조건에 맞는 곡을 고르고, **무엇을 왜 골랐는지 찍는다**(조용한 대체는
 *     나중에 "왜 다른 곡이 나왔지?"로 돌아온다).
 *
 * 조회만 한다 — 생성·수정은 없다(fresh-profile-check.mjs / feedback-round3-check.mjs의 DB 조회 패턴).
 */
import { DatabaseSync } from 'node:sqlite';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const DB_PATH = resolve(dirname(fileURLToPath(import.meta.url)), '../../../everyric2.db');

/** 한 싱크 행이 조건을 만족하는가 — 선호값과 대체 후보에 **같은 잣대**를 댄다 */
function qualifies(parsed, opts) {
  const { minLines = 20, minFirstStart = 0, needTempo = false, routeIn = null } = opts;
  const segs = parsed?.segments ?? [];
  if (segs.length < minLines) return false;
  if (needTempo && !parsed.tempo) return false;
  if (minFirstStart > 0 && !(Number(segs[0]?.start) >= minFirstStart)) return false;
  if (routeIn) {
    // 분석 깊이는 싱크 메타에 있다(서버 라우팅 판정). 구싱크는 이 메타가 없어 null이고,
    // 그때는 조건을 만족하지 못한 것으로 본다 — «모르는 값»을 통과시키면 조건이 무의미해진다.
    const route = parsed?.debug?.routing?.route ?? parsed?.routing?.route ?? null;
    if (!routeIn.includes(route)) return false;
  }
  return true;
}

/**
 * @param {string} preferred 하네스가 원래 쓰던 videoId (**조건까지 만족하면** 그대로 쓴다)
 * @param {object} [opts]
 * @param {number} [opts.minLines=20]      최소 줄 수
 * @param {number} [opts.minFirstStart=0]  첫 줄 시작 시각 하한(초) — 카운트다운처럼 «긴 도입»이 필요한 검사용
 * @param {boolean} [opts.needTempo=false] tempo(BPM)가 있어야 하는가 — 레인 마디선 계열
 * @param {string[]} [opts.routeIn]        분석 깊이 화이트리스트(예: ['fast','medium'])
 * @returns {{ videoId: string, source: 'preferred'|'db'|'fallback', title?: string, lines?: number, route?: string|null, note: string }}
 */
export function resolveVideoId(preferred, opts = {}) {
  if (!existsSync(DB_PATH)) {
    return { videoId: preferred, source: 'fallback', note: `DB 없음(${DB_PATH}) — 선호값 그대로 사용` };
  }
  const db = new DatabaseSync(DB_PATH, { readOnly: true });
  const routeOf = (d) => d?.debug?.routing?.route ?? d?.routing?.route ?? null;
  try {
    const rows = db.prepare(
      'SELECT video_id, title, timestamps FROM sync_results ORDER BY id DESC',
    ).all();
    const parse = (r) => { try { return JSON.parse(r.timestamps); } catch { return null; } };

    if (preferred) {
      // **존재만으로는 부족하다** — 조건까지 봐야 한다. 예: U4(타이밍 안내 배너)는 fast/medium
      // 싱크에서만 뜨는데, 기본 곡이 heavy면 «DB에 있으니 그대로»로 통과시키는 순간 그 검사가
      // 원리적으로 성립하지 못한 채 실패한다(2026-08-04 실측으로 잡힌 U4 노후).
      const hit = rows.find(r => r.video_id === preferred);
      if (hit) {
        const d = parse(hit);
        if (qualifies(d, opts)) {
          return {
            videoId: preferred, source: 'preferred', title: hit.title, route: routeOf(d),
            note: '기본 영상이 조건을 만족해 그대로 사용',
          };
        }
      }
    }
    for (const r of rows) {
      const d = parse(r);
      if (!d || !qualifies(d, opts)) continue;
      return {
        videoId: r.video_id, source: 'db', title: r.title, lines: (d.segments ?? []).length,
        route: routeOf(d),
        note: `기본 영상(${preferred})이 조건에 안 맞아 대체를 골랐다`,
      };
    }
    return { videoId: preferred, source: 'fallback', note: '조건에 맞는 대체 곡이 DB에 없어 선호값 유지' };
  } finally {
    db.close();
  }
}

/** watch URL로 돌려주는 편의판 — 인자를 URL로 받는 하네스가 많다 */
export function resolveVideoUrl(preferredUrlOrId, opts = {}) {
  const id = /^https?:/.test(preferredUrlOrId)
    ? (new URL(preferredUrlOrId).searchParams.get('v') ?? preferredUrlOrId)
    : preferredUrlOrId;
  const picked = resolveVideoId(id, opts);
  return { ...picked, url: `https://www.youtube.com/watch?v=${picked.videoId}` };
}
