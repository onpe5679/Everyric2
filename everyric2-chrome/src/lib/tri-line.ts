/**
 * 검색 결과 가사처럼 「원문 / (발음) / 번역」이 주기적으로 반복되는 붙여넣기 텍스트를
 * 감지해 분해한다. 지원 형태:
 *   - 원문/번역 2줄 교차 (신규)
 *   - 원문/발음/번역 3줄 교차 (기존)
 *   - 위 두 형태 모두, 빈 줄로 구분된 그룹들 또는 빈 줄 없이 이어지는 주기 반복 둘 다
 * 곡 전체에서 한 가지 주기(2 또는 3)만 쓰인다고 가정한다 — 그룹 크기가 섞인 곡은
 * 다루지 않는다. 확신이 없으면 null — 호출부는 원문 그대로 생성 경로를 탄다.
 *
 * **블록을 나누기 전에 파트 표기·주석 줄을 걷어낸다** (lyrics-clean). 표기 한 줄이 섞이면
 * 주기가 밀려 그룹이 깨지거나 (원문,번역,원문) 꼴로 잘못 잘려 감지 자체가 실패했다.
 *
 * ## 그룹 나누기: 빈 줄 구조는 왜 별도로 안 다루나
 * 빈 줄로 나뉜 블록들이 있어도, "블록 길이가 N의 배수"라는 조건만 성립하면 그 블록
 * 경계는 항상 flat 배열을 N개씩 순서대로 끊었을 때의 경계와 정확히 일치한다 (블록
 * 길이의 누적합이 N의 배수의 합이라 항상 N의 배수 지점에서 끊기기 때문). 즉 블록별로
 * 따로 자르나 flat을 통째로 N개씩 자르나 결과가 같다 — 그래서 별도의 블록 인식 단계
 * 없이 flat 배열을 N개씩 자르고, 문자 계열 일치율로 안전성을 검증하는 것만으로
 * 두 구조(빈 줄 구분 / 빈 줄 없이 이어짐) 모두를 같은 코드로 커버한다.
 * 블록 길이가 N의 배수가 아니게 섞인 경우(그룹 하나가 쪼개짐)는 아래 일치율 검증이
 * 걸러낸다 — 어긋난 그룹은 문자 계열이 안 맞아 ok 판정을 못 받고, 전체 일치율이
 * 0.8 미만이면 그 N은 통째로 기각된다.
 */

import { stripPartMarkers } from './lyrics-clean';

export interface TriLine {
  text: string;
  pronunciation?: string;
  translation?: string;
}

const RE_KANA = /[぀-ヿ]/;
const RE_CJK = /[一-鿿]/;
const RE_HANGUL = /[가-힣]/g;
const RE_LATIN = /[A-Za-z]/;

function hangulRatio(s: string): number {
  const total = s.replace(/\s/g, '').length;
  if (total === 0) return 0;
  return (s.match(RE_HANGUL)?.length ?? 0) / total;
}

/** 일어 원문 줄: 가나나 한자를 포함하고 한글은 거의 없음 */
function isJa(s: string): boolean {
  return (RE_KANA.test(s) || RE_CJK.test(s)) && hangulRatio(s) < 0.15;
}

/** 라틴 문자 원문 줄(영어 가사 등): 알파벳을 포함하고 가나·한자·한글은 거의 없음 */
function isLatin(s: string): boolean {
  return RE_LATIN.test(s) && !RE_KANA.test(s) && !RE_CJK.test(s) && hangulRatio(s) < 0.15;
}

/** 그룹의 첫 줄(원문) 판정 — 일어 또는 라틴 */
function isSrc(s: string): boolean {
  return isJa(s) || isLatin(s);
}

/** 한글 줄(발음·번역): 한글이 절반 이상이고 가나 없음 */
function isKo(s: string): boolean {
  return hangulRatio(s) >= 0.5 && !RE_KANA.test(s);
}

/**
 * flat 라인 배열을 n줄씩 순서대로 묶고, 각 그룹이 (원문, 한글, 한글…) 꼴인지로
 * 일치율을 매긴다. 최소 두 그룹은 있어야 주기로 인정한다.
 */
function groupByN(flat: string[], n: number): { groups: string[][]; ratio: number } | null {
  if (flat.length < n * 2 || flat.length % n !== 0) return null;
  const groups: string[][] = [];
  for (let i = 0; i < flat.length; i += n) groups.push(flat.slice(i, i + n));
  let ok = 0;
  for (const [first, ...rest] of groups) {
    if (isSrc(first) && rest.every(isKo)) ok++;
  }
  return { groups, ratio: ok / groups.length };
}

const MATCH_THRESHOLD = 0.8;

export function parseTriLineLyrics(raw: string): TriLine[] | null {
  // 호출부(content.handleGenerate)가 이미 걸러낸 텍스트를 주더라도 판정은 멱등하다 —
  // 표기를 지운 결과에는 새 표기가 생기지 않으므로 두 번 통과해도 같은 텍스트다
  const flat = stripPartMarkers(raw).text.split('\n').map(s => s.trim()).filter(Boolean);

  // n=3(원문/발음/번역)을 먼저 시도한다 — 더 구체적인 패턴이라 정보 손실이 없다.
  // 맞지 않으면 n=2(원문/번역)로 폴백. 둘 다 독립적으로 0.8 문턱을 넘어야 채택된다 —
  // 우연히 다른 주기로도 얼마간 맞아떨어지는 것만으로는 오판을 허용하지 않는다.
  for (const n of [3, 2] as const) {
    const scored = groupByN(flat, n);
    if (!scored || scored.ratio < MATCH_THRESHOLD) continue;
    return scored.groups.map(g => (
      n === 3
        ? { text: g[0], pronunciation: g[1], translation: g[2] }
        : { text: g[0], translation: g[1] }
    ));
  }
  return null;
}
