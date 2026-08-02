import type { LyricLine, PronSegment, Settings } from '../types';

export type PronScript = 'hangul' | 'romaji' | 'kana' | 'ipa';

/**
 * 발음 표기 방식 해석 — 'auto'면 번역 언어 기준 자동 결정표를 따른다
 * (공유 계약: ko→hangul, en→romaji, ja→kana, zh→hangul — zh는 아직 전용 표기가 없어
 * hangul로 폴백한다).
 */
export function resolveScript(
  settings: Pick<Settings, 'pronunciationScript' | 'translationLanguage'>,
): PronScript {
  if (settings.pronunciationScript !== 'auto') return settings.pronunciationScript;
  switch (settings.translationLanguage) {
    case 'en': return 'romaji';
    case 'ja': return 'kana';
    default: return 'hangul'; // ko·zh·그 외
  }
}

/**
 * 표시용 발음 문자열 — line.pron[script]가 있으면 그 값, 없으면 레거시 pronunciation(한글)
 * 으로 폴백한다. 표시 지점은 항상 이 함수를 거쳐야 한다(직접 line.pronunciation을 읽지
 * 않는다) — 서버가 아직 pron dict를 안 주는 동안에도 이 폴백 덕분에 오늘과 동일하게 동작한다.
 *
 * 레거시 폴백은 **script === 'hangul'일 때만** 쓴다 — line.pronunciation은 항상 한글 값이라,
 * romaji·kana를 보는 사용자(en·ja 유저)에게 그대로 돌려주면 자기 표기가 아닌 한글 독음이
 * 뜬다(ja 유저 감사에서 실측 — pron dict에 kana 키가 없는 곡마다 한글이 새어나왔다).
 * hangul 외 표기에서 dict에 값이 없으면 undefined를 그대로 돌려줘 발음 줄 자체를 생략한다.
 */
export function resolvedPronunciation(line: LyricLine, script: PronScript): string | undefined {
  return line.pron?.[script] ?? (script === 'hangul' ? line.pronunciation : undefined);
}

/** 표시용 발음 음절 타이밍 — 규칙은 resolvedPronunciation과 동일(표기별 값 → hangul만 레거시 폴백) */
export function resolvedPronSegments(line: LyricLine, script: PronScript): PronSegment[] | undefined {
  return line.pronSegsByScript?.[script] ?? (script === 'hangul' ? line.pronSegments : undefined);
}

export interface WikiMatchLine {
  text: string;
  translation?: string;
}

/**
 * 위키 페이지 줄과 싱크 세그를 순서 보존 결합 매칭으로 대응시킨다 — 감사 V1.
 *
 * 위키와 싱크는 같은 가사를 **서로 다른 기준**으로 줄 나눈다(실측 H7PR: 미라헤즈 기반
 * 싱크 36줄 vs vocaro 페이지 42줄). 줄 대 줄 완전 일치만 보면 분할이 갈리는 구간마다
 * 전부 미스가 나 매칭률이 실측 50% 미달로 떨어져, 위키 번역이 실제로 있는데도
 * tryWikiTranslationLayer가 포기하고 LLM으로 추락했다.
 *
 * 두 포인터(i=세그, j=위키)로 동시에 훑으며 우선순위대로 시도한다:
 * 1. 정규화 후 완전 일치 — 그대로 대응.
 * 2. **쪼갬**: 세그 하나가 위키 연속 줄 여러 개를 이어 붙인 것과 일치(위키가 더 잘게
 *    나뉜 구간) — 그 위키 줄들의 번역을 이어 붙여 그 세그 하나에 싣는다.
 * 3. **합침**: 위키 줄 하나가 세그 연속 여러 개를 이어 붙인 것과 일치(위키가 더 굵게
 *    나뉜 구간) — 그 위키 줄의 번역을 대응하는 세그 전부에 싣는다(더 잘게 쪼갤 방법이
 *    없으므로 같은 번역을 반복해서라도 각 세그가 빈 줄로 남지 않게 한다).
 * 4. 셋 다 실패하면 **재동기화** — 가까운(맨해튼 거리 오름차순) 다음 완전 일치 지점을
 *    찾아 그 사이는 미매칭으로 남기고 건너뛴다. 그마저 못 찾으면 양쪽을 한 칸씩만 밀고
 *    계속 스캔한다(국소적 어긋남 하나 때문에 그 뒤 전체를 포기하지 않는다).
 *
 * 이어 붙임 비교는 공백을 전부 제거하고 한다(dense) — 원문 줄바꿈 경계에 공백이
 * 있었는지 없었는지는 위키·싱크 어느 쪽도 보장하지 않으므로, 그 유무로 비교가 갈리면
 * 안 된다. 반환값은 항상 seg 개수와 같은 길이 — persistTranslationLayer가 이 배열을
 * 그대로 세그 기준 lines로 서버에 보낸다(위키 줄 기준이 아니라).
 */
export function matchWikiLinesToSegments(
  segs: { text: string }[], wiki: WikiMatchLine[],
): (string | undefined)[] {
  const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
  const dense = (s: string) => s.replace(/\s+/g, '');
  const segNorm = segs.map(s => norm(s.text));
  const wikiNorm = wiki.map(w => norm(w.text));
  const MAX_JOIN = 4; // 한 번에 이어 붙여 볼 최대 줄 수 — 정상적인 분할 차이는 보통 2~3줄 이내
  const ANCHOR_WINDOW = 6; // 재동기화 탐색 범위(맨해튼 거리)

  const translations: (string | undefined)[] = new Array(segs.length).fill(undefined);
  let i = 0; // segs 포인터
  let j = 0; // wiki 포인터

  while (i < segs.length && j < wiki.length) {
    if (segNorm[i] === wikiNorm[j]) {
      translations[i] = wiki[j].translation;
      i++; j++;
      continue;
    }

    // 쪼갬: segNorm[i] === wikiNorm[j..j+k] 이어 붙인 것(공백 무시 비교)
    const segDense = dense(segNorm[i]);
    let splitFound = false;
    let concat = dense(wikiNorm[j]);
    for (let k = 1; k <= MAX_JOIN && j + k < wiki.length; k++) {
      concat += dense(wikiNorm[j + k]);
      if (concat === segDense) {
        translations[i] = wiki.slice(j, j + k + 1).map(w => w.translation).filter(Boolean).join(' ');
        i++; j += k + 1;
        splitFound = true;
        break;
      }
      if (concat.length > segDense.length) break; // 이미 넘어섰다 — 더 이어 붙여도 못 맞는다
    }
    if (splitFound) continue;

    // 합침: wikiNorm[j] === segNorm[i..i+k] 이어 붙인 것(공백 무시 비교)
    const wikiDense = dense(wikiNorm[j]);
    let mergeFound = false;
    let concatSeg = dense(segNorm[i]);
    for (let k = 1; k <= MAX_JOIN && i + k < segs.length; k++) {
      concatSeg += dense(segNorm[i + k]);
      if (concatSeg === wikiDense) {
        for (let m = i; m <= i + k; m++) translations[m] = wiki[j].translation;
        i += k + 1; j++;
        mergeFound = true;
        break;
      }
      if (concatSeg.length > wikiDense.length) break;
    }
    if (mergeFound) continue;

    // 재동기화 — 가까운 다음 완전 일치 지점(앵커)을 찾아 그 사이를 건너뛴다
    let anchor: [number, number] | null = null;
    for (let d = 1; d <= ANCHOR_WINDOW && !anchor; d++) {
      for (let di = 0; di <= d; di++) {
        const dj = d - di;
        const ii = i + di;
        const jj = j + dj;
        if (ii < segs.length && jj < wiki.length && segNorm[ii] === wikiNorm[jj]) {
          anchor = [ii, jj];
          break;
        }
      }
    }
    if (anchor) {
      [i, j] = anchor;
    } else {
      i++; j++; // 앵커를 못 찾았다 — 이 쌍은 포기하고 계속 스캔
    }
  }

  return translations;
}
