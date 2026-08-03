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
 * 라틴 문자 우세 판정 — 영어 곡 구제(오염된 구세대 romaji 표기 교체)와 가라오케 노트
 * 원문 부착에 쓴다. 라틴 문자 수가 가나+한자+한글 합보다 많으면 참.
 */
export function isLatinDominant(text: string): boolean {
  const latin = (text.match(/[A-Za-z]/g) ?? []).length;
  const cjk = (text.match(/[぀-ヿ㐀-鿿가-힣]/g) ?? []).length;
  return latin > cjk;
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
 *
 * **영어 곡 구제**: script가 'romaji'이고 원문이 라틴 우세인 줄은 romaji 대신 line.pron['en']을
 * 쓴다 — 구세대 서버가 영어 가사를 "영어→가타카나→로마자"로 오염시켜 저장한 곡들이
 * 이미 많이 쌓여 있다("za wezaa..."류). 서버는 이제 새로 만드는 곡에 romaji=en(원문
 * 철자)을 넣지만, 이미 저장된 곡은 클라이언트가 이렇게 구제한다.
 *
 * **ipa 폴백**: ipa는 서버 en 정렬 스택의 부산물이라(derive_en_display_units가 en 곡에만
 * owners['ipa']를 얹는다) ja·ko 곡에는 애초에 존재하지 않는다. script가 'ipa'인데 이
 * 줄에 값이 없으면 IPA_FALLBACK_ORDER(hangul→romaji→kana) 순으로 대신 찾는다 — hangul을
 * 최우선으로 두는 이유는 attach_pron_variants가 ja·ko 곡에도 hangul은 항상 채우므로
 * (가나 근사·결정론 RR) 이 순서가 "발음 줄이 통째로 빈다"를 가장 넓게 막기 때문이다.
 */
const IPA_FALLBACK_ORDER: readonly Exclude<PronScript, 'ipa'>[] = ['hangul', 'romaji', 'kana'];

export function resolvedPronunciation(line: LyricLine, script: PronScript): string | undefined {
  if (script === 'romaji' && line.pron?.['en'] && isLatinDominant(line.text)) return line.pron['en'];
  if (script === 'ipa' && !line.pron?.['ipa']) {
    for (const fallback of IPA_FALLBACK_ORDER) {
      const v = resolvedPronunciation(line, fallback);
      if (v) return v;
    }
    return undefined;
  }
  return line.pron?.[script] ?? (script === 'hangul' ? line.pronunciation : undefined);
}

/** 표시용 발음 음절 타이밍 — 규칙은 resolvedPronunciation과 동일(표기별 값 → hangul만 레거시
 *  폴백, 라틴 우세 romaji 줄은 'en' 세그로 구제, ipa 부재 줄은 IPA_FALLBACK_ORDER로 대체).
 *  worker.py의 attach_pron_variants는 ipa 문자열만 얹고 pron_segs(모라 타이밍)는 CTC가
 *  라틴 위에서 신뢰할 수 없어 붙이지 않는다 — 그래서 ipa 세그는 사실상 항상 이 폴백을 탄다. */
export function resolvedPronSegments(line: LyricLine, script: PronScript): PronSegment[] | undefined {
  if (script === 'romaji' && line.pronSegsByScript?.['en'] && isLatinDominant(line.text)) {
    return line.pronSegsByScript['en'];
  }
  if (script === 'ipa' && !line.pronSegsByScript?.['ipa']) {
    for (const fallback of IPA_FALLBACK_ORDER) {
      const v = resolvedPronSegments(line, fallback);
      if (v) return v;
    }
    return undefined;
  }
  return line.pronSegsByScript?.[script] ?? (script === 'hangul' ? line.pronSegments : undefined);
}

/** 비교용 정규화 — shouldShowPron의 "원문과 발음이 사실상 같은 문자열인가" 판정에 쓴다.
 *  NFKC로 정준 결합(전각·호환 문자 통일) 후 소문자화, 그다음 공백(\s)과 유니코드 구두점
 *  (\p{P} — 쉼표·마침표·따옴표·대시 등)을 전부 제거한다. 공백만 접던 이전 버전은
 *  "morning, just"(원문)와 "morning ,just"(발음)처럼 구두점 위치·공백 유무만 다른 줄을
 *  "다른 텍스트"로 오판해 en 곡×en 사용자의 원문이 발음 줄로 한 번 더 뜨는 결함이 있었다
 *  (실사용 제보). 진짜 로마자 발음(가나 곡의 romaji 등)은 이 정규화를 거쳐도 원문과 계속
 *  다르므로 계속 표시된다.
 *  shouldShowPron이 이 판정의 유일한 지점이라 메인 패널·PIP·자막(video-caption.ts)이
 *  전부 이 규칙을 공유한다 — 사용처마다 따로 정규화하지 않는다. */
function normalizeForPronCompare(s: string): string {
  return s.normalize('NFKC').toLowerCase().replace(/[\s\p{P}]+/gu, '');
}

/**
 * 발음 줄을 이 줄에서 보여줄지 — 전체 끔(showPronunciation)과 영어만 끔
 * (hidePronForEnglish)을 한 판정으로 합친다. 레인(pitch-lane.ts)의 노트 부착 발음은
 * 이 함수를 거치지 않는다 — PIP·레인 노트의 발음 표시는 운영자 제약으로 항상 유지된다.
 *
 * resolvedPron을 함께 주면(감사 C4) 원문과 정규화 비교(공백·대소문자 무시)로 사실상
 * 같을 때 발음 줄 자체를 숨긴다 — en 곡×en 사용자는 romaji가 원문 철자 그대로라서,
 * 이 게이트가 없으면 원문 줄 바로 아래 같은 글자가 한 번 더 뜬다.
 */
export function shouldShowPron(
  lineText: string,
  settings: Pick<Settings, 'showPronunciation' | 'hidePronForEnglish'>,
  resolvedPron?: string,
): boolean {
  if (!settings.showPronunciation) return false;
  if (settings.hidePronForEnglish && isLatinDominant(lineText)) return false;
  if (resolvedPron !== undefined && normalizeForPronCompare(resolvedPron) === normalizeForPronCompare(lineText)) {
    return false;
  }
  return true;
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
