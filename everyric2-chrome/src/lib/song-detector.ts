import type { SongInfo } from '../types';

const TITLE_NOISE: RegExp[] = [
  /[([]\s*official[^)\]]*[)\]]/gi,
  /[([][^)\]]*(?:music|lyric|audio)\s*video[^)\]]*[)\]]/gi,
  /[([]\s*lyrics?[^)\]]*[)\]]/gi,
  /[([]\s*audio\s*[)\]]/gi,
  /[([]\s*mv\s*[)\]]/gi,
  /[([]\s*m\/v\s*[)\]]/gi,
  /[([]\s*(?:4k|hd|hq)[^)\]]*[)\]]/gi,
  /[([]\s*(?:color coded|한글 자막|가사)[^)\]]*[)\]]/gi,
  /【[^】]*】/g,
];

export function getCurrentVideoId(): string | null {
  try {
    const url = new URL(location.href);
    if (url.pathname === '/watch') return url.searchParams.get('v');
    // Shorts·임베드·라이브 경로는 videoId가 경로에 실려 온다
    const m = url.pathname.match(/^\/(?:shorts|embed|live)\/([A-Za-z0-9_-]{11})(?:[/?]|$)/);
    if (m) return m[1];
  } catch {
    /* URL 파싱 실패는 videoId 없음으로 처리 */
  }
  return null;
}

export function getVideoElement(): HTMLVideoElement | null {
  const videos = Array.from(document.querySelectorAll<HTMLVideoElement>('video'));
  if (videos.length === 0) return null;
  // 페이지에 프리뷰/광고 등 여러 video가 있을 수 있으므로 실제 재생 중인 것을 우선하되,
  // 그중에서도 본편 플레이어(html5-main-video)를 먼저, 인라인 프리뷰·광고 슬롯 안의
  // video는 후순위로 — 홈 피드 프리뷰가 재생 중이면 가사가 엉뚱한 영상에 붙는다
  const playing = videos.filter(v => !v.paused && v.readyState >= 2 && v.currentTime > 0);
  const mainPlaying = playing.find(v => v.classList.contains('html5-main-video'));
  if (mainPlaying) return mainPlaying;
  const nonPreview = playing.find(
    v => !v.closest('ytd-video-preview, ytd-ad-slot-renderer, ytd-in-feed-ad-layout-renderer'),
  );
  if (nonPreview) return nonPreview;
  if (playing.length > 0) return playing[0];
  return videos.find(v => v.classList.contains('html5-main-video')) ?? videos[0];
}

function textOf(selector: string): string {
  return document.querySelector(selector)?.textContent?.trim() ?? '';
}

export function cleanTitle(raw: string): string {
  let title = raw;
  for (const re of TITLE_NOISE) title = title.replace(re, ' ');
  return title.replace(/\s{2,}/g, ' ').trim();
}

/**
 * "곡명 / 아티스트 [커버 동사]｜채널" 꼴 — 일본어권 커버 영상 제목 관례(実測: PrXtrTgMDEg
 * 「【韓国語で】네모네모(NEMONEMO) / YENA 歌ってみた｜Kotoha」). 아래 일반 분해(구분자
 * 앞=아티스트, 뒤=곡명)와 필드 순서가 **정반대**(앞=곡명, 뒤=아티스트)라 같은 규칙으로
 * 못 묶는다 — 그래서 별도 함수로 분리한다.
 *
 * 슬래시가 있다고 무조건 발동하면 "A / B"류의 정상 협업 제목까지 곡명·아티스트를 뒤바꿔
 * 망가뜨린다. 그래서 슬래시 뒤 조각에 커버 지시어(歌ってみた·cover·커버 등)가 실제로
 * 있을 때만 발동한다(보수적 게이트) — 없으면 null을 돌려줘 호출부가 기존 규칙이나
 * meta.artist 폴백으로 넘어가게 한다.
 */
function splitCoverTitle(title: string): { title: string; artist: string } | null {
  const slash = title.match(/\s*[/／]\s*/);
  if (!slash || slash.index === undefined || slash.index <= 0) return null;
  const left = title.slice(0, slash.index).trim();
  let right = title.slice(slash.index + slash[0].length).trim();
  if (!left || !right) return null;
  // 전각/반각 파이프 뒤는 채널명이다 — 아티스트 후보에서 제외한다(공백 유무 무관, 실제
  // 사고 제목은 "歌ってみた｜Kotoha"처럼 파이프 앞뒤에 공백이 없다)
  const pipe = right.match(/[｜|]/);
  if (pipe && pipe.index !== undefined) right = right.slice(0, pipe.index).trim();
  const COVER_HINT = /歌ってみた|弾いてみた|踊ってみた|cover|커버/i;
  if (!COVER_HINT.test(right)) return null;
  const artist = right.replace(COVER_HINT, '').trim();
  return artist ? { title: left, artist } : null;
}

/** export는 테스트 전용 — detectSong 내부 호출부는 그대로 상대 import로 쓴다 */
export function splitArtistTitle(title: string): { title: string; artist: string | null } {
  const cover = splitCoverTitle(title);
  if (cover) return cover;
  for (const sep of [' - ', ' – ', ' — ', ' | ']) {
    const idx = title.indexOf(sep);
    if (idx > 0) {
      return { artist: title.slice(0, idx).trim(), title: title.slice(idx + sep.length).trim() };
    }
  }
  return { title, artist: null };
}

export function detectSong(): SongInfo | null {
  const videoId = getCurrentVideoId();
  if (!videoId) return null;

  const rawDuration = getVideoElement()?.duration ?? 0;
  const duration = Number.isFinite(rawDuration) ? Math.round(rawDuration) : 0;

  const meta = navigator.mediaSession?.metadata;
  if (meta?.title) {
    // DOM 폴백 경로(아래)와 같은 분해를 거친다 — 예전엔 이 경로만 cleanTitle만 태우고
    // splitArtistTitle을 안 타서, 제목에 아티스트가 박혀 있어도 못 뽑고 meta.artist(대개
    // 채널명, 실제 가수가 아닐 수 있다 — 커버 영상에서 특히 그렇다)를 그대로 썼다(실사고:
    // PrXtrTgMDEg). 제목에서 뽑히면 그게 우선이고, meta.artist는 못 뽑았을 때만 폴백이다.
    const split = splitArtistTitle(cleanTitle(meta.title));
    return {
      title: split.title,
      artist: split.artist ?? (meta.artist || null),
      videoId,
      duration,
      rawTitle: meta.title,
    };
  }

  if (location.host === 'music.youtube.com') {
    const title = textOf('ytmusic-player-bar .title');
    if (title) {
      const byline = textOf('ytmusic-player-bar .byline');
      const artist = byline.split('•')[0]?.trim() || null;
      return { title: cleanTitle(title), artist, videoId, duration, rawTitle: title };
    }
  }

  const rawTitle = textOf('h1.ytd-watch-metadata yt-formatted-string')
    || textOf('#title h1')
    || document.title.replace(/ - YouTube$/, '').trim();
  if (!rawTitle || rawTitle === 'YouTube') return null;

  const channel = textOf('#owner #channel-name a').replace(/ - Topic$/i, '').trim() || null;
  const split = splitArtistTitle(cleanTitle(rawTitle));
  return {
    title: split.title,
    artist: split.artist ?? channel,
    videoId,
    duration,
    rawTitle,
  };
}
