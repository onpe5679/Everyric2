import type { LrcLibResult } from "./types";

const LRCLIB_API = "https://lrclib.net/api";

export async function searchLrcLib(query: string): Promise<LrcLibResult[]> {
  const response = await fetch(`${LRCLIB_API}/search?q=${encodeURIComponent(query)}`);
  
  if (!response.ok) {
    throw new Error(`LRCLIB search failed: ${response.status}`);
  }

  const results = await response.json();
  return results.map((item: Record<string, unknown>) => ({
    id: item.id as number,
    name: item.name as string,
    trackName: item.trackName as string,
    artistName: item.artistName as string,
    albumName: item.albumName as string | undefined,
    duration: item.duration as number,
    syncedLyrics: item.syncedLyrics as string | undefined,
    plainLyrics: item.plainLyrics as string | undefined,
  }));
}

export async function getLrcById(id: number): Promise<LrcLibResult | null> {
  const response = await fetch(`${LRCLIB_API}/get/${id}`);
  
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`LRCLIB get failed: ${response.status}`);
  }

  const item = await response.json();
  return {
    id: item.id,
    name: item.name,
    trackName: item.trackName,
    artistName: item.artistName,
    albumName: item.albumName,
    duration: item.duration,
    syncedLyrics: item.syncedLyrics,
    plainLyrics: item.plainLyrics,
  };
}

export function parseLrcToPlainText(lrc: string): string {
  return lrc
    .split("\n")
    .map((line) => line.replace(/^\[\d{2}:\d{2}.\d{2,3}\]/, "").trim())
    .filter((line) => line.length > 0 && !line.startsWith("["))
    .join("\n");
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
