import { spawn } from "child_process";

import { isNewerVersion, PANEL_VERSION } from "./version";

export interface ComponentRelease {
  version: string;
  zxpUrl?: string;
  wheelUrl?: string;
  releaseUrl?: string;
  engineRange?: string;
}

export interface LatestManifest {
  ae?: ComponentRelease;
  engine?: ComponentRelease;
  chrome?: ComponentRelease;
}

const LATEST_MANIFEST_URL = "https://raw.githubusercontent.com/onpe5679/Everyric2/master/latest.json";
export const RELEASES_URL = "https://github.com/onpe5679/Everyric2/releases";
const CACHE_KEY = "everyric_update_manifest_v1";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

interface CachedManifest {
  fetchedAt: number;
  manifest: LatestManifest;
}

function readCache(): CachedManifest | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedManifest;
    if (typeof parsed.fetchedAt !== "number" || typeof parsed.manifest !== "object" || parsed.manifest === null) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function fetchLatestManifest(force = false): Promise<LatestManifest | null> {
  const cached = readCache();
  if (!force && cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) return cached.manifest;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 10_000);
  try {
    const response = await fetch(LATEST_MANIFEST_URL, { cache: "no-store", signal: controller.signal });
    if (!response.ok) throw new Error(`latest.json HTTP ${response.status}`);
    const manifest = (await response.json()) as LatestManifest;
    localStorage.setItem(CACHE_KEY, JSON.stringify({ fetchedAt: Date.now(), manifest } satisfies CachedManifest));
    return manifest;
  } catch {
    // 오프라인이거나 매니페스트가 아직 없는 경우: 캐시가 있으면 그대로, 없으면 조용히 포기한다.
    return cached?.manifest ?? null;
  } finally {
    window.clearTimeout(timer);
  }
}

export function panelUpdate(manifest: LatestManifest | null): ComponentRelease | null {
  const ae = manifest?.ae;
  if (ae?.version && isNewerVersion(ae.version, PANEL_VERSION)) return ae;
  return null;
}

export function openExternal(url: string): void {
  if (!/^https:\/\//.test(url)) return;
  const cepUtil = (window as { cep?: { util?: { openURLInDefaultBrowser?: (target: string) => void } } }).cep?.util;
  if (cepUtil?.openURLInDefaultBrowser) {
    cepUtil.openURLInDefaultBrowser(url);
    return;
  }
  spawn("explorer.exe", [url], { windowsHide: true, stdio: "ignore", detached: true }).unref();
}
