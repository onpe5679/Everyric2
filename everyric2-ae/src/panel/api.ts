import type { SyncResult, TimestampData } from "./types";

const DEFAULT_API_URL = "https://api.everyric.com";

interface JobStatusResponse {
  status: "processing" | "completed" | "failed";
  progress?: number;
  result?: {
    segments: Array<{
      text: string;
      start: number;
      end: number;
      translation?: string;
      pronunciation?: string;
    }>;
  };
  error?: string;
}

export async function generateSyncCloud(
  audioFile: File | Blob,
  lyrics: string,
  options: {
    apiUrl?: string;
    apiKey?: string;
    language?: string;
    translate?: boolean;
    pronunciation?: boolean;
    onProgress?: (status: string, percent: number) => void;
  } = {}
): Promise<SyncResult> {
  const apiUrl = options.apiUrl || DEFAULT_API_URL;
  
  options.onProgress?.("Uploading audio...", 10);

  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("lyrics", lyrics);
  formData.append("language", options.language || "auto");
  if (options.translate) formData.append("translate", "true");
  if (options.pronunciation) formData.append("pronunciation", "true");

  const headers: Record<string, string> = {};
  if (options.apiKey) {
    headers["Authorization"] = `Bearer ${options.apiKey}`;
  }

  const response = await fetch(`${apiUrl}/api/sync/generate`, {
    method: "POST",
    body: formData,
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
  }

  const { job_id, status } = await response.json();

  if (status === "completed") {
    return fetchJobResult(apiUrl, job_id, headers);
  }

  options.onProgress?.("Processing...", 30);
  return pollJobStatus(apiUrl, job_id, headers, options.onProgress);
}

async function fetchJobResult(
  apiUrl: string,
  jobId: string,
  headers: Record<string, string>
): Promise<SyncResult> {
  const response = await fetch(`${apiUrl}/api/job/${jobId}`, { headers });
  const data = await response.json();
  
  if (!data.result?.segments) {
    throw new Error("Invalid result format");
  }

  const segments: TimestampData[] = data.result.segments.map(
    (s: { text: string; start: number; end: number; translation?: string; pronunciation?: string }) => ({
      text: s.text,
      start: s.start,
      end: s.end,
      translation: s.translation,
      pronunciation: s.pronunciation,
    })
  );

  return {
    segments,
    metadata: {
      duration: segments.length > 0 ? segments[segments.length - 1].end : 0,
      language: "auto",
      engine: "cloud",
    },
  };
}

async function pollJobStatus(
  apiUrl: string,
  jobId: string,
  headers: Record<string, string>,
  onProgress?: (status: string, percent: number) => void,
  maxAttempts = 60
): Promise<SyncResult> {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 2000));

    const response = await fetch(`${apiUrl}/api/job/${jobId}`, { headers });
    const data: JobStatusResponse = await response.json();

    if (data.status === "completed") {
      onProgress?.("Complete!", 100);
      return fetchJobResult(apiUrl, jobId, headers);
    }

    if (data.status === "failed") {
      throw new Error(data.error || "Processing failed");
    }

    const progress = Math.min(30 + (i / maxAttempts) * 60, 90);
    onProgress?.("Processing...", progress);
  }

  throw new Error("Timeout waiting for job completion");
}

export async function checkApiStatus(apiUrl: string = DEFAULT_API_URL): Promise<boolean> {
  try {
    const response = await fetch(`${apiUrl}/health`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}
