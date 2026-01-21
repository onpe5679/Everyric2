import type { SyncResult, TimestampData } from "./types";

declare const window: Window & {
  require?: NodeRequire;
};

interface NodeRequire {
  (module: string): unknown;
}

function getNodeModule<T>(name: string): T {
  const nodeRequire = window.require;
  if (!nodeRequire) {
    throw new Error("Node.js not available in this environment");
  }
  return nodeRequire(name) as T;
}

interface ChildProcess {
  spawn: (
    cmd: string,
    args: string[],
    opts: { shell?: boolean }
  ) => {
    stdout: { on: (event: string, cb: (data: Buffer) => void) => void };
    stderr: { on: (event: string, cb: (data: Buffer) => void) => void };
    on: (event: string, cb: (code: number) => void) => void;
  };
  exec: (
    cmd: string,
    opts: { timeout?: number },
    cb: (err: Error | null, stdout: string, stderr: string) => void
  ) => void;
}

interface Fs {
  writeFileSync: (path: string, data: string) => void;
  readFileSync: (path: string, encoding: string) => string;
  unlinkSync: (path: string) => void;
  existsSync: (path: string) => boolean;
}

interface Os {
  tmpdir: () => string;
}

interface Path {
  join: (...paths: string[]) => string;
}

export async function checkCliInstalled(cliPath: string = "everyric2"): Promise<{
  installed: boolean;
  version: string | null;
}> {
  try {
    const childProcess = getNodeModule<ChildProcess>("child_process");
    return new Promise((resolve) => {
      childProcess.exec(
        `${cliPath} --version`,
        { timeout: 5000 },
        (err: Error | null, stdout: string) => {
          if (err) {
            resolve({ installed: false, version: null });
          } else {
            resolve({ installed: true, version: stdout.trim() });
          }
        }
      );
    });
  } catch {
    return { installed: false, version: null };
  }
}

export async function runLocalAlignment(
  audioPath: string,
  lyrics: string,
  options: {
    cliPath?: string;
    language?: string;
    translate?: boolean;
    pronunciation?: boolean;
    segmentMode?: "line" | "word" | "character";
    onProgress?: (status: string, percent: number) => void;
  } = {}
): Promise<SyncResult> {
  const childProcess = getNodeModule<ChildProcess>("child_process");
  const fs = getNodeModule<Fs>("fs");
  const os = getNodeModule<Os>("os");
  const path = getNodeModule<Path>("path");

  const cliPath = options.cliPath || "everyric2";
  const tmpDir = os.tmpdir();
  const timestamp = Date.now();
  const lyricsPath = path.join(tmpDir, `everyric_lyrics_${timestamp}.txt`);
  const outputPath = path.join(tmpDir, `everyric_output_${timestamp}.json`);

  fs.writeFileSync(lyricsPath, lyrics);

  const args = [
    "sync",
    `"${audioPath}"`,
    `"${lyricsPath}"`,
    "--output",
    `"${outputPath}"`,
    "--format",
    "json",
    "--engine",
    "ctc",
  ];

  if (options.language && options.language !== "auto") {
    args.push("--language", options.language);
  }
  if (options.translate) {
    args.push("--translate");
  }
  if (options.pronunciation) {
    args.push("--pronunciation");
  }
  if (options.segmentMode && options.segmentMode !== "line") {
    args.push("--segment-mode", options.segmentMode);
  }

  return new Promise((resolve, reject) => {
    options.onProgress?.("Starting alignment...", 10);

    const child = childProcess.spawn(cliPath, args.slice(0).map((a) => a.replace(/"/g, "")), {
      shell: true,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
      if (stdout.includes("Loading audio")) {
        options.onProgress?.("Loading audio...", 20);
      } else if (stdout.includes("Separating")) {
        options.onProgress?.("Separating vocals...", 30);
      } else if (stdout.includes("Synchronizing")) {
        options.onProgress?.("Synchronizing lyrics...", 50);
      } else if (stdout.includes("Post-processing")) {
        options.onProgress?.("Post-processing...", 80);
      }
    });

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    child.on("close", (code: number) => {
      try {
        fs.unlinkSync(lyricsPath);
      } catch {
      }

      if (code !== 0) {
        reject(new Error(`CLI error (code ${code}): ${stderr || stdout}`));
        return;
      }

      if (!fs.existsSync(outputPath)) {
        reject(new Error("Output file not created"));
        return;
      }

      try {
        const content = fs.readFileSync(outputPath, "utf8");
        const rawResult = JSON.parse(content);
        fs.unlinkSync(outputPath);

        const segments: TimestampData[] = (rawResult.lyrics || []).map(
          (item: Record<string, unknown>) => ({
            text: item.text as string,
            start: item.start_time as number,
            end: item.end_time as number,
            translation: item.translation as string | undefined,
            pronunciation: item.pronunciation as string | undefined,
            confidence: item.confidence as number | undefined,
          })
        );

        options.onProgress?.("Complete!", 100);

        resolve({
          segments,
          metadata: rawResult.metadata || {
            duration: 0,
            language: options.language || "auto",
            engine: "ctc",
          },
        });
      } catch (e) {
        reject(new Error(`Failed to parse output: ${e}`));
      }
    });
  });
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}
