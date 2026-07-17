import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

import type { LocalSyncOptions } from "./types";
import type { EnvironmentReport } from "./types";

export function readJsonFile(filePath: string): unknown {
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) throw new Error("선택한 경로가 파일이 아닙니다.");
  if (stat.size > 50 * 1024 * 1024) throw new Error("JSON 파일이 50MB 제한을 초과합니다.");
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
}

export function checkLocalEngine(pythonPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonPath || "python",
      ["-c", "import everyric2; print(getattr(everyric2, '__version__', 'available'))"],
      { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString("utf8")));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString("utf8")));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve(stdout.trim() || "available");
      else reject(new Error(stderr.trim() || `Python 종료 코드 ${code}`));
    });
  });
}

function runProbe(command: string, args: string[]): Promise<string> {
  return new Promise((resolve) => {
    const child = spawn(command, args, { windowsHide: true, stdio: ["ignore", "pipe", "ignore"] });
    let stdout = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString("utf8")));
    child.on("error", () => resolve(""));
    child.on("close", (code) => resolve(code === 0 ? stdout.trim() : ""));
  });
}

function parseNvidiaSmi(csv: string): Partial<EnvironmentReport> {
  const first = csv.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)[0];
  if (!first) return {};
  const [gpuName, total, free, cudaVersion] = first.split(",").map((part) => part.trim());
  return {
    gpuName,
    vramTotalMb: Number(total) || undefined,
    vramFreeMb: Number(free) || undefined,
    cudaVersion,
  };
}

export async function inspectEnvironment(pythonPath: string): Promise<EnvironmentReport> {
  const [everyricVersion, gpuCsv] = await Promise.all([
    checkLocalEngine(pythonPath),
    runProbe("nvidia-smi", [
      "--query-gpu=name,memory.total,memory.free,driver_version",
      "--format=csv,noheader,nounits",
    ]),
  ]);
  const gpu = parseNvidiaSmi(gpuCsv);
  const totalMemoryGb = Math.round((os.totalmem() / 1024 / 1024 / 1024) * 10) / 10;
  const notes: string[] = [];
  if (!gpu.gpuName) notes.push("NVIDIA GPU/VRAM 정보는 nvidia-smi로 확인되지 않았습니다.");
  if (gpu.vramTotalMb && gpu.vramTotalMb < 6 * 1024) notes.push("VRAM 6GB 미만에서는 GPU 계열 엔진보다 CTC를 권장합니다.");
  if (totalMemoryGb < 16) notes.push("시스템 RAM 16GB 미만에서는 긴 곡 처리 시 다른 앱을 닫는 편이 안전합니다.");

  return {
    everyricVersion,
    nodeVersion: process.version,
    platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    cpu: os.cpus()[0]?.model ?? "Unknown CPU",
    systemMemoryGb: totalMemoryGb,
    ...gpu,
    recommended: {
      minimumVramGb: 6,
      comfortableVramGb: 8,
      systemMemoryGb: 16,
    },
    notes,
  };
}

export function runLocalSync(
  options: LocalSyncOptions,
  onProgress: (message: string) => void,
  signal?: AbortSignal,
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "everyric-ae-"));
    const lyricsPath = path.join(tempDir, "lyrics.txt");
    const outputPath = path.join(tempDir, "alignment.json");
    fs.writeFileSync(lyricsPath, options.lyrics, "utf8");

    const args = [
      "-m",
      "everyric2.cli",
      "sync",
      options.audioPath,
      lyricsPath,
      "--output",
      outputPath,
      "--format",
      "json",
      "--engine",
      options.engine,
      "--language",
      options.language,
      "--segment-mode",
      "line",
    ];
    const child = spawn(options.pythonPath || "python", args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUTF8: "1", NO_COLOR: "1" },
    });
    let stdout = "";
    let stderr = "";
    let settled = false;

    const cleanup = (): void => {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // A failed cleanup must not hide a successful alignment result.
      }
    };
    const abort = (): void => {
      if (!settled) child.kill();
    };
    signal?.addEventListener("abort", abort, { once: true });

    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      const lastLine = text.trim().split(/\r?\n/).filter(Boolean).pop();
      if (lastLine) onProgress(lastLine.replace(/\x1b\[[0-9;]*m/g, "").slice(0, 140));
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      settled = true;
      signal?.removeEventListener("abort", abort);
      cleanup();
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", abort);
      if (signal?.aborted) {
        cleanup();
        reject(new Error("동기화 작업을 취소했습니다."));
        return;
      }
      try {
        if (code !== 0) {
          throw new Error((stderr || stdout).trim().slice(-1600) || `Everyric2 종료 코드 ${code}`);
        }
        if (!fs.existsSync(outputPath)) throw new Error("Everyric2가 결과 JSON을 생성하지 않았습니다.");
        const payload = readJsonFile(outputPath);
        cleanup();
        resolve(payload);
      } catch (error) {
        cleanup();
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  });
}
