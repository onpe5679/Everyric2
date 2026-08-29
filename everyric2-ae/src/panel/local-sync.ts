import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

import type { EnvironmentReport, LocalSyncOptions } from "./types";

const BRIDGE_PROTOCOL = "everyric-local-bridge/v1";
const RESULT_SCHEMA = "sync-document/v3";

interface BridgeEvent {
  event: string;
  message?: string;
  stage?: string;
  depth?: string;
  document?: { schema: string; result: unknown };
  ready?: boolean;
  engineVersion?: string;
  platform?: string;
  gpu?: { name?: string; vramTotalMb?: number; vramFreeMb?: number; driver?: string } | null;
  cuda?: { runtime?: string };
  problems?: string[];
}

export function readJsonFile(filePath: string): unknown {
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) throw new Error("선택한 경로가 파일이 아닙니다.");
  if (stat.size > 50 * 1024 * 1024) throw new Error("JSON 파일은 50MB를 초과할 수 없습니다.");
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as unknown;
}

function bridge(
  pythonPath: string,
  args: string[],
  onEvent: (event: BridgeEvent) => void,
  signal?: AbortSignal,
): Promise<BridgeEvent[]> {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonPath || "python", ["-m", "everyric2.local_bridge", ...args], {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUTF8: "1", NO_COLOR: "1" },
    });
    const events: BridgeEvent[] = [];
    let pending = "";
    let stderr = "";
    let settled = false;
    const parseLine = (line: string): void => {
      if (!line.trim()) return;
      try {
        const event = JSON.parse(line) as BridgeEvent;
        events.push(event);
        onEvent(event);
      } catch {
        stderr = (stderr + "\n" + line).slice(-4000);
      }
    };
    child.stdout.on("data", (chunk: Buffer) => {
      pending += chunk.toString("utf8");
      const lines = pending.split(/\r?\n/);
      pending = lines.pop() ?? "";
      lines.forEach(parseLine);
    });
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString("utf8")));
    const abort = (): void => {
      child.kill();
    };
    signal?.addEventListener("abort", abort, { once: true });
    child.on("error", (error) => {
      settled = true;
      signal?.removeEventListener("abort", abort);
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) return;
      signal?.removeEventListener("abort", abort);
      parseLine(pending);
      if (signal?.aborted) reject(new Error("동기화 작업을 취소했습니다."));
      else if (code !== 0) {
        const reported = [...events].reverse().find((event) => event.event === "error")?.message;
        reject(new Error(reported || stderr.trim().slice(-1600) || `로컬 브리지 종료 코드 ${code}`));
      } else resolve(events);
    });
  });
}

function writeRequest(payload: object): { directory: string; file: string } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "everyric-bridge-"));
  const file = path.join(directory, "request.json");
  fs.writeFileSync(file, JSON.stringify(payload), "utf8");
  return { directory, file };
}

export async function inspectEnvironment(
  pythonPath: string,
  runtimeRoot?: string,
  modelDir?: string,
): Promise<EnvironmentReport> {
  const request = writeRequest({ runtimeRoot, modelDir });
  try {
    const events = await bridge(pythonPath, ["capabilities", "--request", request.file], () => undefined);
    const health = events.find((event) => event.event === "capabilities");
    if (!health) throw new Error("로컬 브리지가 capability 응답을 보내지 않았습니다.");
    return {
      ready: Boolean(health.ready),
      everyricVersion: health.engineVersion ?? "unknown",
      nodeVersion: process.version,
      platform: health.platform ?? `${os.platform()} ${os.arch()}`,
      cpu: os.cpus()[0]?.model ?? "Unknown CPU",
      systemMemoryGb: Math.round((os.totalmem() / 1024 / 1024 / 1024) * 10) / 10,
      gpuName: health.gpu?.name,
      vramTotalMb: health.gpu?.vramTotalMb,
      vramFreeMb: health.gpu?.vramFreeMb,
      cudaVersion: health.cuda?.runtime,
      recommended: { minimumVramGb: 6, comfortableVramGb: 8, systemMemoryGb: 16 },
      notes: health.problems ?? [],
    };
  } finally {
    fs.rmSync(request.directory, { recursive: true, force: true });
  }
}

export async function runLocalSync(
  options: LocalSyncOptions,
  onProgress: (message: string) => void,
  signal?: AbortSignal,
): Promise<unknown> {
  const request = writeRequest({
    protocol: BRIDGE_PROTOCOL,
    audioPath: options.audioPath,
    lyrics: options.lyrics,
    language: options.language === "auto" ? null : options.language,
    minDepth: options.minDepth,
    runtimeRoot: options.runtimeRoot,
    modelDir: options.modelDir,
  });
  try {
    const events = await bridge(options.pythonPath, ["align", "--request", request.file], (event) => {
      if (event.event === "progress" && event.stage) onProgress(event.stage);
      else if (event.event === "depth" && event.depth) onProgress(`분석 깊이: ${event.depth}`);
    }, signal);
    const document = [...events].reverse().find((event) => event.event === "result")?.document;
    if (!document || document.schema !== RESULT_SCHEMA) {
      throw new Error("로컬 브리지 결과 스키마가 올바르지 않습니다.");
    }
    return document.result;
  } finally {
    fs.rmSync(request.directory, { recursive: true, force: true });
  }
}
