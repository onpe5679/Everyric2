import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

export interface EngineInstallOptions {
  wheelUrl?: string;
  onProgress: (message: string) => void;
  signal?: AbortSignal;
}

const UV_VERSION = "0.11.29";
const UV_DOWNLOAD_URL = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-pc-windows-msvc.zip`;
const PYTHON_VERSION = "3.11";
const CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124";
const FALLBACK_ENGINE_SPEC = "everyric2 @ git+https://github.com/onpe5679/Everyric2.git";

function managedRoot(): string {
  return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "Everyric");
}

export function managedPythonPath(): string {
  return path.join(managedRoot(), "runtime", "Scripts", "python.exe");
}

export function hasManagedRuntime(): boolean {
  return fs.existsSync(managedPythonPath());
}

function abortError(): Error {
  return new Error("엔진 설치를 취소했습니다.");
}

function runCommand(
  command: string,
  args: string[],
  onProgress: (message: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUTF8: "1", NO_COLOR: "1" },
    });
    let tail = "";
    const forward = (chunk: Buffer): void => {
      const text = chunk.toString("utf8");
      tail = (tail + text).slice(-2400);
      const line = text.trim().split(/\r?\n/).filter(Boolean).pop();
      if (line) onProgress(line.replace(/\x1b\[[0-9;]*m/g, "").slice(0, 140));
    };
    child.stdout.on("data", forward);
    child.stderr.on("data", forward);
    const abort = (): void => {
      child.kill();
    };
    signal?.addEventListener("abort", abort, { once: true });
    child.on("error", (error) => {
      signal?.removeEventListener("abort", abort);
      reject(error);
    });
    child.on("close", (code) => {
      signal?.removeEventListener("abort", abort);
      if (signal?.aborted) reject(abortError());
      else if (code === 0) resolve();
      else reject(new Error(tail.trim().slice(-800) || `종료 코드 ${code}`));
    });
  });
}

async function downloadFile(url: string, target: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(url, signal ? { signal } : {});
  if (!response.ok) throw new Error(`다운로드 실패 (HTTP ${response.status}): ${url}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, buffer);
}

async function ensureUv(onProgress: (message: string) => void, signal?: AbortSignal): Promise<string> {
  const binDir = path.join(managedRoot(), "bin");
  const uvPath = path.join(binDir, "uv.exe");
  if (fs.existsSync(uvPath)) return uvPath;
  onProgress(`uv ${UV_VERSION} 다운로드 중…`);
  const zipPath = path.join(binDir, "uv.zip");
  await downloadFile(UV_DOWNLOAD_URL, zipPath, signal);
  if (signal?.aborted) throw abortError();
  await runCommand(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      `Expand-Archive -LiteralPath "${zipPath}" -DestinationPath "${binDir}" -Force`,
    ],
    onProgress,
    signal,
  );
  fs.rmSync(zipPath, { force: true });
  if (!fs.existsSync(uvPath)) throw new Error("uv.exe 압축 해제에 실패했습니다.");
  return uvPath;
}

export function detectNvidiaGpu(): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn("nvidia-smi", ["-L"], { windowsHide: true, stdio: ["ignore", "pipe", "ignore"] });
    let stdout = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString("utf8")));
    child.on("error", () => resolve(false));
    child.on("close", (code) => resolve(code === 0 && stdout.trim().length > 0));
  });
}

export async function installEngine(options: EngineInstallOptions): Promise<string> {
  const { onProgress, signal } = options;
  if (process.platform !== "win32") throw new Error("관리형 런타임 설치는 Windows에서만 지원합니다.");
  fs.mkdirSync(managedRoot(), { recursive: true });
  const uvPath = await ensureUv(onProgress, signal);
  if (signal?.aborted) throw abortError();

  const runtimeDir = path.join(managedRoot(), "runtime");
  const pythonPath = managedPythonPath();
  if (!fs.existsSync(pythonPath)) {
    onProgress(`Python ${PYTHON_VERSION} 가상환경 생성 중… (최초 1회)`);
    await runCommand(uvPath, ["venv", runtimeDir, "--python", PYTHON_VERSION], onProgress, signal);
  }
  if (signal?.aborted) throw abortError();

  const gpu = await detectNvidiaGpu();
  onProgress(gpu ? "엔진 설치 중 · CUDA 빌드 (수 GB 다운로드)…" : "엔진 설치 중 · CPU 빌드…");
  const args = ["pip", "install", "--python", pythonPath, "--upgrade", options.wheelUrl || FALLBACK_ENGINE_SPEC];
  if (gpu) args.push("--extra-index-url", CUDA_INDEX_URL, "--index-strategy", "unsafe-best-match");
  await runCommand(uvPath, args, onProgress, signal);
  return pythonPath;
}
