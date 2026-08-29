import { spawn } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";

export interface EngineInstallOptions {
  wheelUrl?: string;
  onProgress: (message: string) => void;
  signal?: AbortSignal;
  extensionRoot?: string | null;
  modelDir?: string;
  repair?: boolean;
}

const UV_VERSION = "0.11.29";
const PYTHON_VERSION = "3.11";
const CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128";
const FALLBACK_ENGINE_SPEC = "everyric2[desktop] @ git+https://github.com/onpe5679/Everyric2.git";

export function managedRoot(): string {
  if (process.platform === "darwin") return path.join(os.homedir(), "Library", "Application Support", "Everyric");
  if (process.platform === "linux") return path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "everyric");
  return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "Everyric");
}

function activeRuntimeDir(): string {
  const pointer = path.join(managedRoot(), "active-runtime.json");
  try {
    const value = JSON.parse(fs.readFileSync(pointer, "utf8")) as { directory?: string };
    if (value.directory && path.isAbsolute(value.directory)) return value.directory;
  } catch {
    // First install and old 2.x installs have no activation pointer.
  }
  return path.join(managedRoot(), "runtime");
}
function embeddedPythonPath(runtimeDir = activeRuntimeDir()): string {
  return path.join(runtimeDir, "python.exe");
}
function venvPythonPath(runtimeDir = activeRuntimeDir()): string {
  return process.platform === "win32"
    ? path.join(runtimeDir, "Scripts", "python.exe")
    : path.join(runtimeDir, "bin", "python3");
}
export function managedPythonPath(): string {
  return fs.existsSync(embeddedPythonPath()) ? embeddedPythonPath() : venvPythonPath();
}
export function hasManagedRuntime(): boolean { return fs.existsSync(managedPythonPath()); }

function hasNvidiaGpu(): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn("nvidia-smi", ["-L"], {
      windowsHide: true,
      stdio: ["ignore", "pipe", "ignore"],
    });
    let output = "";
    child.stdout.on("data", (chunk: Buffer) => (output += chunk.toString("utf8")));
    child.on("error", () => resolve(false));
    child.on("close", (code) => resolve(code === 0 && output.trim().length > 0));
  });
}

function seedRuntimeDir(extensionRoot: string | null): string | null {
  if (!extensionRoot || process.platform !== "win32") return null;
  const candidate = path.join(extensionRoot, "runtime", "python.exe");
  return fs.existsSync(candidate) ? path.join(extensionRoot, "runtime") : null;
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
      tail = (tail + text).slice(-4000);
      for (const line of text.split(/\r?\n/).filter(Boolean)) {
        try {
          const event = JSON.parse(line) as { event?: string; message?: string; file?: string; bytes?: number; total?: number };
          const progress = event.message || [event.event, event.file, event.total ? `${event.bytes}/${event.total}` : ""].filter(Boolean).join(" · ");
          if (progress) onProgress(progress.slice(0, 180));
        } catch {
          onProgress(line.replace(/\x1b\[[0-9;]*m/g, "").slice(0, 180));
        }
      }
    };
    child.stdout.on("data", forward);
    child.stderr.on("data", forward);
    const abort = (): void => {
      child.kill();
    };
    signal?.addEventListener("abort", abort, { once: true });
    child.on("error", reject);
    child.on("close", (code) => {
      signal?.removeEventListener("abort", abort);
      if (signal?.aborted) reject(new Error("로컬 엔진 작업을 취소했습니다."));
      else if (code === 0) resolve();
      else reject(new Error(tail.trim().slice(-1200) || `종료 코드 ${code}`));
    });
  });
}

async function ensureUv(onProgress: (message: string) => void, signal?: AbortSignal): Promise<string> {
  const binDir = path.join(managedRoot(), "bin");
  const executable = process.platform === "win32" ? "uv.exe" : "uv";
  const uvPath = path.join(binDir, executable);
  if (fs.existsSync(uvPath)) return uvPath;
  if (process.platform !== "win32") throw new Error("이 플랫폼용 로컬 런타임 번들은 아직 배포되지 않았습니다.");
  fs.mkdirSync(binDir, { recursive: true });
  const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-pc-windows-msvc.zip`;
  const zipPath = path.join(binDir, "uv.zip");
  onProgress(`uv ${UV_VERSION} 다운로드`);
  const response = await fetch(url, signal ? { signal } : {});
  if (!response.ok) throw new Error(`uv 다운로드 실패: HTTP ${response.status}`);
  fs.writeFileSync(zipPath, Buffer.from(await response.arrayBuffer()));
  await runCommand("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", `Expand-Archive -LiteralPath '${zipPath.replace(/'/g, "''")}' -DestinationPath '${binDir.replace(/'/g, "''")}' -Force`], onProgress, signal);
  fs.rmSync(zipPath, { force: true });
  if (!fs.existsSync(uvPath)) throw new Error("uv 압축 해제에 실패했습니다.");
  return uvPath;
}

export async function installEngine(options: EngineInstallOptions): Promise<string> {
  if (process.platform !== "win32" || os.arch() !== "x64") {
    throw new Error("현재 릴리스는 Windows x64 + NVIDIA CUDA만 지원합니다.");
  }
  if (!(await hasNvidiaGpu())) {
    throw new Error("NVIDIA GPU 또는 드라이버를 찾지 못했습니다. CPU/원격 폴백은 제공하지 않습니다.");
  }
  const { onProgress, signal } = options;
  fs.mkdirSync(managedRoot(), { recursive: true });
  let pythonPath = managedPythonPath();
  const updating = fs.existsSync(pythonPath) && !options.repair;
  const targetRuntime = updating
    ? path.join(managedRoot(), `runtime-${Date.now()}`)
    : activeRuntimeDir();
  if (updating) {
    const uv = await ensureUv(onProgress, signal);
    onProgress("업데이트용 격리 런타임 준비");
    await runCommand(uv, ["venv", targetRuntime, "--python", PYTHON_VERSION], onProgress, signal);
    pythonPath = venvPythonPath(targetRuntime);
  } else if (!fs.existsSync(pythonPath)) {
    const seed = seedRuntimeDir(options.extensionRoot ?? null);
    if (seed) {
      onProgress("내장 Python 런타임 설치");
      fs.cpSync(seed, targetRuntime, { recursive: true });
      pythonPath = embeddedPythonPath(targetRuntime);
    } else {
      const uv = await ensureUv(onProgress, signal);
      await runCommand(uv, ["venv", targetRuntime, "--python", PYTHON_VERSION], onProgress, signal);
      pythonPath = venvPythonPath(targetRuntime);
    }
  }
  const uv = await ensureUv(onProgress, signal);
  process.env.EVERYRIC_UV_PATH = uv;
  onProgress("PyTorch CUDA 12.8 런타임 설치");
  await runCommand(uv, ["pip", "install", "--python", pythonPath, "--index-url", CUDA_INDEX_URL, "torch==2.8.0+cu128", "torchaudio==2.8.0+cu128", "torchvision==0.23.0+cu128"], onProgress, signal);
  const spec = options.wheelUrl ? `everyric2[desktop] @ ${options.wheelUrl}` : FALLBACK_ENGINE_SPEC;
  onProgress("Everyric Engine 1.x 설치");
  await runCommand(uv, ["pip", "install", "--python", pythonPath, "--upgrade", spec], onProgress, signal);
  const command = options.repair ? "repair" : "provision";
  const args = ["-m", "everyric2.local_runtime", command, "--root", managedRoot()];
  if (options.modelDir) args.push("--models", options.modelDir);
  await runCommand(pythonPath, args, onProgress, signal);
  fs.writeFileSync(
    path.join(managedRoot(), "active-runtime.json"),
    JSON.stringify({ schemaVersion: 1, directory: targetRuntime, activatedAt: new Date().toISOString() }, null, 2),
    "utf8",
  );
  return pythonPath;
}
