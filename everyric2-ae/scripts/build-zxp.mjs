import { execFileSync } from "child_process";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(root, "..");
const releaseDir = path.join(root, "release");
const toolsDir = path.join(root, "scripts", ".tools");
const secretsDir = path.join(repoRoot, "secrets");

const ZXPSIGN_URL = "https://github.com/Adobe-CEP/CEP-Resources/raw/master/ZXPSignCMD/4.1.1/win64/ZXPSignCmd.exe";
const TSA_CANDIDATES = ["http://timestamp.digicert.com", "http://timestamp.sectigo.com", null];
const RUNTIME_ENTRIES = ["index.html", "css", "CSXS", "dist"];

function fail(message) {
  console.error(`[build-zxp] ${message}`);
  process.exit(1);
}

// --- 버전 정합성 검사 ---
const packageJson = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const version = packageJson.version;
const manifestXml = fs.readFileSync(path.join(root, "CSXS", "manifest.xml"), "utf8");
if (!manifestXml.includes(`ExtensionBundleVersion="${version}"`)) {
  fail(`manifest.xml ExtensionBundleVersion이 package.json(${version})과 다릅니다.`);
}
if (!new RegExp(`<Extension Id="com\\.everyric\\.studio\\.panel" Version="${version.replace(/\./g, "\\.")}"`).test(manifestXml)) {
  fail(`manifest.xml Extension Version이 package.json(${version})과 다릅니다.`);
}
const versionTs = fs.readFileSync(path.join(root, "src", "panel", "version.ts"), "utf8");
if (!versionTs.includes(`PANEL_VERSION = "${version}"`)) {
  fail(`src/panel/version.ts PANEL_VERSION이 package.json(${version})과 다릅니다.`);
}
const tag = process.env.GITHUB_REF_NAME;
if (tag && tag.startsWith("ae-v") && tag !== `ae-v${version}`) {
  fail(`태그(${tag})와 package.json(${version})이 다릅니다.`);
}

// --- 빌드 산출물 확인 ---
for (const relative of ["dist/js/main.js", "dist/jsx/host.jsx", "index.html", "css/style.css", "CSXS/manifest.xml"]) {
  if (!fs.existsSync(path.join(root, relative))) fail(`빌드 파일 누락: ${relative} — 먼저 npm run build를 실행하세요.`);
}

// --- 스테이징 (런타임 파일만, .debug 제외) ---
const stagingDir = path.join(releaseDir, "staging");
fs.rmSync(stagingDir, { recursive: true, force: true });
fs.mkdirSync(stagingDir, { recursive: true });
for (const entry of RUNTIME_ENTRIES) {
  fs.cpSync(path.join(root, entry), path.join(stagingDir, entry), { recursive: true });
}

// --- ZXPSignCmd 확보 ---
async function ensureZxpSignCmd() {
  if (process.env.ZXPSIGN_CMD && fs.existsSync(process.env.ZXPSIGN_CMD)) return process.env.ZXPSIGN_CMD;
  const cached = path.join(toolsDir, "ZXPSignCmd.exe");
  if (fs.existsSync(cached)) return cached;
  if (process.platform !== "win32") fail("win32 외 플랫폼에서는 ZXPSIGN_CMD 환경 변수로 ZXPSignCmd 경로를 지정하세요.");
  console.log(`[build-zxp] ZXPSignCmd 다운로드: ${ZXPSIGN_URL}`);
  const response = await fetch(ZXPSIGN_URL);
  if (!response.ok) fail(`ZXPSignCmd 다운로드 실패 (HTTP ${response.status})`);
  fs.mkdirSync(toolsDir, { recursive: true });
  fs.writeFileSync(cached, Buffer.from(await response.arrayBuffer()));
  return cached;
}

// --- 인증서 확보 ---
function generateSelfSignedCert(zxpSignCmd) {
  const certPath = path.join(secretsDir, "everyric-selfsigned.p12");
  const passwordPath = path.join(secretsDir, "cert-password.txt");
  const password = crypto.randomBytes(12).toString("base64url");
  fs.mkdirSync(secretsDir, { recursive: true });
  execFileSync(
    zxpSignCmd,
    ["-selfSignedCert", "KR", "Seoul", "Everyric", "Everyric Studio", password, certPath, "-validityDays", "3650"],
    { stdio: "pipe" },
  );
  fs.writeFileSync(passwordPath, password, "utf8");
  console.log(`[build-zxp] 자가 서명 인증서 생성: ${certPath} (비밀번호는 cert-password.txt)`);
  return { certPath, password };
}

function resolveCertificate(zxpSignCmd) {
  const explicitPath = process.env.EVERYRIC_CERT_PATH;
  const defaultPath = path.join(secretsDir, "ElysianCert.p12");
  const generatedPath = path.join(secretsDir, "everyric-selfsigned.p12");
  const passwordFile = path.join(secretsDir, "cert-password.txt");
  const filePassword = fs.existsSync(passwordFile) ? fs.readFileSync(passwordFile, "utf8").trim() : undefined;
  const password = process.env.EVERYRIC_CERT_PASSWORD ?? filePassword ?? "";
  if (explicitPath) {
    if (!fs.existsSync(explicitPath)) fail(`EVERYRIC_CERT_PATH가 존재하지 않습니다: ${explicitPath}`);
    return { certPath: explicitPath, password };
  }
  if (fs.existsSync(defaultPath)) return { certPath: defaultPath, password };
  if (fs.existsSync(generatedPath) && filePassword !== undefined) {
    return { certPath: generatedPath, password: filePassword };
  }
  return generateSelfSignedCert(zxpSignCmd);
}

// --- 서명 ---
function trySign(zxpSignCmd, certPath, password, zxpPath) {
  let lastError = null;
  for (const tsa of TSA_CANDIDATES) {
    const args = ["-sign", stagingDir, zxpPath, certPath, password];
    if (tsa) args.push("-tsa", tsa);
    else console.warn("[build-zxp] 경고: 타임스탬프 서버 없이 서명합니다. 인증서 만료 후 설치가 거부될 수 있습니다.");
    try {
      fs.rmSync(zxpPath, { force: true });
      const output = execFileSync(zxpSignCmd, args, { stdio: "pipe" }).toString("utf8");
      console.log(`[build-zxp] 서명 완료${tsa ? ` (tsa: ${tsa})` : ""} · ${output.trim()}`);
      return true;
    } catch (error) {
      lastError = error;
      const detail = [error.stdout, error.stderr].map((part) => part?.toString("utf8").trim()).filter(Boolean).join(" / ");
      console.warn(`[build-zxp] 서명 시도 실패${tsa ? ` (tsa: ${tsa})` : ""}: ${detail || error.message}`);
    }
  }
  if (lastError) throw lastError;
  return false;
}

const zxpSignCmd = await ensureZxpSignCmd();
const zxpPath = path.join(releaseDir, `Everyric-Studio-${version}.zxp`);
let { certPath, password } = resolveCertificate(zxpSignCmd);
try {
  trySign(zxpSignCmd, certPath, password, zxpPath);
} catch {
  if (process.env.EVERYRIC_CERT_PATH || process.env.EVERYRIC_CERT_PASSWORD) {
    fail("지정된 인증서로 서명하지 못했습니다. 인증서와 비밀번호를 확인하세요.");
  }
  console.warn("[build-zxp] 기본 인증서 서명 실패 — 자가 서명 인증서를 새로 만들어 재시도합니다.");
  ({ certPath, password } = generateSelfSignedCert(zxpSignCmd));
  trySign(zxpSignCmd, certPath, password, zxpPath);
}

// --- 검증 ---
try {
  const verifyOutput = execFileSync(zxpSignCmd, ["-verify", zxpPath], { stdio: "pipe" }).toString("utf8");
  console.log(`[build-zxp] 서명 검증: ${verifyOutput.trim()}`);
} catch (error) {
  fail(`서명 검증 실패: ${error.stdout?.toString("utf8") || error.message}`);
}

// --- 수동 설치 zip (ZXP Installer가 없는 사용자용, PlayerDebugMode 필요) ---
const manualDir = path.join(releaseDir, "manual");
fs.rmSync(manualDir, { recursive: true, force: true });
fs.mkdirSync(manualDir, { recursive: true });
fs.cpSync(stagingDir, path.join(manualDir, "com.everyric.studio"), { recursive: true });
fs.writeFileSync(path.join(manualDir, "install.bat"), [
  "@echo off",
  "setlocal",
  "set TARGET=%APPDATA%\\Adobe\\CEP\\extensions\\com.everyric.studio",
  "echo Installing Everyric Studio to %TARGET%",
  'if exist "%TARGET%" rmdir /s /q "%TARGET%"',
  'xcopy /e /i /y "%~dp0com.everyric.studio" "%TARGET%" >nul',
  'for %%V in (11 12 13) do reg add "HKCU\\Software\\Adobe\\CSXS.%%V" /v PlayerDebugMode /t REG_SZ /d 1 /f >nul',
  "echo Done. Restart After Effects and open Window ^> Extensions ^> Everyric Studio.",
  "pause",
  "",
].join("\r\n"), "ascii");
fs.writeFileSync(path.join(manualDir, "INSTALL.txt"), [
  "Everyric Studio for After Effects — 수동 설치 / Manual install",
  "",
  "[KO] 권장 설치는 릴리스의 .zxp 파일을 aescripts ZXP Installer로 여는 것입니다.",
  "     이 zip은 ZXP Installer를 쓸 수 없는 경우의 대안입니다. install.bat을 실행하면",
  "     사용자 CEP 확장 폴더에 복사하고 PlayerDebugMode를 활성화합니다.",
  "",
  "[EN] The recommended install is opening the release .zxp with aescripts ZXP Installer.",
  "     This zip is a fallback. Run install.bat to copy the panel into the per-user CEP",
  "     extensions folder and enable PlayerDebugMode.",
  "",
  `Version: ${version}`,
  "",
].join("\r\n"), "utf8");

const manualZipPath = path.join(releaseDir, `Everyric-Studio-${version}-manual.zip`);
fs.rmSync(manualZipPath, { force: true });
execFileSync("powershell.exe", [
  "-NoProfile",
  "-NonInteractive",
  "-ExecutionPolicy",
  "Bypass",
  "-Command",
  `Compress-Archive -Path "${manualDir}\\*" -DestinationPath "${manualZipPath}" -Force`,
], { stdio: "pipe" });

// --- 체크섬 ---
function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}
const sums = [zxpPath, manualZipPath]
  .map((filePath) => `${sha256(filePath)}  ${path.basename(filePath)}`)
  .join("\n");
fs.writeFileSync(path.join(releaseDir, "SHA256SUMS.txt"), `${sums}\n`, "utf8");

console.log("[build-zxp] 완료:");
console.log(`  ${zxpPath}`);
console.log(`  ${manualZipPath}`);
console.log(`  ${path.join(releaseDir, "SHA256SUMS.txt")}`);
