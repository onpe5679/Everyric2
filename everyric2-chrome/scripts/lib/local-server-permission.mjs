// 로컬(자체 호스팅) 싱크 서버의 optional_host_permissions를 E2E 스크립트가 실제
// 사용자 경로로 부여하게 하는 공용 헬퍼.
//
// **왜 이게 필요한가.** manifest.json은 `http://127.0.0.1:8000/*`를 optional_host_permissions로
// 선언한다(src/lib/host-permissions.ts) — 로컬 서버를 쓰는 소수만 허용하면 되므로, 설치 시
// 전원에게 부여되지 않는다. 확장은 옵션 페이지(src/options.ts)의 "허용" 버튼을 눌러야만
// `chrome.permissions.request()`를 부를 수 있다(user gesture 필요, SW 단독 호출은 실패).
//
// **왜 playwright 클릭만으론 안 되는가.** DOM의 "허용" 버튼(#grant) 클릭까지는 playwright로
// 문제없이 재현된다 — 진짜 신뢰된(isTrusted) 클릭 이벤트라 user gesture가 유지된다. 하지만
// 그 클릭이 여는 승인 버블(사용자에게 "…추가 승인을 요청했습니다 / 허용 / 거부"를 보여주는
// 팝업)은 Chrome이 **의도적으로 자동화 표면 밖에 둔 네이티브(OS) UI**다. 실측(2026-08 조사,
// scripts/_perm-experiment*.mjs — 조사 완료 후 삭제됨):
//   - playwright ctx.pages()에 잡히지 않는다(별도 페이지가 아니다)
//   - CDP Target.getTargets에도 안 잡힌다(별도 CDP 타겟이 아니다 — 네이티브 Views UI)
//   - CDP Input.dispatchKeyEvent(Tab/Enter)도 버블에 전달되지 않는다(렌더러 대상이라 못 넘어감)
// 이건 확장이 스스로에게 위험한 권한을 스크립트로 자동 승인하는 것을 막는 크롬의 의도된
// 보안 경계라, 여기서 CDP로 뚫을 우회로는 없다(=주작 없이 이 결론에 도달).
//
// **그래서 어떻게 뚫나.** Windows 접근성 API(UI Automation — 스크린리더가 쓰는 것과 동일한
// 시스템 경로)로 승인 버블의 "허용" 버튼을 찾아 InvokePattern으로 누른다. 이건 매니페스트를
// 고치거나 프로필 파일을 조작하는 우회가 아니라 — 실제 화면에 뜬 실제 네이티브 버튼을,
// CDP가 아닌 다른(그러나 여전히 OS 표준) 자동화 경로로 누르는 것이다. 실측(2026-08):
//   - 버튼은 픽셀 좌표가 아니라 UI 트리 이름("허용"/"Allow")으로 찾으므로 창 배치나 다른
//     창의 가림에 영향받지 않는다(SendInput 좌표 클릭은 실측에서 다른 창에 가려 실패했다).
//   - 클릭 후 수 초 안에 chrome.permissions.contains()가 true로 바뀌는 것까지 확인했다.
//
// **한계.** 이 메커니즘은 Windows 전용이다(System.Windows.Automation). 다른 OS에서는
// 명시적으로 실패한다 — 조용히 넘어가지 않는다(팀 방침: 실패를 삼키지 않는다).
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

/** manifest.json의 optional_host_permissions와 문자열까지 같아야 한다(src/lib/host-permissions.ts와 동기화). */
export const CANONICAL_LOCAL_ORIGIN = 'http://127.0.0.1:8000/*';
export const LOCAL_SERVER_ORIGINS = ['http://localhost:8000/*', CANONICAL_LOCAL_ORIGIN];

/** `http://localhost:8000` → `http://127.0.0.1:8000` (host-permissions.ts normalizeLoopbackUrl과 동일 규칙) */
export function normalizeLoopbackUrl(url) {
  return url.replace(/^(https?:\/\/)localhost(?=[:/]|$)/i, '$1127.0.0.1');
}

export function isLoopbackServerUrl(url) {
  let host;
  try {
    host = new URL(url).hostname.toLowerCase();
  } catch {
    return false;
  }
  if (host === 'localhost' || host.endsWith('.localhost')) return true;
  if (host === '::1' || host === '[::1]') return true;
  return /^127\.\d+\.\d+\.\d+$/.test(host);
}

/** serverUrl이 로컬이면 manifest에 선언된 optional 패턴을, 아니면 null을 돌려준다.
 *  로컬인데 선언 안 된 포트면(허용 불가능한 주소) 명시적으로 null을 돌려준다 — 호출부가
 *  isLoopbackServerUrl로 구분해 "원격이라 필요없음"과 "로컬인데 허용 불가"를 가른다. */
export function declaredLocalPattern(serverUrl) {
  if (!isLoopbackServerUrl(serverUrl)) return null;
  const origin = new URL(normalizeLoopbackUrl(serverUrl)).origin;
  return LOCAL_SERVER_ORIGINS.find(p => p.replace(/\/\*$/, '') === origin) ?? null;
}

async function hasPermission(sw, pattern) {
  return sw.evaluate(p => chrome.permissions.contains({ origins: [p] }), pattern);
}

/**
 * 버블이 뜬 직후의 클릭을 크롬이 무시하는 구간(입력 보호) — 그만큼 기다렸다 Invoke한다.
 *
 * 실측(2026-08-03): 버튼을 **찾자마자** Invoke하면 UI Automation은 성공을 돌려주는데
 * (InvokePattern 예외 없음) 권한은 끝내 부여되지 않는다 — chrome.permissions.contains()가
 * 계속 false다. 같은 코드에서 Invoke 전에 2.5초를 두면 그대로 부여된다. 크롬이 권한
 * 프롬프트에 거는 클릭재킹 방지 지연(버블이 뜬 직후 일정 시간의 입력을 버린다)에 걸린
 * 것이고, 머신 부하에 따라 됐다 안 됐다 하던 원인도 이것이다(빠른 머신일수록 잘 실패한다).
 */
const BUBBLE_INPUT_PROTECTION_MS = 1800;

/**
 * UI Automation으로 옵션 페이지 창(제목에 titleSubstr 포함) 아래에서 네이티브 승인
 * 버블의 "허용"/"Allow" 버튼을 찾아 Invoke한다. timeoutMs 동안 폴링한다.
 * PropertyCondition의 -like 와일드카드 매칭이 아니라 문자열 포함 검사를 PowerShell에서
 * 직접 하므로, 크롬 채널(Chrome for Testing/Chromium/Edge)이 창 제목에 뭘 덧붙이든 안전하다.
 */
async function clickNativeAllowBubble(titleSubstr, timeoutMs) {
  if (/["'$`]/.test(titleSubstr)) throw new Error('titleSubstr에 안전하지 않은 문자가 있음');
  const script = `
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$deadline = (Get-Date).AddMilliseconds(${timeoutMs})
$found = $false
while ((Get-Date) -lt $deadline -and -not $found) {
  $root = [System.Windows.Automation.AutomationElement]::RootElement
  $children = $root.FindAll([System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition)
  foreach ($win in $children) {
    $name = $null
    try { $name = $win.Current.Name } catch { continue }
    if (-not $name -or -not $name.Contains('${titleSubstr}')) { continue }
    $btnCond = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
    $buttons = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants, $btnCond)
    foreach ($b in $buttons) {
      $bn = $null
      try { $bn = $b.Current.Name } catch { continue }
      if ($bn -eq '허용' -or $bn -eq 'Allow') {
        # 버블이 방금 떴다면 크롬이 이 클릭을 버린다 — 입력 보호 구간을 넘기고 누른다
        Start-Sleep -Milliseconds ${BUBBLE_INPUT_PROTECTION_MS}
        try { $b.SetFocus() } catch { }
        $inv = $b.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
        $inv.Invoke()
        Write-Output 'CLICKED'
        $found = $true
        break
      }
    }
    if ($found) { break }
  }
  if (-not $found) { Start-Sleep -Milliseconds 300 }
}
if (-not $found) { Write-Output 'NOT_FOUND' }
`;
  const { stdout } = await execFileAsync(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-Command', script],
    { timeout: timeoutMs + 10000 },
  );
  return stdout.includes('CLICKED');
}

/**
 * 옵션 페이지를 열어 "허용"을 누르고(playwright, 진짜 신뢰된 클릭), 그 클릭이 여는 네이티브
 * 버블은 UI Automation으로 누른다. 이미 허용돼 있으면 아무것도 하지 않고 반환한다.
 *
 * 실패(윈도우 아님 / 버블을 못 찾음 / 끝내 허용 안 됨)는 예외로 던진다 — 여기서 삼키면
 * 이후 모든 단계가 "권한이 없어요" 상태로 조용히 무너진다(팀 리드가 관찰한 원증상).
 */
export async function ensureLocalServerPermission(ctx, sw, extId, {
  pattern = CANONICAL_LOCAL_ORIGIN,
  log = (...a) => console.log(...a),
} = {}) {
  if (await hasPermission(sw, pattern)) {
    log(`local server permission already granted (${pattern})`);
    return;
  }
  if (process.platform !== 'win32') {
    throw new Error(
      `로컬 서버 권한(${pattern})이 없고, 이 스크립트의 자동 부여는 Windows(UI Automation)에서만 ` +
      `지원됩니다. 이 플랫폼(${process.platform})에서는 확장 옵션 페이지(chrome-extension://${extId}/src/options.html)에서 ` +
      '직접 "허용"을 눌러야 합니다.',
    );
  }

  log('local server permission missing — opening options page to grant');
  const optsPage = await ctx.newPage();
  try {
    await optsPage.goto(`chrome-extension://${extId}/src/options.html`, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await optsPage.waitForSelector('#grant', { state: 'visible', timeout: 10000 });
    // 클릭 직후 곧바로 네이티브 버블이 뜬다 — await로 사이를 벌리면 user gesture가 만료될
    // 수 있으니 클릭은 바로 하고, 그 다음에 UI Automation을 기다린다.
    await optsPage.locator('#grant').click();
    log('clicked options page grant button — waiting for native OS permission bubble (Windows UI Automation)');

    const clicked = await clickNativeAllowBubble('Everyric 권한 설정', 15000);
    if (!clicked) {
      throw new Error(
        '네이티브 권한 승인 버블에서 "허용" 버튼을 찾지 못했습니다(UI Automation 타임아웃) — ' +
        '옵션 페이지 창이 실제로 떴는지, 다른 프로세스가 접근성 트리를 가로채지 않는지 확인하세요.',
      );
    }

    // 버블 클릭 후 권한 반영까지 실측 1~수초 — 짧게 폴링해서 확정한다
    const deadline = Date.now() + 10000;
    while (Date.now() < deadline) {
      if (await hasPermission(sw, pattern)) {
        log(`local server permission granted (${pattern})`);
        return;
      }
      await new Promise(r => setTimeout(r, 500));
    }
    throw new Error(`"허용"을 눌렀지만 chrome.permissions.contains(${pattern})가 끝내 true가 되지 않았습니다.`);
  } finally {
    await optsPage.close().catch(() => {});
  }
}

/**
 * serverUrl 하나만 알면 되는 편의 함수 — 원격(비로컬) 서버면 조용히 아무것도 안 한다
 * (필수 host_permissions로 이미 허용돼 있으므로), 로컬인데 manifest에 선언 안 된
 * 포트/주소면 여기서 명확히 실패한다(옵션 페이지에서도 허용할 방법이 없는 경우라
 * 조용히 넘어가면 이후 전부 원인불명 실패가 된다).
 */
export async function ensureLocalServerPermissionForServerUrl(ctx, sw, extId, serverUrl, opts = {}) {
  if (!isLoopbackServerUrl(serverUrl)) return; // 원격 서버 — manifest 필수 권한으로 이미 허용됨
  const pattern = declaredLocalPattern(serverUrl);
  if (pattern === null) {
    const allowed = LOCAL_SERVER_ORIGINS.map(p => p.replace(/\/\*$/, '')).join(', ');
    throw new Error(`${serverUrl}은 확장이 허용할 수 있는 로컬 서버가 아닙니다 (허용 가능: ${allowed})`);
  }
  return ensureLocalServerPermission(ctx, sw, extId, { pattern, ...opts });
}
