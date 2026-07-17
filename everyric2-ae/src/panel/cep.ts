interface AdobeCepBridge {
  evalScript(script: string, callback: (result: string) => void): void;
}

declare global {
  interface Window {
    __adobe_cep__?: AdobeCepBridge;
  }
}

const HOST_FUNCTIONS = new Set([
  "everyricGetCompInfo",
  "everyricGetSelectedTextLayers",
  "everyricApplyTextAssignments",
  "everyricCreateTypography",
  "everyricRemoveGeneratedLayers",
  "everyricCreateLineMarkers",
  "everyricRemoveGeneratedMarkers",
  "everyricPickFile",
]);

function bridge(): AdobeCepBridge {
  if (!window.__adobe_cep__) {
    throw new Error("After Effects CEP 브리지를 찾을 수 없습니다.");
  }
  return window.__adobe_cep__;
}

export function isCepHost(): boolean {
  return Boolean(window.__adobe_cep__);
}

export function evalHost<T>(functionName: string, payload?: unknown): Promise<T> {
  if (!HOST_FUNCTIONS.has(functionName)) {
    return Promise.reject(new Error(`허용되지 않은 호스트 함수: ${functionName}`));
  }
  const encoded = payload === undefined ? "" : JSON.stringify(JSON.stringify(payload));
  const script = `${functionName}(${encoded})`;
  return new Promise((resolve, reject) => {
    try {
      bridge().evalScript(script, (result) => {
        if (!result || result === "EvalScript error.") {
          reject(new Error("After Effects 스크립트 실행에 실패했습니다."));
          return;
        }
        try {
          resolve(JSON.parse(result) as T);
        } catch {
          reject(new Error(`After Effects 응답을 해석할 수 없습니다: ${result.slice(0, 180)}`));
        }
      });
    } catch (error) {
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
