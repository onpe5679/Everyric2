export const PANEL_VERSION = "2.0.0";

// 이 패널이 지원하는 everyric2 엔진 버전 범위. 릴리즈 시 latest.json의 engineRange와 함께 관리한다.
export const SUPPORTED_ENGINE_RANGE = ">=0.1.0 <1.0.0";

export function parseVersion(value: string): number[] {
  const match = value.trim().replace(/^v/i, "").match(/^(\d+)\.(\d+)(?:\.(\d+))?/);
  if (!match) return [];
  return [Number(match[1]), Number(match[2]), Number(match[3] ?? 0)];
}

export function compareVersions(a: string, b: string): number {
  const left = parseVersion(a);
  const right = parseVersion(b);
  if (left.length === 0 || right.length === 0) return 0;
  for (let index = 0; index < 3; index += 1) {
    const diff = (left[index] ?? 0) - (right[index] ?? 0);
    if (diff !== 0) return diff < 0 ? -1 : 1;
  }
  return 0;
}

export function isNewerVersion(candidate: string, current: string): boolean {
  if (parseVersion(candidate).length === 0 || parseVersion(current).length === 0) return false;
  return compareVersions(candidate, current) > 0;
}

export function satisfiesRange(version: string, range: string): boolean {
  if (parseVersion(version).length === 0) return false;
  const clauses = range.trim().split(/\s+/).filter(Boolean);
  if (clauses.length === 0) return false;
  for (const clause of clauses) {
    const match = clause.match(/^(>=|<=|>|<|=)?(.+)$/);
    if (!match) return false;
    const operator = match[1] ?? "=";
    const target = match[2] ?? "";
    if (parseVersion(target).length === 0) return false;
    const diff = compareVersions(version, target);
    if (operator === ">=" && diff < 0) return false;
    if (operator === "<=" && diff > 0) return false;
    if (operator === ">" && diff <= 0) return false;
    if (operator === "<" && diff >= 0) return false;
    if (operator === "=" && diff !== 0) return false;
  }
  return true;
}
