interface CSInterface {
  getHostEnvironment(): HostEnvironment;
  closeExtension(): void;
  getSystemPath(pathType: string): string;
  evalScript(script: string, callback?: (result: string) => void): void;
  getApplicationID(): string;
  openURLInDefaultBrowser(url: string): void;
  getExtensionID(): string;
  getScaleFactor(): number;
  setScaleFactorChangedHandler(handler: () => void): void;
  getCurrentApiVersion(): ApiVersion;
}

interface HostEnvironment {
  appId: string;
  appVersion: string;
  appLocale: string;
  appUILocale: string;
  isAppOnline: boolean;
  appSkinInfo: AppSkinInfo;
  extensionId: string;
}

interface AppSkinInfo {
  baseFontFamily: string;
  baseFontSize: number;
  appBarBackgroundColor: UIColor;
  panelBackgroundColor: UIColor;
}

interface UIColor {
  color: RGBColor;
}

interface RGBColor {
  red: number;
  green: number;
  blue: number;
  alpha: number;
}

interface ApiVersion {
  major: number;
  minor: number;
  micro: number;
}

declare const SystemPath: {
  USER_DATA: string;
  COMMON_FILES: string;
  MY_DOCUMENTS: string;
  APPLICATION: string;
  EXTENSION: string;
  HOST_APPLICATION: string;
};

declare class CSInterface {
  constructor();
}

interface Window {
  cep?: {
    fs: {
      readFile(path: string, encoding?: string): { err: number; data: string };
      writeFile(path: string, data: string, encoding?: string): { err: number };
      stat(path: string): { err: number; data: { isFile(): boolean; isDirectory(): boolean } };
      makedir(path: string): { err: number };
    };
    process: {
      stdout: string;
      stderr: string;
      isRunning: boolean;
    };
  };
  __adobe_cep__?: {
    evalScript(script: string, callback: (result: string) => void): void;
    getHostEnvironment(): string;
    closeExtension(): void;
    getSystemPath(pathType: string): string;
    invokeSync(name: string, data: string): string;
  };
}

declare const require: (module: string) => unknown;
declare const process: {
  platform: string;
  env: Record<string, string | undefined>;
};
