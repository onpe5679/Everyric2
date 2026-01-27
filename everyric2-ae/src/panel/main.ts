import type { Settings, CompInfo, TimestampData, ProcessingStatus, LrcLibResult } from "./types";
import { checkCliInstalled, runLocalAlignment, formatDuration } from "./local";
import { generateSyncCloud } from "./api";
import { searchLrcLib, parseLrcToPlainText, formatDuration as formatLrcDuration } from "./lrclib";

declare const CSInterface: new () => {
  evalScript: (script: string, callback?: (result: string) => void) => void;
  getSystemPath: (pathType: string) => string;
};

const cs = new CSInterface();

const DEFAULT_SETTINGS: Settings = {
  cliPath: "everyric2",
  apiUrl: "https://api.everyric.com",
  markerColor: "green",
  fontSize: 60,
  processMode: "local",
  outputType: "both",
  translate: false,
  pronunciation: false,
  language: "auto",
  segmentMode: "line",
};

class EveryricPanel {
  private settings: Settings;
  private status: ProcessingStatus = "idle";
  private currentAudioPath: string | null = null;
  private currentAudioLayerIndex: number | null = null;
  private lastResult: TimestampData[] | null = null;
  private selectedSearchResult: LrcLibResult | null = null;
  private abortController: AbortController | null = null;

  constructor() {
    this.settings = this.loadSettings();
    this.initializeUI();
    this.checkEnvironment();
  }

  private loadSettings(): Settings {
    try {
      const stored = localStorage.getItem("everyric2_settings");
      if (stored) {
        return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
      }
    } catch {
    }
    return { ...DEFAULT_SETTINGS };
  }

  private saveSettings(): void {
    localStorage.setItem("everyric2_settings", JSON.stringify(this.settings));
  }

  private initializeUI(): void {
    this.bindElement("settingsBtn", "click", () => this.showOverlay("settingsOverlay"));
    this.bindElement("selectAudioBtn", "click", () => this.selectAudioLayer());
    this.bindElement("searchLrcBtn", "click", () => this.showOverlay("searchOverlay"));
    this.bindElement("loadLrcBtn", "click", () => this.loadLrcFile());
    this.bindElement("generateBtn", "click", () => this.generateSync());
    this.bindElement("cancelBtn", "click", () => this.cancelProcessing());

    this.bindElement("settingsCancelBtn", "click", () => this.hideOverlay("settingsOverlay"));
    this.bindElement("settingsSaveBtn", "click", () => this.saveSettingsFromUI());

    this.bindElement("searchCancelBtn", "click", () => this.hideOverlay("searchOverlay"));
    this.bindElement("searchExecuteBtn", "click", () => this.executeSearch());
    this.bindElement("searchSelectBtn", "click", () => this.selectSearchResult());
    this.bindElement("searchQueryInput", "keypress", (e) => {
      if ((e as KeyboardEvent).key === "Enter") this.executeSearch();
    });

    this.bindElement("newTaskBtn", "click", () => this.resetForNewTask());
    this.bindElement("saveLrcBtn", "click", () => this.saveAsLrc());
    this.bindElement("saveSrtBtn", "click", () => this.saveAsSrt());

    document.querySelectorAll('input[name="outputType"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        this.settings.outputType = (e.target as HTMLInputElement).value as Settings["outputType"];
      });
    });

    document.querySelectorAll('input[name="processMode"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        this.settings.processMode = (e.target as HTMLInputElement).value as Settings["processMode"];
      });
    });

    this.bindElement("translateCheck", "change", (e) => {
      this.settings.translate = (e.target as HTMLInputElement).checked;
    });

    this.bindElement("pronunciationCheck", "change", (e) => {
      this.settings.pronunciation = (e.target as HTMLInputElement).checked;
    });

    this.bindElement("languageSelect", "change", (e) => {
      this.settings.language = (e.target as HTMLSelectElement).value;
    });

    document.querySelectorAll('input[name="segmentMode"]').forEach((el) => {
      el.addEventListener("change", (e) => {
        this.settings.segmentMode = (e.target as HTMLInputElement).value as Settings["segmentMode"];
      });
    });

    this.applySettingsToUI();
  }

  private bindElement(id: string, event: string, handler: (e: Event) => void): void {
    const el = document.getElementById(id);
    if (el) el.addEventListener(event, handler);
  }

  private applySettingsToUI(): void {
    const cliPath = document.getElementById("cliPathInput") as HTMLInputElement;
    const apiUrl = document.getElementById("apiUrlInput") as HTMLInputElement;
    const markerColor = document.getElementById("markerColorSelect") as HTMLSelectElement;
    const fontSize = document.getElementById("fontSizeInput") as HTMLInputElement;

    if (cliPath) cliPath.value = this.settings.cliPath;
    if (apiUrl) apiUrl.value = this.settings.apiUrl;
    if (markerColor) markerColor.value = this.settings.markerColor;
    if (fontSize) fontSize.value = String(this.settings.fontSize);

    const outputRadio = document.querySelector(
      `input[name="outputType"][value="${this.settings.outputType}"]`
    ) as HTMLInputElement;
    if (outputRadio) outputRadio.checked = true;

    const modeRadio = document.querySelector(
      `input[name="processMode"][value="${this.settings.processMode}"]`
    ) as HTMLInputElement;
    if (modeRadio) modeRadio.checked = true;

    const translateCheck = document.getElementById("translateCheck") as HTMLInputElement;
    const pronunciationCheck = document.getElementById("pronunciationCheck") as HTMLInputElement;
    const languageSelect = document.getElementById("languageSelect") as HTMLSelectElement;

    if (translateCheck) translateCheck.checked = this.settings.translate;
    if (pronunciationCheck) pronunciationCheck.checked = this.settings.pronunciation;
    if (languageSelect) languageSelect.value = this.settings.language;

    const segmentRadio = document.querySelector(
      `input[name="segmentMode"][value="${this.settings.segmentMode}"]`
    ) as HTMLInputElement;
    if (segmentRadio) segmentRadio.checked = true;
  }

  private saveSettingsFromUI(): void {
    const cliPath = document.getElementById("cliPathInput") as HTMLInputElement;
    const apiUrl = document.getElementById("apiUrlInput") as HTMLInputElement;
    const markerColor = document.getElementById("markerColorSelect") as HTMLSelectElement;
    const fontSize = document.getElementById("fontSizeInput") as HTMLInputElement;

    if (cliPath) this.settings.cliPath = cliPath.value || "everyric2";
    if (apiUrl) this.settings.apiUrl = apiUrl.value || DEFAULT_SETTINGS.apiUrl;
    if (markerColor) this.settings.markerColor = markerColor.value;
    if (fontSize) this.settings.fontSize = parseInt(fontSize.value) || 60;

    this.saveSettings();
    this.hideOverlay("settingsOverlay");
    this.setStatus("Settings saved");
  }

  private async checkEnvironment(): Promise<void> {
    const compInfo = await this.getCompInfo();
    if (compInfo.hasComp) {
      this.updateCompDisplay(compInfo);
    }

    if (this.settings.processMode === "local") {
      const cliStatus = await checkCliInstalled(this.settings.cliPath);
      if (!cliStatus.installed) {
        this.setStatus("everyric2 CLI not found - using cloud mode");
        this.settings.processMode = "cloud";
        const modeRadio = document.querySelector(
          'input[name="processMode"][value="cloud"]'
        ) as HTMLInputElement;
        if (modeRadio) modeRadio.checked = true;
      } else {
        this.setStatus(`Ready (CLI v${cliStatus.version})`);
      }
    } else {
      this.setStatus("Ready (Cloud mode)");
    }
  }

  private getCompInfo(): Promise<CompInfo> {
    return new Promise((resolve) => {
      cs.evalScript("getActiveCompInfo()", (result) => {
        try {
          resolve(JSON.parse(result));
        } catch {
          resolve({ hasComp: false, error: "Failed to parse comp info" });
        }
      });
    });
  }

  private updateCompDisplay(info: CompInfo): void {
    const audioInfo = document.getElementById("audioInfo");
    if (!audioInfo) return;

    if (info.audioLayers && info.audioLayers.length > 0) {
      const layer = info.audioLayers[0];
      this.currentAudioPath = layer.filePath || null;
      this.currentAudioLayerIndex = layer.index;
      audioInfo.innerHTML = `
        <span class="layer-name">${layer.name}</span>
        <span class="duration">${formatDuration(layer.duration)}</span>
      `;
    } else {
      audioInfo.innerHTML = '<span class="placeholder">No audio layer found</span>';
    }
  }

  private selectAudioLayer(): void {
    cs.evalScript("getAudioLayerPath()", (result) => {
      try {
        const data = JSON.parse(result);
        if (data.success) {
          this.currentAudioPath = data.filePath;
          this.currentAudioLayerIndex = data.layerIndex;
          const audioInfo = document.getElementById("audioInfo");
          if (audioInfo) {
            audioInfo.innerHTML = `
              <span class="layer-name">${data.layerName}</span>
              <span class="duration">${formatDuration(data.duration)}</span>
            `;
          }
          this.setStatus("Audio layer selected");
        } else {
          this.setStatus(data.error || "Failed to select audio");
        }
      } catch {
        this.setStatus("Failed to get audio layer");
      }
    });
  }

  private async generateSync(): Promise<void> {
    const lyrics = (document.getElementById("lyricsInput") as HTMLTextAreaElement)?.value?.trim();
    if (!lyrics) {
      this.setStatus("Please enter lyrics");
      return;
    }

    if (!this.currentAudioPath) {
      const compInfo = await this.getCompInfo();
      if (compInfo.audioLayers && compInfo.audioLayers.length > 0) {
        this.currentAudioPath = compInfo.audioLayers[0].filePath || null;
      }
    }

    if (!this.currentAudioPath) {
      this.setStatus("No audio file found");
      return;
    }

    this.status = "processing";
    this.showOverlay("progressOverlay");
    this.abortController = new AbortController();

    try {
      let result;
      if (this.settings.processMode === "local") {
        result = await runLocalAlignment(this.currentAudioPath, lyrics, {
          cliPath: this.settings.cliPath,
          language: this.settings.language,
          translate: this.settings.translate,
          pronunciation: this.settings.pronunciation,
          segmentMode: this.settings.segmentMode,
          onProgress: (status, percent) => this.updateProgress(status, percent),
        });
      } else {
        const audioBlob = await this.loadAudioFile(this.currentAudioPath);
        result = await generateSyncCloud(audioBlob, lyrics, {
          apiUrl: this.settings.apiUrl,
          language: this.settings.language,
          translate: this.settings.translate,
          pronunciation: this.settings.pronunciation,
          onProgress: (status, percent) => this.updateProgress(status, percent),
        });
      }

      this.lastResult = result.segments;
      await this.applyResultsToAE(result.segments);

      this.status = "success";
      this.hideOverlay("progressOverlay");
      this.showResultOverlay(result.segments.length);
    } catch (e) {
      this.status = "error";
      this.hideOverlay("progressOverlay");
      this.setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  private async loadAudioFile(path: string): Promise<Blob> {
    const fs = (window as Window & { cep?: { fs: { readFile: (p: string) => { err: number; data: string } } } }).cep?.fs;
    if (!fs) {
      throw new Error("File system not available");
    }
    
    const result = fs.readFile(path);
    if (result.err !== 0) {
      throw new Error("Failed to read audio file");
    }
    
    const binaryString = atob(result.data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  private updateProgress(status: string, percent: number): void {
    const progressText = document.getElementById("progressText");
    const progressFill = document.getElementById("progressFill");
    if (progressText) progressText.textContent = status;
    if (progressFill) progressFill.style.width = `${percent}%`;
  }

  private async applyResultsToAE(segments: TimestampData[]): Promise<void> {
    const timestampsJson = JSON.stringify(segments);
    const outputType = this.settings.outputType;

    if (outputType === "markers" || outputType === "both") {
      const markerOptions = JSON.stringify({
        color: this.settings.markerColor,
        clearExisting: true,
      });
      await this.evalScriptAsync(`createMarkersFromTimestamps('${this.escapeForJsx(timestampsJson)}', '${this.escapeForJsx(markerOptions)}')`);
    }

    if (outputType === "textLayers" || outputType === "both") {
      const textOptions = JSON.stringify({
        fontSize: this.settings.fontSize,
        includeTranslation: this.settings.translate,
        includePronunciation: this.settings.pronunciation,
      });
      await this.evalScriptAsync(`createTextLayersFromTimestamps('${this.escapeForJsx(timestampsJson)}', '${this.escapeForJsx(textOptions)}')`);
    }
  }

  private escapeForJsx(str: string): string {
    return str.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, "\\n");
  }

  private evalScriptAsync(script: string): Promise<string> {
    return new Promise((resolve) => {
      cs.evalScript(script, resolve);
    });
  }

  private showResultOverlay(count: number): void {
    const stats = document.getElementById("resultStats");
    if (stats) {
      const outputType = this.settings.outputType;
      let html = "";
      if (outputType === "markers" || outputType === "both") {
        html += `<p>Markers created: ${count}</p>`;
      }
      if (outputType === "textLayers" || outputType === "both") {
        html += `<p>Text layers created: ${count}</p>`;
      }
      stats.innerHTML = html;
    }
    this.showOverlay("resultOverlay");
  }

  private cancelProcessing(): void {
    this.abortController?.abort();
    this.hideOverlay("progressOverlay");
    this.status = "idle";
    this.setStatus("Cancelled");
  }

  private resetForNewTask(): void {
    this.hideOverlay("resultOverlay");
    this.lastResult = null;
    (document.getElementById("lyricsInput") as HTMLTextAreaElement).value = "";
    this.setStatus("Ready");
  }

  private async executeSearch(): Promise<void> {
    const query = (document.getElementById("searchQueryInput") as HTMLInputElement)?.value?.trim();
    if (!query) return;

    const resultsContainer = document.getElementById("searchResults");
    if (!resultsContainer) return;

    resultsContainer.innerHTML = '<p class="placeholder">Searching...</p>';

    try {
      const results = await searchLrcLib(query);
      if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="placeholder">No results found</p>';
        return;
      }

      resultsContainer.innerHTML = results
        .slice(0, 10)
        .map(
          (r, i) => `
        <div class="search-result-item" data-index="${i}">
          <div class="title">${this.escapeHtml(r.trackName)}</div>
          <div class="artist">${this.escapeHtml(r.artistName)}</div>
          <div class="meta">${r.syncedLyrics ? "Synced" : "Plain"} | ${formatLrcDuration(r.duration)}</div>
        </div>
      `
        )
        .join("");

      const items = resultsContainer.querySelectorAll(".search-result-item");
      items.forEach((item, index) => {
        item.addEventListener("click", () => {
          items.forEach((i) => i.classList.remove("selected"));
          item.classList.add("selected");
          this.selectedSearchResult = results[index];
          (document.getElementById("searchSelectBtn") as HTMLButtonElement).disabled = false;
        });
      });
    } catch (e) {
      resultsContainer.innerHTML = `<p class="placeholder">Error: ${e instanceof Error ? e.message : String(e)}</p>`;
    }
  }

  private selectSearchResult(): void {
    if (!this.selectedSearchResult) return;

    const lyrics = this.selectedSearchResult.syncedLyrics
      ? parseLrcToPlainText(this.selectedSearchResult.syncedLyrics)
      : this.selectedSearchResult.plainLyrics || "";

    (document.getElementById("lyricsInput") as HTMLTextAreaElement).value = lyrics;
    this.hideOverlay("searchOverlay");
    this.selectedSearchResult = null;
    (document.getElementById("searchSelectBtn") as HTMLButtonElement).disabled = true;
    this.setStatus("Lyrics loaded from LRCLIB");
  }

  private loadLrcFile(): void {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".lrc,.txt";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      
      const reader = new FileReader();
      reader.onload = (ev) => {
        const content = ev.target?.result as string;
        const plainText = parseLrcToPlainText(content);
        (document.getElementById("lyricsInput") as HTMLTextAreaElement).value = plainText;
        this.setStatus("LRC file loaded");
      };
      reader.readAsText(file);
    };
    input.click();
  }

  private saveAsLrc(): void {
    if (!this.lastResult) return;
    const content = this.lastResult
      .map((s) => `[${this.formatLrcTime(s.start)}]${s.text}`)
      .join("\n");
    this.downloadFile(content, "lyrics.lrc", "text/plain");
  }

  private saveAsSrt(): void {
    if (!this.lastResult) return;
    const content = this.lastResult
      .map((s, i) => {
        const startTime = this.formatSrtTime(s.start);
        const endTime = this.formatSrtTime(s.end || s.start + 3);
        return `${i + 1}\n${startTime} --> ${endTime}\n${s.text}\n`;
      })
      .join("\n");
    this.downloadFile(content, "lyrics.srt", "text/plain");
  }

  private formatLrcTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins.toString().padStart(2, "0")}:${secs.padStart(5, "0")}`;
  }

  private formatSrtTime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")},${ms.toString().padStart(3, "0")}`;
  }

  private downloadFile(content: string, filename: string, mimeType: string): void {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  private showOverlay(id: string): void {
    document.getElementById(id)?.classList.remove("hidden");
  }

  private hideOverlay(id: string): void {
    document.getElementById(id)?.classList.add("hidden");
  }

  private setStatus(message: string): void {
    const statusText = document.getElementById("statusText");
    if (statusText) statusText.textContent = message;
  }

  private escapeHtml(str: string): string {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new EveryricPanel();
});
