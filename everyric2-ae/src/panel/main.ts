import { evalHost, extensionRoot, isCepHost } from "./cep";
import {
  buildCutSession,
  computePieces,
  defaultCutTime,
  moveCut,
  pieceWarnings,
  toggleCut,
} from "./cutter";
import { installEngine } from "./engine-install";
import { inspectEnvironment, readJsonFile, runLocalSync } from "./local-sync";
import { normalizeSyncPayload, planLayerFill, planLineLyrics, planTypography } from "./planner";
import { resolveScript, selectableLanguages, withTranslationLanguage } from "./lang";
import { compKey, forgetSyncForComp, loadSyncForComp, saveSyncForComp } from "./sync-store";
import { fetchLatestManifest, openExternal, panelUpdate, RELEASES_URL } from "./updater";
import type { LatestManifest } from "./updater";
import { isNewerVersion, parseVersion, satisfiesRange, SUPPORTED_ENGINE_RANGE } from "./version";
import type {
  AppSettings,
  CompInfo,
  CutPoint,
  CutSession,
  Density,
  FillAssignment,
  HostResult,
  TextLayerInfo,
  PlannerOptions,
  SyncLine,
  SyncDocument,
  TypographyPlan,
  TypographyBlock,
  EnvironmentReport,
  UiLocale,
} from "./types";

const DEFAULT_SETTINGS: AppSettings = {
  uiLocale: "ko",
  pythonPath: "python",
  runtimeRoot: "",
  modelDir: "",
  minDepth: "fast",
  language: "auto",
  density: "balanced",
  typographyMode: "designed",
  layout: "auto",
  fontSize: 94,
  preRollFrames: 3,
  postRollFrames: 8,
  pauseThreshold: 0.32,
  maxBlocksPerCard: 4,
  phraseTargetChars: 9,
  maxTokensPerBlock: 4,
  revealMode: "cumulative",
  layerOrder: "bottom-to-top",
  replacePrevious: true,
  autoLabelColors: false,
  keepCutPosition: false,
  cutReveal: "cumulative",
  translationLanguage: "ko",
  pronunciationScript: "auto",
};

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`UI 요소 누락: ${id}`);
  return found as T;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[char] ?? char);
}

function timeLabel(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

const DENSITY_PRESETS: Record<Density, { target: number; tokens: number; pause: number; label: string }> = {
  readable: { target: 12, tokens: 5, pause: 0.48, label: "Readable" },
  balanced: { target: 9, tokens: 4, pause: 0.32, label: "Balanced" },
  rhythmic: { target: 6, tokens: 3, pause: 0.2, label: "Rhythmic" },
};

interface RiskItem {
  level: "warn" | "danger";
  cardId: string;
  blockId: string;
  text: string;
  duration: number;
  charsPerSecond: number;
  reason: string;
  advice: string;
}

interface PlanAnalysis {
  cards: number;
  layers: number;
  averageCharsPerSecond: number;
  shortestDuration: number;
  warnings: RiskItem[];
  dangers: RiskItem[];
}

function visibleLength(text: string): number {
  return Array.from(text).filter((char) => !/\s/.test(char)).length;
}

type LocaleDictionary = Record<string, string>;

const UI_TEXT: Record<UiLocale, LocaleDictionary> = {
  ko: {
    settingsAria: "설정",
    refreshTitle: "컴포지션 새로고침",
    tabSync: "싱크",
    tabFill: "레이어 채우기",
    tabType: "타이포 생성",
    syncTitle: "싱크 데이터",
    syncIntro: "Everyric2 결과를 읽기 좋은 타이포 계획으로 변환합니다.",
    loadJson: "JSON 불러오기",
    localDivider: "또는 로컬에서 새로 정렬",
    lyricsLabel: "가사 원문",
    lyricsPlaceholder: "노래 가사를 줄 단위로 붙여넣으세요.",
    engineCheck: "환경 확인",
    runSync: "현재 오디오 싱크 생성",
    cancel: "취소",
    envHint: "환경 확인을 누르면 Everyric2, RAM, GPU/VRAM 상태와 권장 기준을 표시합니다.",
    progressRecent: "최근 작업 로그",
    syncMicro: "활성 컴포지션의 첫 번째 파일 기반 오디오 레이어를 사용합니다. 서버 코드는 수정하지 않습니다.",
    fillTitle: "선택 레이어 채우기",
    fillIntro: "레이어의 시간과 디자인은 그대로 두고 Source Text만 교체합니다.",
    preserve: "보존",
    preserveText: "Source Text 키프레임과 잠긴 레이어는 안전을 위해 자동으로 건너뜁니다.",
    previewFill: "배정 새로고침",
    applyText: "텍스트 적용",
    addMarkers: "라인 타이밍 마커 추가",
    removeMarkers: "Everyric 마커 전부 제거",
    typeTitle: "타이포그래피 생성",
    typeIntro: "먼저 읽기 호흡을 정하고, 실제 가사가 어떻게 잘리는지 확인한 뒤 생성합니다.",
    rhythm: "읽기 리듬",
    rhythmHint: "프리셋을 고르면 실제 가사 샘플이 바로 바뀝니다.",
    readableDesc: "문장 호흡",
    readableTag: "발라드 · OST · 느린 곡",
    balancedDesc: "권장 기본",
    balancedTag: "팝 · 록 · 대부분의 곡",
    rhythmicDesc: "빠른 호흡",
    rhythmicTag: "랩 · 댄스 · 빠른 훅",
    composition: "화면 구성",
    generationMode: "생성 모드",
    designedMode: "타이포 분할",
    lineMode: "줄가사",
    blockChars: "블록 글자 수",
    reveal: "등장 방식",
    layout: "레이아웃",
    baseSize: "기본 크기",
    advanced: "세부 조정",
    wordsPerBlock: "블록당 최대 단어",
    pauseCut: "쉼 분할 기준",
    preRoll: "미리 등장",
    postRoll: "여운 유지",
    blocksPerCard: "카드당 블록 수",
    showCards: "카드 상세 보기",
    hideCards: "카드 상세 숨기기",
    generateLayers: "AE 레이어 생성",
    cleanup: "참고 마커 · 정리 도구",
    cleanupHint: "Everyric이 만든 것만 지웁니다. 사용자 레이어·마커는 건드리지 않습니다.",
    removeLayers: "Everyric 레이어 제거",
    removeTypeMarkers: "Everyric 마커 제거",
    typeMicro: "생성물은 일반 텍스트 레이어와 마커뿐입니다. Text Animator와 모션 키는 만들지 않습니다.",
    preferences: "작업 설정",
    uiLanguageHint: "패널 UI 언어입니다.",
    pythonHint: "Everyric2가 설치된 Python 실행 파일입니다.",
    layerOrderHint: "타임라인 레이어 스택 방향입니다.",
    autoLabelColors: "레이어 라벨 컬러 자동 분할",
    autoLabelColorsHint: "카드/라인 번호별로 AE 라벨 색을 나눕니다.",
    replacePrevious: "이전 Everyric 생성물 교체",
    replacePreviousHint: "EV2 메타데이터가 있는 레이어만 정리합니다.",
    saveSettings: "설정 저장",
    initialized: "초기화 중…",
    noComp: "컴포지션 없음",
    noCompHint: "AE에서 작업할 컴포지션을 활성화해주세요.",
    noAudio: "파일 기반 오디오 레이어 없음",
    audio: "오디오",
    selectedText: "텍스트 {count}개 선택",
    syncEmpty: "JSON을 불러오거나 로컬 싱크를 생성하면 여기에 구간이 표시됩니다.",
    fillEmpty: "텍스트 레이어를 선택한 뒤 배정을 미리보세요.",
    typeEmpty: "싱크 데이터를 준비하면 카드 상세가 표시됩니다.",
    typeNeedsData: "싱크 데이터가 필요합니다. 싱크 탭에서 JSON을 불러오거나 새로 정렬하세요.",
    splitSampleEmpty: "실제 가사 샘플이 여기에 표시됩니다.",
    planWaiting: "계획 요약 대기 중",
    sampleTitle: "분할 미리보기",
    sampleHint: "실제 가사 기준",
    noRisk: "위험 없음",
    warning: "주의",
    danger: "위험",
    shortest: "최단 노출",
    existingLayers: "기존 Everyric 레이어",
    cardHint: "카드 상세 보기를 누르면 전체 카드 목록을 확인할 수 있습니다.",
    confirmTitle: "생성 확인",
    confirmCreate: "{count}개 텍스트 레이어를 생성합니다.",
    previousGenerated: "이전 Everyric 생성물 {count}개가 있습니다.",
    replace: "교체",
    replaceHint: "Everyric 레이어만 정리 후 생성합니다.",
    confirmSafe: "사용자 레이어·마커는 건드리지 않습니다. Undo 1회로 전체 복원할 수 있습니다.",
    back: "돌아가기",
    runGenerate: "생성 실행",
    statusOpenInAe: "After Effects 안에서 패널을 열어주세요.",
    statusCompReady: "{name} · {fps} fps",
    statusSettingsSaved: "설정을 저장했습니다.",
    statusNeedActiveComp: "활성 컴포지션을 열어주세요.",
    statusSyncReady: "{count}개 가사 구간을 준비했습니다.",
    statusPresetApplied: "리듬 프리셋을 적용했습니다.",
    statusEnvChecking: "Everyric2, 시스템 메모리, GPU/VRAM을 확인하는 중…",
    statusEnvSuccess: "환경 확인 완료 · Everyric2 {version}{vram}",
    statusEnvFailed: "환경 확인 실패 · {error}",
    statusNoAudio: "활성 컴포지션에서 파일 기반 오디오 레이어를 찾지 못했습니다.",
    statusNeedLyrics: "가사를 입력해주세요.",
    statusSyncRunning: "Everyric2가 오디오와 가사를 정렬하는 중…",
    statusSyncProgress: "싱크 진행 · {message}",
    statusCanceling: "작업을 취소하는 중…",
    statusNeedSyncData: "먼저 싱크 데이터를 준비해주세요.",
    statusNoSelectedText: "선택한 텍스트 레이어가 없습니다.",
    statusAutoFillPreviewing: "선택 레이어 배정을 자동으로 미리보는 중…",
    statusFillPreview: "{count}개 레이어에 들어갈 내용을 미리봤습니다.",
    statusApplyingFill: "선택한 레이어의 텍스트를 적용하는 중…",
    statusLayerApplyFailed: "레이어 적용 실패",
    statusFillApplied: "{count}개 레이어의 텍스트를 교체했습니다. Undo 1회로 복원할 수 있습니다.",
    statusGenerateRunning: "After Effects에 타이포그래피 레이어를 생성하는 중…",
    statusGenerateFailed: "타이포그래피 생성 실패",
    statusGenerateSuccess: "{created}개 텍스트 레이어를 생성했습니다. 기존 생성물 {removed}개를 정리했습니다.",
    statusConfirmGenerate: "{count}개 레이어 생성 전 범위를 확인해주세요.",
    statusMarkersAdding: "라인별 참고 타이밍 마커를 추가하는 중…",
    statusMarkersFailed: "마커 생성 실패",
    statusMarkersAdded: "{count}개 라인 타이밍 마커를 추가했습니다.",
    statusMarkersRemoving: "Everyric 참고 마커를 제거하는 중…",
    statusMarkersRemoveFailed: "마커 제거 실패",
    statusMarkersRemoved: "{count}개 Everyric 마커를 제거했습니다.",
    statusLayersRemoving: "Everyric 생성 레이어를 제거하는 중…",
    statusLayersRemoveFailed: "생성 레이어 제거 실패",
    statusLayersRemoved: "{count}개 Everyric 레이어를 제거했습니다. 사용자 레이어는 건드리지 않았습니다.",
    statusParseError: "{error}",
    statusCardDetailsEmpty: "카드 상세가 여기에 표시됩니다.",
    assignedEmpty: "(배정된 가사 없음)",
    vramSuffix: " · VRAM {gb}GB",
    unknown: "확인 안 됨",
    envRecommendation: "권장: 기본 CTC는 CPU/RAM 16GB 이상, GPU 계열 엔진은 VRAM 최소 {min}GB · 권장 {recommended}GB 이상.",
    durationReason: "노출 {seconds}초",
    speedReason: "{speed}자/초",
    durationAdvice: "쉼 분할 기준을 낮추면 앞뒤 블록과 합쳐집니다.",
    speedAdvice: "블록 글자 수를 줄이거나 Rhythmic 프리셋을 시도해보세요.",
    presetModified: "수정됨",
    charsPerSecondMetric: "자/초",
    updateAvailable: "새 버전 v{version} — 클릭하면 릴리스 페이지를 엽니다.",
    engineInstall: "엔진 설치",
    engineUpdate: "엔진 업데이트",
    engineMissingHint: "Everyric2 엔진을 찾지 못했습니다. 아래 버튼으로 자동 설치할 수 있습니다. GPU 감지 시 CUDA 빌드(약 2.5GB), 그 외 CPU 빌드(약 300MB)를 내려받습니다.",
    statusEngineInstalling: "엔진 설치 · {message}",
    statusEngineInstalled: "엔진 설치 완료 · Python 경로를 관리형 런타임으로 설정했습니다.",
    statusEngineInstallFailed: "엔진 설치 실패 · {error}",
    statusEngineUpdateAvailable: "새 엔진 {version} 사용 가능 · 엔진 업데이트 버튼으로 갱신하세요.",
    statusEngineOutOfRange: "엔진 {version}이 지원 범위({range}) 밖입니다. 엔진 업데이트를 권장합니다.",
    tabCut: "글자 커팅",
    cutTitle: "글자 커팅",
    cutIntro: "타이밍이 붙은 텍스트 레이어를 글자 사이에서 자릅니다. 자르는 순간 레이어가 나뉘고 타이밍이 맞춰집니다.",
    loadCutLayer: "선택 레이어 다시 읽기",
    cutRevealLabel: "등장 방식",
    cutRevealCumulative: "이어서 쌓임",
    cutRevealSequential: "차례로 교체",
    applyCut: "자르기 적용",
    keepCutPosition: "원래 위치 유지",
    keepCutPositionHint: "끄면 각 조각이 원래 그 글자가 있던 자리에 남습니다.",
    cutEmpty: "AE에서 텍스트 레이어를 하나 선택하고 불러오세요.",
    cutMicro: "글자 사이를 누르면 자르고, 자른 자리를 다시 누르면 되붙입니다. 경계는 좌우로 끌어 미세 조정할 수 있습니다.",
    cutMatchExact: "싱크 라인과 정확히 일치",
    cutMatchSubstring: "싱크 라인의 일부",
    cutMatchTime: "시간이 겹치는 라인에서 추정",
    cutMatchNone: "싱크 라인 없음 · 레이어 구간을 글자 수로 균등 배분",
    cutPronLabel: "발음",
    cutTransLabel: "번역",
    cutPieceCount: "조각 {count}개",
    cutNoCuts: "아직 자른 곳이 없습니다.",
    statusCutLoaded: "「{name}」 불러옴 · 글자 {count}자",
    statusCutNoSelection: "AE에서 텍스트 레이어를 선택하세요.",
    statusCutApplying: "레이어를 자르는 중…",
    statusCutApplied: "조각 {count}개로 나눴습니다.",
    fetchServer: "서버에서 가져오기",
    videoUrlPlaceholder: "유튜브 주소 또는 영상 ID (서버에 이미 만들어진 싱크)",
    serverUrlHint: "이미 만들어진 싱크를 영상 ID로 조회할 서버입니다.",
    serverKeyHint: "공개 서버에서는 비워 두세요.",
    statusServerNeedVideo: "유튜브 주소나 영상 ID를 입력하세요.",
    statusServerFetching: "서버에서 싱크를 가져오는 중…",
    statusServerFetched: "서버 싱크 {count}줄을 불러왔습니다.",
    statusServerLinked: "다른 영상({source})의 싱크를 빌려온 것입니다. 오프셋이 이 영상과 맞는지 확인하세요.",
    syncStoreNote: "싱크는 컴포지션마다 저장돼, 패널을 닫았다 열어도 그대로 돌아옵니다.",
    forgetSync: "저장된 싱크 지우기",
    statusSyncRestored: "이 컴포지션에 저장해 둔 싱크 {count}줄을 불러왔습니다.",
    statusSyncForgotten: "이 컴포지션의 저장된 싱크를 지웠습니다.",
    translationLangLabel: "번역 언어",
    pronScriptLabel: "발음 표기",
    pronScriptAuto: "자동 (번역 언어 기준)",
    pronScriptHangul: "한글",
    statusLangUnavailable: "이 곡에는 {lang} 번역이 아직 없습니다. 준비된 언어: {available}",
    statusLangNeedsRefetch: "{lang} 번역이 없습니다. 로컬 싱크를 다시 실행해 주세요. 준비된 언어: {available}",
  },
  ja: {
    settingsAria: "設定",
    refreshTitle: "コンポジションを更新",
    tabSync: "同期",
    tabFill: "レイヤー入力",
    tabType: "タイポ生成",
    syncTitle: "同期データ",
    syncIntro: "Everyric2 の結果を読みやすいタイポ計画に変換します。",
    loadJson: "JSONを読み込む",
    localDivider: "またはローカルで新規同期",
    lyricsLabel: "歌詞本文",
    lyricsPlaceholder: "歌詞を行単位で貼り付けてください。",
    engineCheck: "環境確認",
    runSync: "現在のオーディオで同期生成",
    cancel: "キャンセル",
    envHint: "Everyric2、RAM、GPU/VRAM、推奨環境を確認します。",
    progressRecent: "最近の処理ログ",
    syncMicro: "アクティブコンポジションの最初のファイルベース音声レイヤーを使用します。",
    fillTitle: "選択レイヤー入力",
    fillIntro: "レイヤーの時間とデザインは維持し、Source Text だけ置き換えます。",
    preserve: "維持",
    preserveText: "Source Text キーフレームとロック済みレイヤーは安全のため自動的にスキップします。",
    previewFill: "割り当てを更新",
    applyText: "テキスト適用",
    addMarkers: "行タイミングマーカー追加",
    removeMarkers: "Everyricマーカーを全削除",
    typeTitle: "タイポグラフィ生成",
    typeIntro: "読みのリズムを選び、実際の歌詞の分割を確認してから生成します。",
    rhythm: "読みリズム",
    rhythmHint: "プリセットを選ぶと実際の歌詞サンプルが更新されます。",
    readableDesc: "文章の呼吸",
    readableTag: "バラード · OST · 遅い曲",
    balancedDesc: "推奨標準",
    balancedTag: "ポップ · ロック · 多くの曲",
    rhythmicDesc: "速い呼吸",
    rhythmicTag: "ラップ · ダンス · 速いフック",
    composition: "画面構成",
    generationMode: "生成モード",
    designedMode: "タイポ分割",
    lineMode: "行歌詞",
    blockChars: "ブロック文字数",
    reveal: "表示方式",
    layout: "レイアウト",
    baseSize: "基本サイズ",
    advanced: "詳細調整",
    wordsPerBlock: "最大単語数",
    pauseCut: "休止分割基準",
    preRoll: "先行表示",
    postRoll: "余韻保持",
    blocksPerCard: "カード内ブロック数",
    showCards: "カード詳細を表示",
    hideCards: "カード詳細を隠す",
    generateLayers: "AEレイヤー生成",
    cleanup: "参考マーカー · 整理ツール",
    cleanupHint: "Everyricが作成したものだけを削除します。ユーザーのレイヤーやマーカーは変更しません。",
    removeLayers: "Everyricレイヤー削除",
    removeTypeMarkers: "Everyricマーカー削除",
    typeMicro: "生成物は通常のテキストレイヤーとマーカーのみです。Text Animatorやモーションキーは作成しません。",
    preferences: "作業設定",
    uiLanguageHint: "パネルUIの言語です。",
    pythonHint: "Everyric2がインストールされたPython実行ファイルです。",
    layerOrderHint: "タイムラインのレイヤースタック方向です。",
    autoLabelColors: "レイヤーラベル色を自動分割",
    autoLabelColorsHint: "カード/行番号ごとにAEラベル色を分けます。",
    replacePrevious: "以前のEveryric生成物を置換",
    replacePreviousHint: "EV2メタデータを持つレイヤーだけ整理します。",
    saveSettings: "設定を保存",
    initialized: "初期化中…",
    noComp: "コンポジションなし",
    noCompHint: "AEで作業するコンポジションをアクティブにしてください。",
    noAudio: "ファイルベースの音声レイヤーなし",
    audio: "オーディオ",
    selectedText: "テキスト {count}個選択",
    syncEmpty: "JSONを読み込むかローカル同期を生成すると、ここに区間が表示されます。",
    fillEmpty: "テキストレイヤーを選択して割り当てをプレビューしてください。",
    typeEmpty: "同期データを準備するとカード詳細が表示されます。",
    typeNeedsData: "同期データが必要です。同期タブでJSONを読み込むか新規同期してください。",
    splitSampleEmpty: "実際の歌詞サンプルがここに表示されます。",
    planWaiting: "計画サマリー待機中",
    sampleTitle: "分割プレビュー",
    sampleHint: "実際の歌詞基準",
    noRisk: "リスクなし",
    warning: "注意",
    danger: "危険",
    shortest: "最短表示",
    existingLayers: "既存Everyricレイヤー",
    cardHint: "カード詳細を表示すると全カード一覧を確認できます。",
    confirmTitle: "生成確認",
    confirmCreate: "{count}個のテキストレイヤーを生成します。",
    previousGenerated: "以前のEveryric生成物が{count}個あります。",
    replace: "置換",
    replaceHint: "Everyricレイヤーだけ整理してから生成します。",
    confirmSafe: "ユーザーのレイヤーやマーカーは変更しません。Undo 1回で復元できます。",
    back: "戻る",
    runGenerate: "生成実行",
    statusOpenInAe: "After Effects内でパネルを開いてください。",
    statusCompReady: "{name} · {fps} fps",
    statusSettingsSaved: "設定を保存しました。",
    statusNeedActiveComp: "アクティブなコンポジションを開いてください。",
    statusSyncReady: "{count}行の歌詞区間を準備しました。",
    statusPresetApplied: "リズムプリセットを適用しました。",
    statusEnvChecking: "Everyric2、システムメモリ、GPU/VRAMを確認中…",
    statusEnvSuccess: "環境確認完了 · Everyric2 {version}{vram}",
    statusEnvFailed: "環境確認失敗 · {error}",
    statusNoAudio: "アクティブコンポジションでファイルベースの音声レイヤーが見つかりません。",
    statusNeedLyrics: "歌詞を入力してください。",
    statusSyncRunning: "Everyric2がオーディオと歌詞を整列中…",
    statusSyncProgress: "同期進行 · {message}",
    statusCanceling: "処理をキャンセル中…",
    statusNeedSyncData: "先に同期データを準備してください。",
    statusNoSelectedText: "選択されたテキストレイヤーがありません。",
    statusAutoFillPreviewing: "選択レイヤーの割り当てを自動プレビュー中…",
    statusFillPreview: "{count}個のレイヤーに入る内容をプレビューしました。",
    statusApplyingFill: "選択レイヤーのテキストを適用中…",
    statusLayerApplyFailed: "レイヤー適用失敗",
    statusFillApplied: "{count}個のレイヤーのテキストを置き換えました。Undo 1回で復元できます。",
    statusGenerateRunning: "After Effectsにタイポグラフィレイヤーを生成中…",
    statusGenerateFailed: "タイポグラフィ生成失敗",
    statusGenerateSuccess: "{created}個のテキストレイヤーを生成しました。既存生成物{removed}個を整理しました。",
    statusConfirmGenerate: "{count}個のレイヤー生成前に範囲を確認してください。",
    statusMarkersAdding: "行ごとの参考タイミングマーカーを追加中…",
    statusMarkersFailed: "マーカー生成失敗",
    statusMarkersAdded: "{count}個の行タイミングマーカーを追加しました。",
    statusMarkersRemoving: "Everyric参考マーカーを削除中…",
    statusMarkersRemoveFailed: "マーカー削除失敗",
    statusMarkersRemoved: "{count}個のEveryricマーカーを削除しました。",
    statusLayersRemoving: "Everyric生成レイヤーを削除中…",
    statusLayersRemoveFailed: "生成レイヤー削除失敗",
    statusLayersRemoved: "{count}個のEveryricレイヤーを削除しました。ユーザーレイヤーは変更していません。",
    statusParseError: "{error}",
    statusCardDetailsEmpty: "カード詳細がここに表示されます。",
    assignedEmpty: "(割り当てられた歌詞なし)",
    vramSuffix: " · VRAM {gb}GB",
    unknown: "未確認",
    envRecommendation: "推奨: 基本CTCはCPU/RAM 16GB以上、GPU系エンジンはVRAM最低{min}GB · 推奨{recommended}GB以上。",
    durationReason: "表示 {seconds}秒",
    speedReason: "{speed}文字/秒",
    durationAdvice: "休止分割基準を下げると前後のブロックと結合されます。",
    speedAdvice: "ブロック文字数を減らすか、Rhythmicプリセットを試してください。",
    presetModified: "変更済み",
    charsPerSecondMetric: "文字/秒",
    updateAvailable: "新バージョン v{version} — クリックでリリースページを開きます。",
    engineInstall: "エンジンをインストール",
    engineUpdate: "エンジンを更新",
    engineMissingHint: "Everyric2 エンジンが見つかりません。下のボタンで自動インストールできます。GPU検出時はCUDAビルド(約2.5GB)、それ以外はCPUビルド(約300MB)を取得します。",
    statusEngineInstalling: "エンジンインストール · {message}",
    statusEngineInstalled: "エンジンのインストール完了 · Pythonパスを管理ランタイムに設定しました。",
    statusEngineInstallFailed: "エンジンのインストール失敗 · {error}",
    statusEngineUpdateAvailable: "新しいエンジン {version} が利用可能 · エンジン更新ボタンで更新してください。",
    statusEngineOutOfRange: "エンジン {version} はサポート範囲({range})外です。エンジン更新を推奨します。",
    tabCut: "文字カット",
    cutTitle: "文字カット",
    cutIntro: "タイミングの付いたテキストレイヤーを文字の間で切ります。切ると同時にレイヤーが分かれ、タイミングが揃います。",
    loadCutLayer: "選択レイヤーを再読み込み",
    cutRevealLabel: "登場方法",
    cutRevealCumulative: "積み重なる",
    cutRevealSequential: "順に入れ替え",
    applyCut: "カットを適用",
    keepCutPosition: "元の位置を保つ",
    keepCutPositionHint: "オフにすると、各断片は元の文字があった位置に残ります。",
    cutEmpty: "AEでテキストレイヤーを1つ選んで読み込んでください。",
    cutMicro: "文字の間を押すと切れ、切った所をもう一度押すと繋がります。境界は左右にドラッグして微調整できます。",
    cutMatchExact: "同期行と完全一致",
    cutMatchSubstring: "同期行の一部",
    cutMatchTime: "時間が重なる行から推定",
    cutMatchNone: "同期行なし · レイヤー区間を文字数で均等配分",
    cutPronLabel: "発音",
    cutTransLabel: "翻訳",
    cutPieceCount: "断片 {count}個",
    cutNoCuts: "まだ切った所がありません。",
    statusCutLoaded: "「{name}」を読み込みました · {count}文字",
    statusCutNoSelection: "AEでテキストレイヤーを選択してください。",
    statusCutApplying: "レイヤーを切っています…",
    statusCutApplied: "{count}個の断片に分けました。",
    fetchServer: "サーバーから取得",
    videoUrlPlaceholder: "YouTubeのURLまたは動画ID（サーバーにある同期データ）",
    serverUrlHint: "既存の同期データを動画IDで照会するサーバーです。",
    serverKeyHint: "公開サーバーでは空のままにしてください。",
    statusServerNeedVideo: "YouTubeのURLか動画IDを入力してください。",
    statusServerFetching: "サーバーから同期データを取得中…",
    statusServerFetched: "サーバーの同期データ {count} 行を読み込みました。",
    statusServerLinked: "別の動画({source})の同期データを借りたものです。オフセットがこの動画と合うか確認してください。",
    syncStoreNote: "同期データはコンポジションごとに保存され、パネルを閉じても戻ってきます。",
    forgetSync: "保存した同期データを削除",
    statusSyncRestored: "このコンポジションに保存された同期データ {count} 行を読み込みました。",
    statusSyncForgotten: "このコンポジションの保存済み同期データを削除しました。",
    translationLangLabel: "翻訳言語",
    pronScriptLabel: "発音表記",
    pronScriptAuto: "自動（翻訳言語に合わせる）",
    pronScriptHangul: "ハングル",
    statusLangUnavailable: "この曲には{lang}の翻訳がまだありません。利用可能: {available}",
    statusLangNeedsRefetch: "{lang}の翻訳がありません。ローカル同期を再実行してください。利用可能: {available}",
  },
  en: {
    settingsAria: "Settings",
    refreshTitle: "Refresh composition",
    tabSync: "Sync",
    tabFill: "Fill Layers",
    tabType: "Create Type",
    syncTitle: "Sync Data",
    syncIntro: "Convert Everyric2 results into a readable typography plan.",
    loadJson: "Load JSON",
    localDivider: "Or align locally",
    lyricsLabel: "Lyrics",
    lyricsPlaceholder: "Paste lyrics line by line.",
    engineCheck: "Check Environment",
    runSync: "Sync Current Audio",
    cancel: "Cancel",
    envHint: "Check Everyric2, RAM, GPU/VRAM, and recommended specs.",
    progressRecent: "Recent activity",
    syncMicro: "Uses the first file-based audio layer in the active composition.",
    fillTitle: "Fill Selected Layers",
    fillIntro: "Keep layer timing and design; replace Source Text only.",
    preserve: "Preserve",
    preserveText: "Layers with Source Text keyframes or locked state are skipped for safety.",
    previewFill: "Refresh Assignment",
    applyText: "Apply Text",
    addMarkers: "Add Line Timing Markers",
    removeMarkers: "Remove Everyric Markers",
    typeTitle: "Create Typography",
    typeIntro: "Choose the reading rhythm, review real lyric splits, then generate layers.",
    rhythm: "Reading Rhythm",
    rhythmHint: "Choosing a preset updates the real lyric sample immediately.",
    readableDesc: "Sentence breath",
    readableTag: "Ballad · OST · slow songs",
    balancedDesc: "Recommended",
    balancedTag: "Pop · rock · most songs",
    rhythmicDesc: "Fast breath",
    rhythmicTag: "Rap · dance · fast hooks",
    composition: "Composition",
    generationMode: "Generation mode",
    designedMode: "Typography split",
    lineMode: "Line lyrics",
    blockChars: "Block characters",
    reveal: "Reveal",
    layout: "Layout",
    baseSize: "Base size",
    advanced: "Advanced",
    wordsPerBlock: "Max words per block",
    pauseCut: "Pause split",
    preRoll: "Pre-roll",
    postRoll: "Hold after",
    blocksPerCard: "Blocks per card",
    showCards: "Show Card Details",
    hideCards: "Hide Card Details",
    generateLayers: "Create AE Layers",
    cleanup: "Guide Markers · Cleanup",
    cleanupHint: "Only removes items created by Everyric. User layers and markers are untouched.",
    removeLayers: "Remove Everyric Layers",
    removeTypeMarkers: "Remove Everyric Markers",
    typeMicro: "Outputs normal text layers and markers only. No Text Animator or motion keys are created.",
    preferences: "Preferences",
    uiLanguageHint: "Panel UI language.",
    pythonHint: "Python executable where Everyric2 is installed.",
    layerOrderHint: "Timeline layer stack direction.",
    autoLabelColors: "Auto-split layer label colors",
    autoLabelColorsHint: "Assign AE label colors by card/line number.",
    replacePrevious: "Replace previous Everyric output",
    replacePreviousHint: "Only cleans layers with EV2 metadata.",
    saveSettings: "Save Settings",
    initialized: "Initializing…",
    noComp: "No composition",
    noCompHint: "Activate a composition in After Effects.",
    noAudio: "No file-based audio layer",
    audio: "Audio",
    selectedText: "{count} text layers selected",
    syncEmpty: "Load JSON or run local sync to show timing ranges here.",
    fillEmpty: "Select text layers, then preview the assignment.",
    typeEmpty: "Prepare sync data to see card details.",
    typeNeedsData: "Sync data is required. Load JSON or run sync in the Sync tab.",
    splitSampleEmpty: "Real lyric samples will appear here.",
    planWaiting: "Waiting for plan summary",
    sampleTitle: "Split Preview",
    sampleHint: "Based on real lyrics",
    noRisk: "No risk",
    warning: "Warning",
    danger: "Danger",
    shortest: "Shortest",
    existingLayers: "Existing Everyric layers",
    cardHint: "Show card details to review the full card list.",
    confirmTitle: "Confirm Generation",
    confirmCreate: "Create {count} text layers.",
    previousGenerated: "{count} previous Everyric items found.",
    replace: "Replace",
    replaceHint: "Clean Everyric layers only, then generate.",
    confirmSafe: "User layers and markers are untouched. One Undo restores the whole operation.",
    back: "Back",
    runGenerate: "Run Generation",
    statusOpenInAe: "Open this panel inside After Effects.",
    statusCompReady: "{name} · {fps} fps",
    statusSettingsSaved: "Settings saved.",
    statusNeedActiveComp: "Open an active composition.",
    statusSyncReady: "{count} lyric timing lines are ready.",
    statusPresetApplied: "Rhythm preset applied.",
    statusEnvChecking: "Checking Everyric2, system memory, GPU/VRAM…",
    statusEnvSuccess: "Environment check complete · Everyric2 {version}{vram}",
    statusEnvFailed: "Environment check failed · {error}",
    statusNoAudio: "No file-based audio layer was found in the active composition.",
    statusNeedLyrics: "Enter lyrics first.",
    statusSyncRunning: "Everyric2 is aligning audio and lyrics…",
    statusSyncProgress: "Sync progress · {message}",
    statusCanceling: "Canceling task…",
    statusNeedSyncData: "Prepare sync data first.",
    statusNoSelectedText: "No selected text layers.",
    statusAutoFillPreviewing: "Auto-previewing selected layer assignment…",
    statusFillPreview: "Previewed text for {count} layers.",
    statusApplyingFill: "Applying text to selected layers…",
    statusLayerApplyFailed: "Layer apply failed",
    statusFillApplied: "Replaced text in {count} layers. One Undo restores it.",
    statusGenerateRunning: "Creating typography layers in After Effects…",
    statusGenerateFailed: "Typography generation failed",
    statusGenerateSuccess: "Created {created} text layers. Removed {removed} previous generated items.",
    statusConfirmGenerate: "Confirm the scope before creating {count} layers.",
    statusMarkersAdding: "Adding line timing guide markers…",
    statusMarkersFailed: "Marker creation failed",
    statusMarkersAdded: "Added {count} line timing markers.",
    statusMarkersRemoving: "Removing Everyric guide markers…",
    statusMarkersRemoveFailed: "Marker removal failed",
    statusMarkersRemoved: "Removed {count} Everyric markers.",
    statusLayersRemoving: "Removing Everyric generated layers…",
    statusLayersRemoveFailed: "Generated layer removal failed",
    statusLayersRemoved: "Removed {count} Everyric layers. User layers were untouched.",
    statusParseError: "{error}",
    statusCardDetailsEmpty: "Card details will appear here.",
    assignedEmpty: "(No assigned lyrics)",
    vramSuffix: " · VRAM {gb}GB",
    unknown: "Not detected",
    envRecommendation: "Recommended: CPU/RAM 16GB+ for basic CTC; VRAM minimum {min}GB, recommended {recommended}GB+ for GPU engines.",
    durationReason: "Visible for {seconds}s",
    speedReason: "{speed} chars/sec",
    durationAdvice: "Lower the pause split threshold to merge with neighboring blocks.",
    speedAdvice: "Reduce block characters or try the Rhythmic preset.",
    presetModified: "Modified",
    charsPerSecondMetric: "chars/sec",
    updateAvailable: "New version v{version} — click to open the release page.",
    engineInstall: "Install Engine",
    engineUpdate: "Update Engine",
    engineMissingHint: "Everyric2 engine was not found. Use the button below to install it automatically. Downloads the CUDA build (~2.5GB) when a GPU is detected, otherwise the CPU build (~300MB).",
    statusEngineInstalling: "Engine install · {message}",
    statusEngineInstalled: "Engine installed · Python path now points to the managed runtime.",
    statusEngineInstallFailed: "Engine install failed · {error}",
    statusEngineUpdateAvailable: "Engine {version} is available · use Update Engine to upgrade.",
    statusEngineOutOfRange: "Engine {version} is outside the supported range ({range}). Updating is recommended.",
    tabCut: "Character Cut",
    cutTitle: "Character cut",
    cutIntro: "Cut a timed text layer between characters. The split and the retiming happen in one step.",
    loadCutLayer: "Reload selected layer",
    cutRevealLabel: "Reveal",
    cutRevealCumulative: "Stays on screen",
    cutRevealSequential: "Replaces previous",
    applyCut: "Apply cut",
    keepCutPosition: "Keep original position",
    keepCutPositionHint: "When off, each piece stays where its characters were drawn.",
    cutEmpty: "Select one text layer in After Effects, then load it.",
    cutMicro: "Click between characters to cut, click a cut again to rejoin. Drag a boundary sideways to fine-tune it.",
    cutMatchExact: "Exact match with a sync line",
    cutMatchSubstring: "Part of a sync line",
    cutMatchTime: "Estimated from the overlapping line",
    cutMatchNone: "No sync line · the layer's range is spread evenly across characters",
    cutPronLabel: "Pronunciation",
    cutTransLabel: "Translation",
    cutPieceCount: "{count} pieces",
    cutNoCuts: "Nothing has been cut yet.",
    statusCutLoaded: "Loaded \"{name}\" · {count} characters",
    statusCutNoSelection: "Select a text layer in After Effects.",
    statusCutApplying: "Splitting the layer…",
    statusCutApplied: "Split into {count} pieces.",
    fetchServer: "Fetch from server",
    videoUrlPlaceholder: "YouTube URL or video ID (a sync the server already has)",
    serverUrlHint: "Server to look up existing syncs by video ID.",
    serverKeyHint: "Leave empty on the public server.",
    statusServerNeedVideo: "Enter a YouTube URL or video ID.",
    statusServerFetching: "Fetching the sync from the server…",
    statusServerFetched: "Loaded {count} lines from the server.",
    statusServerLinked: "This sync is borrowed from another video ({source}). Check that the offset fits this one.",
    syncStoreNote: "Syncs are stored per composition, so closing the panel does not lose them.",
    forgetSync: "Forget stored sync",
    statusSyncRestored: "Restored {count} lines stored for this composition.",
    statusSyncForgotten: "Removed the sync stored for this composition.",
    translationLangLabel: "Translation",
    pronScriptLabel: "Pronunciation",
    pronScriptAuto: "Auto (follows translation)",
    pronScriptHangul: "Hangul",
    statusLangUnavailable: "This song has no {lang} translation yet. Available: {available}",
    statusLangNeedsRefetch: "The {lang} translation is missing. Run local sync again. Available: {available}",
  },
};

class EveryricStudioPanel {
  private settings = this.loadSettings();
  private comp: CompInfo | null = null;
  private syncDocument: SyncDocument | null = null;
  private typographyPlan: TypographyPlan | null = null;
  private fillAssignments: FillAssignment[] = [];
  private progressLines: Array<{ kind: string; message: string; time: string }> = [];
  private abortController: AbortController | null = null;
  private planTimer: number | null = null;
  private fillPreviewTimer: number | null = null;
  private fillSelectionTimer: number | null = null;
  private lastFillSelectionSignature = "";
  private showCardDetails = false;
  private generationConfirmOpen = false;
  private busy = false;
  private latestManifest: LatestManifest | null = null;
  private engineInstallMode: "install" | "update" = "install";
  private cutSession: CutSession | null = null;
  private cutPoints: CutPoint[] = [];
  private cutSelectionTimer: number | null = null;
  private lastCutSelectionSignature = "";
  /** 지금 보고 있는 컴포지션의 저장 열쇠. 바뀌면 싱크를 갈아탄다. */
  private compStorageKey: string | null = null;

  constructor() {
    this.bindUI();
    this.applySettingsToUI();
    this.renderEmptyState();
    if (!isCepHost()) {
      this.statusKey("error", "statusOpenInAe");
      return;
    }
    void this.refreshComp();
    void this.checkForUpdates();
  }

  private loadSettings(): AppSettings {
    try {
      const saved = localStorage.getItem("everyric_studio_settings_v2");
      if (saved) return { ...DEFAULT_SETTINGS, ...(JSON.parse(saved) as Partial<AppSettings>) };
    } catch {
      // Invalid legacy settings are discarded.
    }
    return { ...DEFAULT_SETTINGS };
  }

  private saveSettings(): void {
    localStorage.setItem("everyric_studio_settings_v2", JSON.stringify(this.settings));
  }

  private bindClick(id: string, handler: () => void | Promise<void>): void {
    element<HTMLButtonElement>(id).addEventListener("click", () => void handler());
  }

  private bindUI(): void {
    document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
      button.addEventListener("click", () => this.showView(button.dataset.view ?? "sync"));
    });
    this.bindClick("refreshCompBtn", () => this.refreshComp());
    this.bindClick("loadJsonBtn", () => this.loadJson());
    this.bindClick("forgetSyncBtn", () => this.forgetStoredSync());
    this.bindClick("checkEngineBtn", () => this.checkEngine());
    this.bindClick("runSyncBtn", () => this.runSync());
    this.bindClick("cancelSyncBtn", () => this.cancelSync());
    this.bindClick("previewFillBtn", () => this.previewFill());
    this.bindClick("applyFillBtn", () => this.applyFill());
    this.bindClick("previewTypeBtn", () => this.toggleCardDetails());
    this.bindClick("generateTypeBtn", () => this.generateTypography());
    this.bindClick("addLineMarkersBtn", () => this.addLineMarkers());
    this.bindClick("removeMarkersBtn", () => this.removeMarkers());
    this.bindClick("removeGeneratedLayersBtn", () => this.removeGeneratedLayers());
    this.bindClick("loadCutLayerBtn", () => this.loadCutLayer());
    this.bindClick("applyCutBtn", () => this.applyCut());
    this.bindCutStage();
    element<HTMLSelectElement>("translationLangSelect").addEventListener("change", () => {
      this.captureSettings(false);
      this.switchTranslationLanguage();
    });
    element<HTMLSelectElement>("pronScriptSelect").addEventListener("change", () => {
      this.captureSettings(false);
      // 표기는 이미 받아 둔 값 중에서 고르는 것이라 다시 그리기만 하면 된다.
      if (this.cutSession) this.reloadCutSessionForScript();
      this.renderCut();
    });
    ["keepCutPositionCheck", "cutRevealSelect"].forEach((id) => {
      element<HTMLElement>(id).addEventListener("change", () => {
        this.captureSettings(false);
        this.renderCut();
      });
    });
    this.bindClick("installEngineBtn", () => this.installEngineFlow());
    this.bindClick("repairEngineBtn", () => this.installEngineFlow(true));
    this.bindClick("updateBadge", () => openExternal(this.latestManifest?.ae?.releaseUrl ?? RELEASES_URL));
    this.bindClick("removeTypeMarkersBtn", () => this.removeMarkers());
    this.bindClick("openSettingsBtn", () => this.toggleSettings(true));
    this.bindClick("closeSettingsBtn", () => this.toggleSettings(false));
    this.bindClick("saveSettingsBtn", () => this.captureSettings());

    element<HTMLSelectElement>("uiLocaleSelect").addEventListener("change", () => {
      this.settings.uiLocale = element<HTMLSelectElement>("uiLocaleSelect").value as UiLocale;
      this.saveSettings();
      this.applyLocaleToUI();
      this.renderComp();
      if (this.syncDocument) this.setSyncDocument(this.syncDocument);
      else this.renderEmptyState();
      this.recalculateTypographyPlan();
    });

    document.querySelectorAll<HTMLButtonElement>("[data-density-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const density = button.dataset.densityPreset as Density;
        element<HTMLSelectElement>("densitySelect").value = density;
        this.applyDensityPreset();
        this.captureSettings(false);
        this.scheduleTypographyPlan("statusPresetApplied");
      });
    });

    element<HTMLSelectElement>("densitySelect").addEventListener("change", () => {
      this.applyDensityPreset();
      this.captureSettings(false);
      this.scheduleTypographyPlan();
    });

    [
      "layoutSelect",
      "typographyModeSelect",
      "fontSizeInput",
      "phraseTargetInput",
      "maxTokensInput",
      "revealSelect",
      "preRollInput",
      "postRollInput",
      "pauseInput",
      "maxBlocksInput",
      "layerOrderSelect",
      "replacePreviousCheck",
      "autoLabelColorsCheck",
    ].forEach((id) => {
      element<HTMLElement>(id).addEventListener("change", () => {
        this.captureSettings(false);
        this.scheduleTypographyPlan();
      });
    });
  }

  private showView(name: string): void {
    document.querySelectorAll<HTMLElement>(".workspace").forEach((view) => {
      view.classList.toggle("active", view.id === `view-${name}`);
    });
    document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === name);
    });
    if (name === "fill") this.scheduleFillPreview();
    if (name === "fill") this.startFillSelectionWatch();
    else this.stopFillSelectionWatch();
    // 커팅 탭에서는 AE에서 레이어를 고르는 것만으로 바로 불러온다.
    if (name === "cut") this.startCutSelectionWatch();
    else this.stopCutSelectionWatch();
  }

  private currentView(): string {
    return document.querySelector<HTMLElement>(".workspace.active")?.id.replace(/^view-/, "") ?? "sync";
  }

  private t(key: string, vars: Record<string, string | number> = {}): string {
    const table = UI_TEXT[this.settings.uiLocale] ?? UI_TEXT.ko;
    let value = table[key] ?? UI_TEXT.ko[key] ?? key;
    Object.entries(vars).forEach(([name, replacement]) => {
      value = value.replace(new RegExp(`\\{${name}\\}`, "g"), String(replacement));
    });
    return value;
  }

  private setText(selector: string, key: string): void {
    const node = document.querySelector<HTMLElement>(selector);
    if (node) node.textContent = this.t(key);
  }

  private setLabelText(controlId: string, key: string): void {
    const control = document.getElementById(controlId);
    const label = control?.closest("label");
    const span = label?.querySelector<HTMLElement>("span");
    if (span) span.textContent = this.t(key);
  }

  private setLabelHint(controlId: string, key: string): void {
    const control = document.getElementById(controlId);
    const label = control?.closest("label");
    const hint = label?.querySelector<HTMLElement>("small");
    if (hint) hint.textContent = this.t(key);
  }

  private setToggleText(controlId: string, titleKey: string, hintKey: string): void {
    const control = document.getElementById(controlId);
    const label = control?.closest("label");
    const title = label?.querySelector<HTMLElement>("b");
    const hint = label?.querySelector<HTMLElement>("small");
    if (title) title.textContent = this.t(titleKey);
    if (hint) hint.textContent = this.t(hintKey);
  }

  private setOptionText(selectId: string, value: string, key: string): void {
    const option = element<HTMLSelectElement>(selectId).querySelector<HTMLOptionElement>(`option[value="${value}"]`);
    if (option) option.textContent = this.t(key);
  }

  private applyLocaleToUI(): void {
    document.documentElement.lang = this.settings.uiLocale;
    element<HTMLButtonElement>("openSettingsBtn").setAttribute("aria-label", this.t("settingsAria"));
    element<HTMLButtonElement>("refreshCompBtn").title = this.t("refreshTitle");
    element<HTMLTextAreaElement>("lyricsInput").placeholder = this.t("lyricsPlaceholder");
    this.setText('.mode-tabs [data-view="sync"]', "tabSync");
    this.setText('.mode-tabs [data-view="fill"]', "tabFill");
    this.setText('.mode-tabs [data-view="type"]', "tabType");
    this.setText('.mode-tabs [data-view="cut"]', "tabCut");
    document.querySelectorAll<HTMLButtonElement>(".mode-tabs button").forEach((button, index) => {
      const label = button.textContent ?? "";
      button.innerHTML = `<span>${String(index + 1).padStart(2, "0")}</span>${escapeHtml(label.replace(/^\d+/, ""))}`;
    });
    this.setText("#view-sync .section-heading h1", "syncTitle");
    this.setText("#view-sync .section-heading p", "syncIntro");
    this.setText("#loadJsonBtn", "loadJson");
    this.setLabelText("translationLangSelect", "translationLangLabel");
    this.setLabelText("pronScriptSelect", "pronScriptLabel");
    this.setOptionText("pronScriptSelect", "auto", "pronScriptAuto");
    this.setOptionText("pronScriptSelect", "hangul", "pronScriptHangul");
    this.setText(".divider span", "localDivider");
    this.setText('label[for="lyricsInput"]', "lyricsLabel");
    this.setText("#checkEngineBtn", "engineCheck");
    this.setText("#runSyncBtn", "runSync");
    this.setText("#cancelSyncBtn", "cancel");
    this.setText("#environmentReport span", "envHint");
    this.setText(".progress-head small", "progressRecent");
    this.setText("#view-sync > .microcopy:not(.sync-store-note)", "syncMicro");
    const storeNote = document.querySelector<HTMLElement>(".sync-store-note");
    if (storeNote) {
      storeNote.childNodes[0]!.textContent = `${this.t("syncStoreNote")} `;
    }
    this.setText("#forgetSyncBtn", "forgetSync");
    this.setText("#view-fill .section-heading h1", "fillTitle");
    this.setText("#view-fill .section-heading p", "fillIntro");
    this.setText(".principle-card span", "preserve");
    this.setText(".principle-card small", "preserveText");
    this.setText("#previewFillBtn", "previewFill");
    this.setText("#applyFillBtn", "applyText");
    this.setText("#addLineMarkersBtn", "addMarkers");
    this.setText("#removeMarkersBtn", "removeMarkers");
    this.setText("#view-type .section-heading h1", "typeTitle");
    this.setText("#view-type .section-heading p", "typeIntro");
    this.setText(".intent-panel .tuning-head .eyebrow", "rhythm");
    this.setText(".intent-panel .tuning-head small", "rhythmHint");
    this.setText('[data-density-preset="readable"] span', "readableDesc");
    this.setText('[data-density-preset="readable"] small', "readableTag");
    this.setText('[data-density-preset="balanced"] span', "balancedDesc");
    this.setText('[data-density-preset="balanced"] small', "balancedTag");
    this.setText('[data-density-preset="rhythmic"] span', "rhythmicDesc");
    this.setText('[data-density-preset="rhythmic"] small', "rhythmicTag");
    this.setText(".tuning-panel > .tuning-head .eyebrow", "composition");
    this.setLabelText("typographyModeSelect", "generationMode");
    this.setOptionText("typographyModeSelect", "designed", "designedMode");
    this.setOptionText("typographyModeSelect", "line", "lineMode");
    this.setLabelText("phraseTargetInput", "blockChars");
    this.setLabelText("revealSelect", "reveal");
    this.setLabelText("layoutSelect", "layout");
    this.setLabelText("fontSizeInput", "baseSize");
    this.setText("#advancedTypeControls summary", "advanced");
    this.setLabelText("maxTokensInput", "wordsPerBlock");
    this.setLabelText("pauseInput", "pauseCut");
    this.setLabelText("preRollInput", "preRoll");
    this.setLabelText("postRollInput", "postRoll");
    this.setLabelText("maxBlocksInput", "blocksPerCard");
    this.setText("#previewTypeBtn", this.showCardDetails ? "hideCards" : "showCards");
    this.setText("#generateTypeBtn", "generateLayers");
    this.setText(".cleanup-panel summary", "cleanup");
    this.setText(".cleanup-panel p", "cleanupHint");
    this.setText("#removeGeneratedLayersBtn", "removeLayers");
    this.setText("#removeTypeMarkersBtn", "removeTypeMarkers");
    this.setText("#view-type .microcopy", "typeMicro");
    this.setText("#view-cut .section-heading h1", "cutTitle");
    this.setText("#view-cut .section-heading p", "cutIntro");
    this.setText("#loadCutLayerBtn", "loadCutLayer");
    this.setText("#applyCutBtn", "applyCut");
    this.setLabelText("cutRevealSelect", "cutRevealLabel");
    this.setOptionText("cutRevealSelect", "cumulative", "cutRevealCumulative");
    this.setOptionText("cutRevealSelect", "sequential", "cutRevealSequential");
    this.setToggleText("keepCutPositionCheck", "keepCutPosition", "keepCutPositionHint");
    this.setText("#view-cut .microcopy", "cutMicro");
    if (this.cutSession) this.renderCut();
    this.setText(".drawer-head h2", "preferences");
    this.setLabelHint("uiLocaleSelect", "uiLanguageHint");
    this.setLabelHint("pythonPathInput", "pythonHint");
    this.setLabelHint("layerOrderSelect", "layerOrderHint");
    this.setToggleText("autoLabelColorsCheck", "autoLabelColors", "autoLabelColorsHint");
    this.setToggleText("replacePreviousCheck", "replacePrevious", "replacePreviousHint");
    this.setText("#saveSettingsBtn", "saveSettings");
    this.setText("#installEngineBtn", this.engineInstallMode === "update" ? "engineUpdate" : "engineInstall");
  }

  private setStatus(kind: "ready" | "busy" | "success" | "error", message: string): void {
    const bar = element<HTMLElement>("statusBar");
    bar.dataset.kind = kind;
    element<HTMLElement>("statusText").textContent = message;
    this.addProgress(kind, message);
  }

  private statusKey(
    kind: "ready" | "busy" | "success" | "error",
    key: string,
    vars: Record<string, string | number> = {},
  ): void {
    this.setStatus(kind, this.t(key, vars));
  }

  private addProgress(kind: string, message: string): void {
    const now = new Date();
    this.progressLines.unshift({
      kind,
      message,
      time: now.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    });
    this.progressLines = this.progressLines.slice(0, 18);
    const panel = document.getElementById("progressLog");
    if (!panel) return;
    panel.innerHTML = this.progressLines.map((entry) => `
      <div class="progress-entry" data-kind="${escapeHtml(entry.kind)}">
        <span>${escapeHtml(entry.time)}</span><b>${escapeHtml(entry.message)}</b>
      </div>
    `).join("");
  }

  private setBusy(value: boolean, message?: string): void {
    this.busy = value;
    document.body.classList.toggle("is-busy", value);
    document.querySelectorAll<HTMLButtonElement>("button").forEach((button) => {
      if (button.id !== "cancelSyncBtn") button.disabled = value;
    });
    if (!value) {
      element<HTMLButtonElement>("applyFillBtn").disabled =
        this.fillAssignments.filter((assignment) => !assignment.skippedReason && assignment.text).length === 0;
      element<HTMLButtonElement>("generateTypeBtn").disabled =
        !this.typographyPlan || this.typographyPlan.blocks.length === 0;
    }
    element<HTMLButtonElement>("cancelSyncBtn").hidden = !value || !this.abortController;
    if (message) this.setStatus(value ? "busy" : "ready", message);
  }

  private setBusyKey(
    value: boolean,
    key?: string,
    vars: Record<string, string | number> = {},
  ): void {
    this.setBusy(value, key ? this.t(key, vars) : undefined);
  }

  private applySettingsToUI(): void {
    element<HTMLSelectElement>("uiLocaleSelect").value = this.settings.uiLocale;
    element<HTMLInputElement>("pythonPathInput").value = this.settings.pythonPath;
    element<HTMLInputElement>("runtimeRootInput").value = this.settings.runtimeRoot;
    element<HTMLInputElement>("modelDirInput").value = this.settings.modelDir;
    element<HTMLSelectElement>("minDepthSelect").value = this.settings.minDepth;
    element<HTMLSelectElement>("languageSelect").value = this.settings.language;
    element<HTMLSelectElement>("densitySelect").value = this.settings.density;
    element<HTMLSelectElement>("typographyModeSelect").value = this.settings.typographyMode;
    element<HTMLSelectElement>("layoutSelect").value = this.settings.layout;
    element<HTMLInputElement>("fontSizeInput").value = String(this.settings.fontSize);
    element<HTMLInputElement>("preRollInput").value = String(this.settings.preRollFrames);
    element<HTMLInputElement>("postRollInput").value = String(this.settings.postRollFrames);
    element<HTMLInputElement>("pauseInput").value = String(this.settings.pauseThreshold);
    element<HTMLInputElement>("maxBlocksInput").value = String(this.settings.maxBlocksPerCard);
    element<HTMLInputElement>("phraseTargetInput").value = String(this.settings.phraseTargetChars);
    element<HTMLInputElement>("maxTokensInput").value = String(this.settings.maxTokensPerBlock);
    element<HTMLSelectElement>("revealSelect").value = this.settings.revealMode;
    element<HTMLSelectElement>("layerOrderSelect").value = this.settings.layerOrder;
    element<HTMLInputElement>("replacePreviousCheck").checked = this.settings.replacePrevious;
    element<HTMLInputElement>("autoLabelColorsCheck").checked = this.settings.autoLabelColors;
    element<HTMLInputElement>("keepCutPositionCheck").checked = this.settings.keepCutPosition;
    element<HTMLSelectElement>("cutRevealSelect").value = this.settings.cutReveal;
    element<HTMLSelectElement>("translationLangSelect").value = this.settings.translationLanguage;
    element<HTMLSelectElement>("pronScriptSelect").value = this.settings.pronunciationScript;
    this.renderPresetState();
    this.applyLocaleToUI();
  }

  private applyDensityPreset(): void {
    const density = element<HTMLSelectElement>("densitySelect").value as AppSettings["density"];
    const preset = DENSITY_PRESETS[density];
    element<HTMLInputElement>("phraseTargetInput").value = String(preset.target);
    element<HTMLInputElement>("maxTokensInput").value = String(preset.tokens);
    element<HTMLInputElement>("pauseInput").value = String(preset.pause);
    this.renderPresetState();
  }

  private captureSettings(close = true): void {
    this.settings = {
      pythonPath: element<HTMLInputElement>("pythonPathInput").value.trim() || "python",
      uiLocale: element<HTMLSelectElement>("uiLocaleSelect").value as UiLocale,
      runtimeRoot: element<HTMLInputElement>("runtimeRootInput").value.trim(),
      modelDir: element<HTMLInputElement>("modelDirInput").value.trim(),
      minDepth: element<HTMLSelectElement>("minDepthSelect").value as AppSettings["minDepth"],
      language: element<HTMLSelectElement>("languageSelect").value,
      density: element<HTMLSelectElement>("densitySelect").value as AppSettings["density"],
      typographyMode: element<HTMLSelectElement>("typographyModeSelect").value as AppSettings["typographyMode"],
      layout: element<HTMLSelectElement>("layoutSelect").value as AppSettings["layout"],
      fontSize: Math.max(12, Math.min(400, Number(element<HTMLInputElement>("fontSizeInput").value) || 94)),
      preRollFrames: Math.max(0, Math.min(60, Number(element<HTMLInputElement>("preRollInput").value) || 0)),
      postRollFrames: Math.max(0, Math.min(120, Number(element<HTMLInputElement>("postRollInput").value) || 0)),
      pauseThreshold: Math.max(0.05, Math.min(2, Number(element<HTMLInputElement>("pauseInput").value) || 0.32)),
      maxBlocksPerCard: Math.max(1, Math.min(6, Number(element<HTMLInputElement>("maxBlocksInput").value) || 4)),
      phraseTargetChars: Math.max(3, Math.min(24, Number(element<HTMLInputElement>("phraseTargetInput").value) || 9)),
      maxTokensPerBlock: Math.max(1, Math.min(8, Number(element<HTMLInputElement>("maxTokensInput").value) || 4)),
      revealMode: element<HTMLSelectElement>("revealSelect").value as AppSettings["revealMode"],
      layerOrder: element<HTMLSelectElement>("layerOrderSelect").value as AppSettings["layerOrder"],
      replacePrevious: element<HTMLInputElement>("replacePreviousCheck").checked,
      autoLabelColors: element<HTMLInputElement>("autoLabelColorsCheck").checked,
      keepCutPosition: element<HTMLInputElement>("keepCutPositionCheck").checked,
      cutReveal: element<HTMLSelectElement>("cutRevealSelect").value as AppSettings["cutReveal"],
      translationLanguage: element<HTMLSelectElement>("translationLangSelect").value as AppSettings["translationLanguage"],
      pronunciationScript: element<HTMLSelectElement>("pronScriptSelect").value as AppSettings["pronunciationScript"],
    };
    this.saveSettings();
    if (close) {
      this.toggleSettings(false);
      this.statusKey("success", "statusSettingsSaved");
    }
  }

  private toggleSettings(show: boolean): void {
    element<HTMLElement>("settingsDrawer").classList.toggle("open", show);
  }

  private async refreshComp(preserveStatus = false): Promise<void> {
    try {
      this.comp = await evalHost<CompInfo>("everyricGetCompInfo");
      this.renderComp();
      // 컴포지션이 바뀌었으면 그 컴포지션에 매어 둔 싱크로 갈아탄다. 같은 컴포지션을 보고
      // 있는 동안에는 손대지 않는다 — 방금 불러온 것을 덮어쓰면 안 된다.
      const key = compKey(this.comp.projectPath, this.comp.compId);
      let restored = false;
      if (key !== this.compStorageKey) {
        this.compStorageKey = key;
        restored = this.adoptSyncForCurrentComp();
      }
      if (preserveStatus) {
        return;
      }
      if (!this.comp.hasComp) {
        this.statusKey("ready", "statusNeedActiveComp");
      } else if (restored) {
        // 복원했다는 사실이 컴포지션 안내보다 알릴 값어치가 크다 — 덮어쓰지 않는다.
      } else {
        this.statusKey("ready", "statusCompReady", {
          name: this.comp.name ?? "Untitled",
          fps: this.comp.frameRate?.toFixed(2) ?? "?",
        });
      }
      this.scheduleTypographyPlan();
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    }
  }

  /**
   * 지금 컴포지션에 저장된 싱크를 꺼내 온다. 없으면 비운다 — 다른 컴포지션의 가사를
   * 그대로 들고 있으면 엉뚱한 곳에 텍스트를 채우게 된다.
   */
  private adoptSyncForCurrentComp(): boolean {
    if (!this.comp?.hasComp) return false;
    const restored = loadSyncForComp(this.comp.projectPath, this.comp.compId);
    if (restored) {
      this.setSyncDocument(restored, { persist: false, restored: true });
      return true;
    }
    if (!this.syncDocument) return false;
    this.syncDocument = null;
    this.typographyPlan = null;
    this.fillAssignments = [];
    this.cutSession = null;
    this.cutPoints = [];
    this.lastCutSelectionSignature = "";
    element<HTMLElement>("dataBadge").textContent = "NO DATA";
    element<HTMLElement>("dataBadge").classList.remove("ready");
    this.renderEmptyState();
    this.renderCut();
    return false;
  }

  private renderComp(): void {
    const card = element<HTMLElement>("compCard");
    if (!this.comp?.hasComp) {
      card.innerHTML = `<span class="eyebrow">ACTIVE COMP</span><strong>${escapeHtml(this.t("noComp"))}</strong><small>${escapeHtml(this.t("noCompHint"))}</small>`;
      return;
    }
    const selected = this.comp.selectedTextLayers?.length ?? 0;
    const audio = this.comp.audioLayers?.[0];
    card.innerHTML = `
      <span class="eyebrow">ACTIVE COMP</span>
      <strong>${escapeHtml(this.comp.name ?? "Untitled")}</strong>
      <small>${this.comp.width}×${this.comp.height} · ${timeLabel(this.comp.duration ?? 0)} · ${escapeHtml(this.t("selectedText", { count: selected }))}</small>
      <div class="comp-audio ${audio ? "found" : ""}">${audio ? `${escapeHtml(this.t("audio"))} · ${escapeHtml(audio.name)}` : escapeHtml(this.t("noAudio"))}</div>
    `;
  }

  private renderEmptyState(): void {
    element<HTMLElement>("syncSummary").innerHTML = `<div class="empty-state">${escapeHtml(this.t("syncEmpty"))}</div>`;
    element<HTMLElement>("fillPreview").innerHTML = `<div class="empty-state">${escapeHtml(this.t("fillEmpty"))}</div>`;
    element<HTMLElement>("typePreview").innerHTML = `<div class="empty-state">${escapeHtml(this.t("typeEmpty"))}</div>`;
    element<HTMLElement>("typeDataState").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("typeNeedsData"))}</div>`;
    element<HTMLElement>("splitSample").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("splitSampleEmpty"))}</div>`;
    element<HTMLElement>("typePlanSummary").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("planWaiting"))}</div>`;
    element<HTMLButtonElement>("applyFillBtn").disabled = true;
    element<HTMLButtonElement>("generateTypeBtn").disabled = true;
  }

  private setSyncDocument(
    documentValue: SyncDocument,
    options: { persist?: boolean; restored?: boolean } = {},
  ): void {
    this.syncDocument = documentValue;
    this.typographyPlan = null;
    this.fillAssignments = [];
    const summary = element<HTMLElement>("syncSummary");
    const preview = documentValue.lines.slice(0, 6).map((line, index) => `
      <div class="sync-line">
        <span>${String(index + 1).padStart(2, "0")}</span>
        <div><strong>${escapeHtml(line.text)}</strong><small>${timeLabel(line.start)} — ${timeLabel(line.end)} · atom ${line.atoms.length}</small></div>
      </div>
    `).join("");
    summary.innerHTML = `
      <div class="summary-strip">
        <div><b>${documentValue.lines.length}</b><span>LINES</span></div>
        <div><b>${documentValue.lines.reduce((sum, line) => sum + line.atoms.length, 0)}</b><span>ATOMS</span></div>
        <div><b>${timeLabel(documentValue.duration)}</b><span>END</span></div>
      </div>
      <div class="sync-list">${preview}</div>
      ${documentValue.lines.length > 6 ? `<small class="more-note">외 ${documentValue.lines.length - 6}개 구간</small>` : ""}
    `;
    element<HTMLElement>("dataBadge").textContent = documentValue.sourceLabel;
    element<HTMLElement>("dataBadge").classList.add("ready");
    // 이 컴포지션에 매어 둔다. 패널을 닫았다 열어도 다시 불러오지 않아도 되게.
    if (options.persist !== false) {
      saveSyncForComp(this.comp?.projectPath, this.comp?.compId, documentValue, this.comp?.name);
    }
    if (options.restored) {
      this.statusKey("ready", "statusSyncRestored", { count: documentValue.lines.length });
    } else {
      this.statusKey("success", "statusSyncReady", { count: documentValue.lines.length });
    }
    this.scheduleTypographyPlan();
    if (this.currentView() === "fill") this.scheduleFillPreview();
  }

  private renderPresetState(): void {
    const density = element<HTMLSelectElement>("densitySelect").value as Density;
    const preset = DENSITY_PRESETS[density];
    const modified =
      Number(element<HTMLInputElement>("phraseTargetInput").value) !== preset.target ||
      Number(element<HTMLInputElement>("maxTokensInput").value) !== preset.tokens ||
      Number(element<HTMLInputElement>("pauseInput").value) !== preset.pause;
    document.querySelectorAll<HTMLButtonElement>("[data-density-preset]").forEach((button) => {
      button.classList.toggle("active", button.dataset.densityPreset === density);
    });
    element<HTMLElement>("presetStateText").textContent = `${preset.label}${modified ? ` · ${this.t("presetModified")}` : ""}`;
  }

  private scheduleTypographyPlan(statusKey?: string): void {
    this.renderPresetState();
    this.generationConfirmOpen = false;
    element<HTMLElement>("generationConfirm").hidden = true;
    if (this.planTimer !== null) window.clearTimeout(this.planTimer);
    this.planTimer = window.setTimeout(() => {
      this.planTimer = null;
      this.recalculateTypographyPlan(statusKey);
    }, 300);
  }

  private recalculateTypographyPlan(statusKey?: string): void {
    if (!this.syncDocument) {
      this.typographyPlan = null;
      this.renderEmptyTypePlan();
      return;
    }
    if (!this.comp?.hasComp) {
      this.renderEmptyTypePlan(this.t("statusNeedActiveComp"));
      return;
    }
    try {
      this.typographyPlan = this.createTypographyPlan(this.syncDocument);
      this.renderTypographyUx();
      element<HTMLButtonElement>("generateTypeBtn").disabled = this.typographyPlan.blocks.length === 0;
      if (statusKey) this.statusKey("ready", statusKey);
    } catch (error) {
      this.typographyPlan = null;
      element<HTMLButtonElement>("generateTypeBtn").disabled = true;
      element<HTMLElement>("typePlanSummary").innerHTML = `<div class="warning">${escapeHtml(errorMessage(error))}</div>`;
    }
  }

  private renderEmptyTypePlan(message = this.t("typeNeedsData")): void {
    element<HTMLElement>("typeDataState").innerHTML = `<div class="empty-state compact">${escapeHtml(message)}</div>`;
    element<HTMLElement>("splitSample").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("splitSampleEmpty"))}</div>`;
    element<HTMLElement>("typePlanSummary").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("planWaiting"))}</div>`;
    element<HTMLElement>("typePreview").innerHTML = `<div class="empty-state">${escapeHtml(this.t("typeEmpty"))}</div>`;
    element<HTMLButtonElement>("generateTypeBtn").disabled = true;
  }

  private representativeLines(): SyncLine[] {
    if (!this.syncDocument) return [];
    const lines = [...this.syncDocument.lines].filter((line) => line.text.trim());
    if (lines.length <= 2) return lines;
    const byLength = [...lines].sort((a, b) => visibleLength(a.text) - visibleLength(b.text));
    const median = byLength[Math.floor(byLength.length / 2)];
    const longest = byLength[byLength.length - 1];
    return [median, longest].filter((line, index, array): line is SyncLine => Boolean(line) && array.indexOf(line) === index);
  }

  private planForLine(line: SyncLine): TypographyPlan {
    const documentValue: SyncDocument = {
      lines: [line],
      language: this.syncDocument?.language ?? "auto",
      sourceLabel: "sample",
      duration: line.end,
    };
    return this.createTypographyPlan(documentValue, "SAMPLE");
  }

  private createTypographyPlan(documentValue: SyncDocument, groupId?: string): TypographyPlan {
    const options = this.plannerOptions();
    return this.settings.typographyMode === "line"
      ? planLineLyrics(documentValue, options, groupId)
      : planTypography(documentValue, options, groupId);
  }

  private renderSplitSample(): void {
    const lines = this.representativeLines();
    if (!lines.length) {
      element<HTMLElement>("splitSample").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("splitSampleEmpty"))}</div>`;
      return;
    }
    element<HTMLElement>("splitSample").innerHTML = `
      <div class="sample-head"><span class="eyebrow">${escapeHtml(this.t("sampleTitle"))}</span><small>${escapeHtml(this.t("sampleHint"))}</small></div>
      ${lines.map((line) => {
        const plan = this.planForLine(line);
        const chips = plan.blocks.map((block) => `<span>${escapeHtml(block.text)}</span>`).join("");
        return `<div class="sample-line"><p>${escapeHtml(line.text)}</p><div class="block-chips">${chips}</div></div>`;
      }).join("")}
    `;
  }

  private analyzePlan(plan: TypographyPlan): PlanAnalysis {
    let totalChars = 0;
    let totalSeconds = 0;
    let shortestDuration = Number.POSITIVE_INFINITY;
    const risks: RiskItem[] = [];
    for (const block of plan.blocks) {
      const chars = visibleLength(block.text);
      const duration = Math.max(0.001, block.end - block.start);
      const charsPerSecond = chars / duration;
      totalChars += chars;
      totalSeconds += duration;
      shortestDuration = Math.min(shortestDuration, duration);
      const item = this.blockRisk(block, duration, charsPerSecond);
      if (item) risks.push(item);
    }
    return {
      cards: plan.cards.length,
      layers: plan.blocks.length,
      averageCharsPerSecond: totalChars / Math.max(0.001, totalSeconds),
      shortestDuration: Number.isFinite(shortestDuration) ? shortestDuration : 0,
      warnings: risks.filter((item) => item.level === "warn"),
      dangers: risks.filter((item) => item.level === "danger"),
    };
  }

  private blockRisk(block: TypographyBlock, duration: number, charsPerSecond: number): RiskItem | null {
    const durationDanger = duration < 0.4;
    const speedDanger = charsPerSecond > 10;
    const durationWarn = duration < 0.8;
    const speedWarn = charsPerSecond > 7;
    if (!durationDanger && !speedDanger && !durationWarn && !speedWarn) return null;
    const durationReason = this.t("durationReason", { seconds: duration.toFixed(2) });
    const speedReason = this.t("speedReason", { speed: charsPerSecond.toFixed(1) });
    const reason = durationDanger || durationWarn ? durationReason : speedReason;
    return {
      level: durationDanger || speedDanger ? "danger" : "warn",
      cardId: block.cardId,
      blockId: block.id,
      text: block.text,
      duration,
      charsPerSecond,
      reason,
      advice: durationDanger || durationWarn ? this.t("durationAdvice") : this.t("speedAdvice"),
    };
  }

  private renderTypographyUx(): void {
    if (!this.syncDocument || !this.typographyPlan) return;
    this.renderSplitSample();
    const atomCount = this.syncDocument.lines.reduce((sum, line) => sum + line.atoms.length, 0);
    element<HTMLElement>("typeDataState").innerHTML = `
      <div class="data-state-row"><b>${escapeHtml(this.syncDocument.sourceLabel)}</b><span>${this.syncDocument.lines.length}라인 · atom ${atomCount} · ${timeLabel(this.syncDocument.duration)}</span></div>
    `;
    const analysis = this.analyzePlan(this.typographyPlan);
    element<HTMLElement>("typePlanSummary").innerHTML = `
      <div class="summary-strip plan-metrics">
        <div><b>${analysis.cards}</b><span>CARDS</span></div>
        <div><b>${analysis.layers}</b><span>LAYERS</span></div>
        <div><b>${analysis.averageCharsPerSecond.toFixed(1)}</b><span>${escapeHtml(this.t("charsPerSecondMetric"))}</span></div>
      </div>
      <div class="risk-bar ${analysis.dangers.length ? "danger" : analysis.warnings.length ? "warn" : "ok"}">
        <b>${analysis.dangers.length ? `${this.t("danger")} ${analysis.dangers.length}` : analysis.warnings.length ? `${this.t("warning")} ${analysis.warnings.length}` : this.t("noRisk")}</b>
        <span>${escapeHtml(this.t("shortest"))} ${analysis.shortestDuration.toFixed(2)}s · ${escapeHtml(this.t("existingLayers"))} ${this.comp?.generatedLayerCount ?? 0}</span>
      </div>
      ${[...analysis.dangers, ...analysis.warnings].slice(0, 4).map((item) => `
        <div class="risk-item ${item.level}">
          <b>${item.level === "danger" ? this.t("danger") : this.t("warning")} · ${escapeHtml(item.cardId)} · ${escapeHtml(item.text)}</b>
          <span>${escapeHtml(item.reason)} · ${escapeHtml(item.advice)}</span>
        </div>
      `).join("")}
      ${this.typographyPlan.warnings.map((warning) => `<div class="warning">${escapeHtml(warning)}</div>`).join("")}
    `;
    this.renderTypographyPreview();
  }

  private async loadJson(): Promise<void> {
    try {
      const picked = await evalHost<{ ok: boolean; path?: string; error?: string }>("everyricPickFile", { kind: "json" });
      if (!picked.ok || !picked.path) {
        if (picked.error) this.statusKey("error", "statusParseError", { error: picked.error });
        return;
      }
      const payload = readJsonFile(picked.path);
      this.setSyncDocument(normalizeSyncPayload(payload, picked.path.split(/[\\/]/).pop() ?? "JSON"));
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    }
  }

  /**
   * 번역 언어를 바꾼다. 서버가 언어별 번역을 함께 줬으면 재조회 없이 갈아끼우고,
   * 없으면 다시 받아야 한다고 알린다(로컬 JSON으로 불러온 문서가 그렇다).
   */
  private switchTranslationLanguage(): void {
    const current = this.syncDocument;
    if (!current) return;
    const lang = this.settings.translationLanguage;
    if (current.translationLang === lang) return;
    const table = current.translationsByLang?.[lang];
    if (!table) {
      const available = selectableLanguages(current);
      this.statusKey("ready", "statusLangNeedsRefetch", {
        lang,
        available: available.length > 0 ? available.join(", ") : "-",
      });
      return;
    }
    this.setSyncDocument(withTranslationLanguage(current, lang));
    this.reloadCutSessionForScript();
    this.renderCut();
  }

  /** 표기·언어가 바뀌면 지금 보고 있는 커팅 세션의 발음도 다시 뽑는다. */
  private reloadCutSessionForScript(): void {
    const session = this.cutSession;
    if (!session) return;
    const layer: TextLayerInfo = {
      index: session.layerIndex,
      name: session.layerName,
      inPoint: session.inPoint,
      outPoint: session.outPoint,
      text: session.text,
      sourceTextKeys: 0,
      locked: false,
    };
    const rebuilt = buildCutSession(layer, this.syncDocument, resolveScript(this.settings));
    // 찍어 둔 컷은 글자 인덱스 기준이라 그대로 살린다 — 글자 수가 같으면 유효하다.
    if (rebuilt.chars.length === session.chars.length) {
      this.cutSession = rebuilt;
    } else {
      this.cutSession = rebuilt;
      this.cutPoints = [];
    }
  }

  private forgetStoredSync(): void {
    forgetSyncForComp(this.comp?.projectPath, this.comp?.compId);
    this.statusKey("ready", "statusSyncForgotten");
  }

  private async checkEngine(): Promise<void> {
    this.captureSettings(false);
    this.setBusyKey(true, "statusEnvChecking");
    try {
      const report = await inspectEnvironment(
        this.settings.pythonPath,
        this.settings.runtimeRoot || undefined,
        this.settings.modelDir || undefined,
      );
      this.renderEnvironmentReport(report);
      const vram = report.vramTotalMb
        ? this.t("vramSuffix", { gb: Math.round(report.vramTotalMb / 1024) })
        : "";
      this.statusKey("success", "statusEnvSuccess", { version: report.everyricVersion, vram });
      this.updateEngineSetupRow(report.everyricVersion);
    } catch (error) {
      this.statusKey("error", "statusEnvFailed", { error: errorMessage(error) });
      this.showEngineInstallOffer();
    } finally {
      this.setBusy(false);
    }
  }

  private renderEnvironmentReport(report: EnvironmentReport): void {
    const vram = report.vramTotalMb
      ? `${Math.round(report.vramFreeMb ?? 0)}MB free / ${Math.round(report.vramTotalMb)}MB total`
      : this.t("unknown");
    element<HTMLElement>("environmentReport").innerHTML = `
      <div class="env-grid">
        <div><span>EVERYRIC2</span><b>${escapeHtml(report.everyricVersion)}</b></div>
        <div><span>GPU</span><b>${escapeHtml(report.gpuName ?? "N/A")}</b></div>
        <div><span>VRAM</span><b>${escapeHtml(vram)}</b></div>
        <div><span>RAM</span><b>${report.systemMemoryGb}GB</b></div>
      </div>
      <p>${escapeHtml(this.t("envRecommendation", { min: report.recommended.minimumVramGb, recommended: report.recommended.comfortableVramGb }))}</p>
      ${report.notes.length ? `<ul>${report.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>` : ""}
    `;
  }

  private async checkForUpdates(): Promise<void> {
    this.latestManifest = await fetchLatestManifest();
    const update = panelUpdate(this.latestManifest);
    const badge = element<HTMLButtonElement>("updateBadge");
    if (update) {
      badge.hidden = false;
      badge.textContent = `v${update.version} ↗`;
      badge.title = this.t("updateAvailable", { version: update.version });
    } else {
      badge.hidden = true;
    }
  }

  private showEngineInstallOffer(): void {
    this.engineInstallMode = "install";
    element<HTMLElement>("environmentReport").innerHTML = `<span>${escapeHtml(this.t("engineMissingHint"))}</span>`;
    element<HTMLButtonElement>("installEngineBtn").textContent = this.t("engineInstall");
    element<HTMLElement>("engineSetupRow").hidden = false;
  }

  private updateEngineSetupRow(engineVersion: string): void {
    const row = element<HTMLElement>("engineSetupRow");
    if (parseVersion(engineVersion).length === 0) {
      row.hidden = true;
      return;
    }
    const latest = this.latestManifest?.engine?.version;
    const outOfRange = !satisfiesRange(engineVersion, SUPPORTED_ENGINE_RANGE);
    const updateAvailable = Boolean(latest && isNewerVersion(latest, engineVersion));
    if (!outOfRange && !updateAvailable) {
      row.hidden = true;
      return;
    }
    this.engineInstallMode = "update";
    element<HTMLButtonElement>("installEngineBtn").textContent = this.t("engineUpdate");
    row.hidden = false;
    if (outOfRange) {
      this.statusKey("ready", "statusEngineOutOfRange", { version: engineVersion, range: SUPPORTED_ENGINE_RANGE });
    } else if (latest) {
      this.statusKey("ready", "statusEngineUpdateAvailable", { version: latest });
    }
  }

  private async installEngineFlow(repair = false): Promise<void> {
    this.latestManifest = this.latestManifest ?? (await fetchLatestManifest());
    this.abortController = new AbortController();
    this.setBusyKey(true, "statusEngineInstalling", { message: "…" });
    try {
      const pythonPath = await installEngine({
        wheelUrl: this.latestManifest?.engine?.wheelUrl,
        onProgress: (message) => this.statusKey("busy", "statusEngineInstalling", { message }),
        signal: this.abortController.signal,
        extensionRoot: extensionRoot(),
        modelDir: this.settings.modelDir || undefined,
        repair,
      });
      this.settings.pythonPath = pythonPath;
      this.saveSettings();
      element<HTMLInputElement>("pythonPathInput").value = pythonPath;
      element<HTMLElement>("engineSetupRow").hidden = true;
      this.statusKey("success", "statusEngineInstalled");
    } catch (error) {
      this.statusKey("error", "statusEngineInstallFailed", { error: errorMessage(error) });
      return;
    } finally {
      this.abortController = null;
      this.setBusy(false);
    }
    await this.checkEngine();
  }

  private async runSync(): Promise<void> {
    if (!this.comp?.hasComp) await this.refreshComp();
    const audioPath = this.comp?.audioLayers?.find((layer) => layer.filePath)?.filePath;
    const lyrics = element<HTMLTextAreaElement>("lyricsInput").value.trim();
    if (!audioPath) {
      this.statusKey("error", "statusNoAudio");
      return;
    }
    if (!lyrics) {
      this.statusKey("error", "statusNeedLyrics");
      return;
    }
    this.captureSettings(false);
    this.abortController = new AbortController();
    this.setBusyKey(true, "statusSyncRunning");
    try {
      const payload = await runLocalSync({
        pythonPath: this.settings.pythonPath,
        language: this.settings.language,
        audioPath,
        lyrics,
        runtimeRoot: this.settings.runtimeRoot || undefined,
        modelDir: this.settings.modelDir || undefined,
        minDepth: this.settings.minDepth,
      }, (message) => this.statusKey("busy", "statusSyncProgress", { message }), this.abortController.signal);
      this.setSyncDocument(normalizeSyncPayload(payload, "로컬 Everyric2"));
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.abortController = null;
      this.setBusy(false);
    }
  }

  private cancelSync(): void {
    this.abortController?.abort();
    this.statusKey("busy", "statusCanceling");
  }

  private scheduleFillPreview(): void {
    if (this.fillPreviewTimer !== null) window.clearTimeout(this.fillPreviewTimer);
    this.fillPreviewTimer = window.setTimeout(() => {
      this.fillPreviewTimer = null;
      void this.previewFill({ automatic: true });
    }, 150);
  }

  private startFillSelectionWatch(): void {
    if (this.fillSelectionTimer !== null) return;
    this.fillSelectionTimer = window.setInterval(() => {
      if (this.busy || this.currentView() !== "fill" || !this.syncDocument) return;
      void this.refreshFillPreviewIfSelectionChanged();
    }, 700);
  }

  private stopFillSelectionWatch(): void {
    if (this.fillSelectionTimer !== null) {
      window.clearInterval(this.fillSelectionTimer);
      this.fillSelectionTimer = null;
    }
  }

  private fillSelectionSignature(layers: CompInfo["selectedTextLayers"] = []): string {
    return layers
      .map((layer) => [
        layer.index,
        layer.name,
        layer.inPoint.toFixed(3),
        layer.outPoint.toFixed(3),
        layer.sourceTextKeys,
        layer.locked ? 1 : 0,
      ].join(":"))
      .join("|");
  }

  private async refreshFillPreviewIfSelectionChanged(): Promise<void> {
    try {
      const response = await evalHost<{ ok: boolean; layers: CompInfo["selectedTextLayers"]; error?: string }>("everyricGetSelectedTextLayers");
      const layers = response.layers ?? [];
      const signature = this.fillSelectionSignature(layers);
      if (signature === this.lastFillSelectionSignature) return;
      this.lastFillSelectionSignature = signature;
      await this.previewFill({ automatic: true, layers, error: response.ok ? undefined : response.error });
    } catch (error) {
      this.fillAssignments = [];
      this.renderFillPreview();
      element<HTMLButtonElement>("applyFillBtn").disabled = true;
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    }
  }

  private async previewFill(options: { automatic?: boolean; layers?: CompInfo["selectedTextLayers"]; error?: string } = {}): Promise<void> {
    if (!this.syncDocument) {
      this.statusKey("error", "statusNeedSyncData");
      if (!options.automatic) this.showView("sync");
      return;
    }
    if (options.automatic) this.statusKey("busy", "statusAutoFillPreviewing");
    try {
      const response = options.layers
        ? { ok: !options.error, layers: options.layers, error: options.error }
        : await evalHost<{ ok: boolean; layers: CompInfo["selectedTextLayers"]; error?: string }>("everyricGetSelectedTextLayers");
      const layers = response.layers ?? [];
      this.lastFillSelectionSignature = this.fillSelectionSignature(layers);
      if (!response.ok || layers.length === 0) throw new Error(response.error || this.t("statusNoSelectedText"));
      this.fillAssignments = planLayerFill(this.syncDocument, layers, false);
      this.renderFillPreview();
      const writable = this.fillAssignments.filter((assignment) => !assignment.skippedReason && assignment.text).length;
      element<HTMLButtonElement>("applyFillBtn").disabled = writable === 0;
      this.statusKey("ready", "statusFillPreview", { count: writable });
    } catch (error) {
      this.fillAssignments = [];
      this.renderFillPreview();
      element<HTMLButtonElement>("applyFillBtn").disabled = true;
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    }
  }

  private renderFillPreview(): void {
    if (this.fillAssignments.length === 0) {
      element<HTMLElement>("fillPreview").innerHTML = `<div class="empty-state">${escapeHtml(this.t("fillEmpty"))}</div>`;
      return;
    }
    element<HTMLElement>("fillPreview").innerHTML = this.fillAssignments.map((assignment) => `
      <div class="assignment ${assignment.skippedReason ? "skipped" : ""}">
        <div class="assignment-head"><b>${escapeHtml(assignment.layerName)}</b><span>${timeLabel(assignment.inPoint)}–${timeLabel(assignment.outPoint)}</span></div>
        <p>${escapeHtml(assignment.text || this.t("assignedEmpty"))}</p>
        ${assignment.skippedReason ? `<small>${escapeHtml(assignment.skippedReason)}</small>` : ""}
      </div>
    `).join("");
  }

  private async applyFill(): Promise<void> {
    const assignments = this.fillAssignments.filter((assignment) => !assignment.skippedReason && assignment.text);
    if (assignments.length === 0) return;
    this.setBusyKey(true, "statusApplyingFill");
    try {
      const result = await evalHost<HostResult>("everyricApplyTextAssignments", { assignments });
      if (!result.ok) throw new Error(result.error || this.t("statusLayerApplyFailed"));
      this.statusKey("success", "statusFillApplied", { count: result.updated ?? 0 });
      await this.refreshComp(true);
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }

  private plannerOptions(): PlannerOptions {
    this.captureSettings(false);
    return {
      density: this.settings.density,
      layout: this.settings.layout,
      width: this.comp?.width ?? 1920,
      height: this.comp?.height ?? 1080,
      frameRate: this.comp?.frameRate ?? 30,
      fontSize: this.settings.fontSize,
      preRollFrames: this.settings.preRollFrames,
      postRollFrames: this.settings.postRollFrames,
      pauseThreshold: this.settings.pauseThreshold,
      maxBlocksPerCard: this.settings.maxBlocksPerCard,
      phraseTargetChars: this.settings.phraseTargetChars,
      maxTokensPerBlock: this.settings.maxTokensPerBlock,
      revealMode: this.settings.revealMode,
    };
  }

  private async previewTypography(): Promise<void> {
    this.recalculateTypographyPlan();
  }

  private renderTypographyPreview(): void {
    if (!this.typographyPlan) {
      element<HTMLElement>("typePreview").innerHTML = `<div class="empty-state">${escapeHtml(this.t("statusCardDetailsEmpty"))}</div>`;
      return;
    }
    if (!this.showCardDetails) {
      element<HTMLElement>("typePreview").innerHTML = `<div class="empty-state compact">${escapeHtml(this.t("cardHint"))}</div>`;
      element<HTMLButtonElement>("previewTypeBtn").textContent = this.t("showCards");
      return;
    }
    element<HTMLButtonElement>("previewTypeBtn").textContent = this.t("hideCards");
    element<HTMLElement>("typePreview").innerHTML = `
      <div class="summary-strip">
        <div><b>${this.typographyPlan.cards.length}</b><span>CARDS</span></div>
        <div><b>${this.typographyPlan.blocks.length}</b><span>LAYERS</span></div>
        <div><b>${this.settings.phraseTargetChars}자</b><span>PHRASE</span></div>
      </div>
      <div class="card-list">
        ${this.typographyPlan.cards.slice(0, 12).map((card) => `
          <div class="plan-card">
            <div class="plan-card-head"><b>${card.id}</b><span>${timeLabel(card.start)}–${timeLabel(card.end)}</span></div>
            <div class="block-chips">${card.blocks.map((block) => `<span>${escapeHtml(block.text)}</span>`).join("")}</div>
          </div>
        `).join("")}
      </div>
      ${this.typographyPlan.warnings.map((warning) => `<div class="warning">${escapeHtml(warning)}</div>`).join("")}
    `;
  }

  private toggleCardDetails(): void {
    this.showCardDetails = !this.showCardDetails;
    this.renderTypographyPreview();
  }

  private async generateTypography(): Promise<void> {
    if (!this.typographyPlan) this.recalculateTypographyPlan();
    if (!this.typographyPlan || this.typographyPlan.blocks.length === 0) return;
    if (!this.generationConfirmOpen) {
      this.renderGenerationConfirm();
      return;
    }
    this.captureSettings(false);
    this.setBusyKey(true, "statusGenerateRunning");
    try {
      const result = await evalHost<HostResult>("everyricCreateTypography", {
        plan: this.typographyPlan,
        replacePrevious: this.settings.replacePrevious,
        layerOrder: this.settings.layerOrder,
        autoLabelColors: this.settings.autoLabelColors,
      });
      if (!result.ok) throw new Error(result.error || this.t("statusGenerateFailed"));
      this.statusKey("success", "statusGenerateSuccess", {
        created: result.created ?? 0,
        removed: result.removed ?? 0,
      });
      this.generationConfirmOpen = false;
      element<HTMLElement>("generationConfirm").hidden = true;
      await this.refreshComp(true);
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }

  private renderGenerationConfirm(): void {
    if (!this.typographyPlan) return;
    this.generationConfirmOpen = true;
    const previous = this.comp?.generatedLayerCount ?? 0;
    const panel = element<HTMLElement>("generationConfirm");
    panel.hidden = false;
    panel.innerHTML = `
      <div class="confirm-card">
        <span class="eyebrow">${escapeHtml(this.t("confirmTitle"))}</span>
        <h3>${escapeHtml(this.t("confirmCreate", { count: this.typographyPlan.blocks.length }))}</h3>
        <p>${escapeHtml(this.t("previousGenerated", { count: previous }))}</p>
        <label class="toggle-row compact"><input id="generationReplaceCheck" type="checkbox" ${this.settings.replacePrevious ? "checked" : ""}><span><b>${escapeHtml(this.t("replace"))}</b><small>${escapeHtml(this.t("replaceHint"))}</small></span></label>
        <p class="confirm-safe">${escapeHtml(this.t("confirmSafe"))}</p>
        <div class="action-row two">
          <button id="cancelGenerateConfirmBtn" class="button ghost" type="button">${escapeHtml(this.t("back"))}</button>
          <button id="runGenerateConfirmBtn" class="button primary" type="button">${escapeHtml(this.t("runGenerate"))}</button>
        </div>
      </div>
    `;
    element<HTMLInputElement>("generationReplaceCheck").addEventListener("change", () => {
      this.settings.replacePrevious = element<HTMLInputElement>("generationReplaceCheck").checked;
      element<HTMLInputElement>("replacePreviousCheck").checked = this.settings.replacePrevious;
      this.saveSettings();
    });
    element<HTMLButtonElement>("cancelGenerateConfirmBtn").addEventListener("click", () => {
      this.generationConfirmOpen = false;
      panel.hidden = true;
    });
    element<HTMLButtonElement>("runGenerateConfirmBtn").addEventListener("click", () => void this.generateTypography());
    this.statusKey("ready", "statusConfirmGenerate", { count: this.typographyPlan.blocks.length });
  }

  private async addLineMarkers(): Promise<void> {
    if (!this.syncDocument) {
      this.statusKey("error", "statusNeedSyncData");
      this.showView("sync");
      return;
    }
    try {
      this.setBusyKey(true, "statusMarkersAdding");
      const result = await evalHost<HostResult>("everyricCreateLineMarkers", { document: this.syncDocument });
      if (!result.ok) throw new Error(result.error || this.t("statusMarkersFailed"));
      this.statusKey("success", "statusMarkersAdded", { count: result.created ?? 0 });
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }

  private async removeMarkers(): Promise<void> {
    try {
      this.setBusyKey(true, "statusMarkersRemoving");
      const result = await evalHost<HostResult>("everyricRemoveGeneratedMarkers");
      if (!result.ok) throw new Error(result.error || this.t("statusMarkersRemoveFailed"));
      this.statusKey("success", "statusMarkersRemoved", { count: result.removed ?? 0 });
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }

  private async removeGeneratedLayers(): Promise<void> {
    try {
      this.setBusyKey(true, "statusLayersRemoving");
      const result = await evalHost<HostResult>("everyricRemoveGeneratedLayers");
      if (!result.ok) throw new Error(result.error || this.t("statusLayersRemoveFailed"));
      this.statusKey("success", "statusLayersRemoved", { count: result.removed ?? 0 });
      await this.refreshComp(true);
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }

  /** 커팅 탭에 있는 동안 AE의 선택을 지켜본다. 레이어를 고르면 그것으로 바로 갈아탄다. */
  private startCutSelectionWatch(): void {
    if (this.cutSelectionTimer !== null) return;
    // 들어오자마자 한 번, 그 뒤로는 주기적으로.
    void this.refreshCutIfSelectionChanged();
    this.cutSelectionTimer = window.setInterval(() => {
      if (this.busy || this.currentView() !== "cut") return;
      void this.refreshCutIfSelectionChanged();
    }, 700);
  }

  private stopCutSelectionWatch(): void {
    if (this.cutSelectionTimer !== null) {
      window.clearInterval(this.cutSelectionTimer);
      this.cutSelectionTimer = null;
    }
  }

  /** 레이어를 특정하는 지문. 텍스트까지 넣어야 내용이 바뀐 것도 잡힌다. */
  private cutSelectionSignature(layer: TextLayerInfo | undefined): string {
    if (!layer) return "";
    return [layer.index, layer.text, layer.inPoint.toFixed(3), layer.outPoint.toFixed(3), layer.sourceTextKeys, layer.locked ? 1 : 0].join(":");
  }

  private async refreshCutIfSelectionChanged(): Promise<void> {
    try {
      const response = await evalHost<{ ok: boolean; layers?: TextLayerInfo[]; error?: string }>(
        "everyricGetSelectedTextLayers",
      );
      const layer = (response.layers ?? [])[0];
      const signature = this.cutSelectionSignature(layer);
      // 같은 레이어를 계속 보고 있으면 찍어 둔 컷을 건드리지 않는다.
      if (signature === this.lastCutSelectionSignature) return;
      this.lastCutSelectionSignature = signature;
      if (!layer) {
        this.cutSession = null;
        this.cutPoints = [];
        this.renderCut();
        return;
      }
      this.cutSession = buildCutSession(layer, this.syncDocument, resolveScript(this.settings));
      this.cutPoints = [];
      this.renderCut();
      this.statusKey("ready", "statusCutLoaded", { name: layer.name, count: this.cutSession.chars.length });
    } catch {
      // 선택을 못 읽는 것은 흔한 일(컴포지션 전환 중 등)이라 조용히 넘긴다.
    }
  }

  private async loadCutLayer(options: { quiet?: boolean } = {}): Promise<void> {
    try {
      const response = await evalHost<{ ok: boolean; layers?: TextLayerInfo[]; error?: string }>(
        "everyricGetSelectedTextLayers",
      );
      const layer = (response.layers ?? [])[0];
      if (!response.ok || !layer) throw new Error(response.error || this.t("statusCutNoSelection"));
      this.cutSession = buildCutSession(layer, this.syncDocument, resolveScript(this.settings));
      this.cutPoints = [];
      this.lastCutSelectionSignature = this.cutSelectionSignature(layer);
      this.renderCut();
      this.statusKey("ready", "statusCutLoaded", {
        name: layer.name,
        count: this.cutSession.chars.length,
      });
    } catch (error) {
      this.cutSession = null;
      this.cutPoints = [];
      this.renderCut();
      // 뷰에 들어오자마자 자동으로 부를 때는 선택이 없는 것이 정상이라 조용히 넘긴다.
      if (!options.quiet) this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    }
  }

  private matchQualityKey(session: CutSession): string {
    if (session.matchQuality === "exact") return "cutMatchExact";
    if (session.matchQuality === "substring") return "cutMatchSubstring";
    if (session.matchQuality === "time") return "cutMatchTime";
    return "cutMatchNone";
  }

  private renderCut(): void {
    const stage = element<HTMLElement>("cutStage");
    const piecesNode = element<HTMLElement>("cutPieces");
    const applyButton = element<HTMLButtonElement>("applyCutBtn");
    const session = this.cutSession;

    if (!session) {
      stage.innerHTML = `<div class="empty-state">${escapeHtml(this.t("cutEmpty"))}</div>`;
      piecesNode.innerHTML = "";
      applyButton.disabled = true;
      return;
    }

    const cutIndexes = new Set(this.cutPoints.map((cut) => cut.index));
    const glyphs = session.chars
      .map((entry, index) => {
        const caret =
          index === 0
            ? ""
            : `<button class="cut-caret${cutIndexes.has(index) ? " cut" : ""}" data-caret="${index}" type="button"></button>`;
        const classes = ["cut-char"];
        if (!entry.visible) classes.push("space");
        if (entry.interpolated) classes.push("estimated");
        const glyph = entry.visible ? escapeHtml(entry.char) : "␣";
        return `${caret}<span class="${classes.join(" ")}"><b>${glyph}</b><em>${entry.start.toFixed(2)}</em></span>`;
      })
      .join("");

    const meta: string[] = [
      `<span class="cut-quality ${session.matchQuality}">${escapeHtml(this.t(this.matchQualityKey(session)))}</span>`,
    ];
    if (session.pronunciation) {
      meta.push(`<span class="cut-aux"><i>${escapeHtml(this.t("cutPronLabel"))}</i>${escapeHtml(session.pronunciation)}</span>`);
    }
    if (session.translation) {
      meta.push(`<span class="cut-aux"><i>${escapeHtml(this.t("cutTransLabel"))}</i>${escapeHtml(session.translation)}</span>`);
    }

    stage.innerHTML = `
      <div class="cut-head"><b>${escapeHtml(session.layerName)}</b><span>${timeLabel(session.inPoint)}–${timeLabel(session.outPoint)}</span></div>
      ${session.blocked ? `<div class="cut-blocked">${escapeHtml(session.blocked)}</div>` : ""}
      <div class="cut-line">${glyphs}</div>
      <div class="cut-meta">${meta.join("")}</div>
    `;

    const pieces = computePieces(session, this.cutPoints, this.settings.cutReveal);
    const warnings = pieceWarnings(session, pieces);
    if (this.cutPoints.length === 0) {
      piecesNode.innerHTML = `<div class="empty-state">${escapeHtml(this.t("cutNoCuts"))}</div>`;
    } else {
      piecesNode.innerHTML = `
        <div class="cut-piece-head">${escapeHtml(this.t("cutPieceCount", { count: pieces.length }))}</div>
        ${pieces
          .map(
            (piece) => `
          <div class="assignment">
            <div class="assignment-head"><b>${escapeHtml(piece.text)}</b><span>${timeLabel(piece.start)}–${timeLabel(piece.end)}</span></div>
          </div>`,
          )
          .join("")}
        ${warnings.map((warning) => `<small class="cut-warning">${escapeHtml(warning)}</small>`).join("")}
      `;
    }

    applyButton.disabled = Boolean(session.blocked) || pieces.length < 2;
  }

  /**
   * 캐럿 하나에 클릭과 드래그를 같이 건다.
   *
   * 움직이지 않고 뗐으면 자르기/되붙이기, 옆으로 끌었으면 이미 있는 컷의 시각 조정이다.
   * 드래그 중 renderCut이 DOM을 새로 그리므로 리스너는 document에 건다.
   */
  private bindCutStage(): void {
    element<HTMLElement>("cutStage").addEventListener("mousedown", (event) => {
      const target = (event.target as HTMLElement | null)?.closest<HTMLElement>("[data-caret]");
      const session = this.cutSession;
      if (!target || !session || session.blocked) return;
      event.preventDefault();

      const index = Number(target.dataset.caret);
      const existing = this.cutPoints.find((cut) => cut.index === index);
      const startX = event.clientX;
      const startTime = existing?.time ?? defaultCutTime(session, index);
      let dragged = false;

      const onMove = (moveEvent: MouseEvent): void => {
        const dx = moveEvent.clientX - startX;
        if (!dragged && Math.abs(dx) < 3) return;
        dragged = true;
        if (!existing) return;
        const perPixel = moveEvent.shiftKey ? 0.002 : 0.01;
        this.cutPoints = moveCut(session, this.cutPoints, index, startTime + dx * perPixel);
        this.renderCut();
      };
      const onUp = (): void => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (dragged) return;
        this.cutPoints = toggleCut(session, this.cutPoints, index);
        this.renderCut();
      };

      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  private async applyCut(): Promise<void> {
    const session = this.cutSession;
    if (!session || session.blocked) return;
    const pieces = computePieces(session, this.cutPoints, this.settings.cutReveal);
    if (pieces.length < 2) return;

    this.setBusyKey(true, "statusCutApplying");
    try {
      const result = await evalHost<HostResult>("everyricSplitTextLayer", {
        layerIndex: session.layerIndex,
        pieces,
        keepOriginalPosition: element<HTMLInputElement>("keepCutPositionCheck").checked,
      });
      if (!result.ok) throw new Error(result.error || this.t("statusLayerApplyFailed"));
      (result.warnings ?? []).forEach((warning) => this.addProgress("warn", warning));
      this.statusKey("success", "statusCutApplied", { count: result.created ?? pieces.length });
      this.cutSession = null;
      this.cutPoints = [];
      // 원본이 사라졌으니 다음 감시 때 새 선택을 반드시 다시 읽어야 한다.
      this.lastCutSelectionSignature = "";
      this.renderCut();
      await this.refreshComp(true);
    } catch (error) {
      this.statusKey("error", "statusParseError", { error: errorMessage(error) });
    } finally {
      this.setBusy(false);
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  try {
    new EveryricStudioPanel();
  } catch (error) {
    const status = document.getElementById("statusText");
    if (status) status.textContent = errorMessage(error);
  }
});
