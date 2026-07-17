import type {
  Density,
  FillAssignment,
  LayoutPreset,
  PlannerOptions,
  SyncDocument,
  SyncLine,
  TextLayerInfo,
  TimingAtom,
  TypographyBlock,
  TypographyCard,
  TypographyPlan,
} from "./types";

type UnknownRecord = Record<string, unknown>;

interface TokenSpan extends TimingAtom {
  leadingSpace: boolean;
}

const PUNCTUATION_END = /[.!?。！？…,:;，、]$/;
const CLOSING_PUNCTUATION = /^[.!?。！？…,:;，、）)\]」』]/;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function textValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function extractRawLines(payload: unknown): unknown[] {
  if (Array.isArray(payload)) return payload;
  if (!isRecord(payload)) return [];
  for (const key of ["segments", "results", "timestamps"]) {
    const candidate = payload[key];
    if (Array.isArray(candidate)) return candidate;
  }
  return [];
}

function normalizeAtoms(raw: UnknownRecord, lineText: string, start: number, end: number): TimingAtom[] {
  const sources = [raw.atoms, raw.words, raw.word_segments, raw.pron_segments];
  for (const source of sources) {
    if (!Array.isArray(source) || source.length === 0) continue;
    const atoms = source
      .map((item): TimingAtom | null => {
        if (!isRecord(item)) return null;
        const atomStart = finiteNumber(item.start ?? item.start_time);
        const atomEnd = finiteNumber(item.end ?? item.end_time);
        const atomText = textValue(item.text ?? item.word);
        if (atomStart === undefined || atomEnd === undefined || !atomText || atomEnd < atomStart) {
          return null;
        }
        const confidence = finiteNumber(item.confidence);
        return { text: atomText, start: atomStart, end: atomEnd, ...(confidence === undefined ? {} : { confidence }) };
      })
      .filter((item): item is TimingAtom => item !== null)
      .sort((a, b) => a.start - b.start || a.end - b.end);
    if (atoms.length > 0) return atoms;
  }

  const visibleCharacters = Array.from(lineText).filter((char) => !/\s/.test(char));
  if (visibleCharacters.length === 0) return [];
  const duration = Math.max(0.001, end - start);
  return visibleCharacters.map((char, index) => ({
    text: char,
    start: start + (duration * index) / visibleCharacters.length,
    end: start + (duration * (index + 1)) / visibleCharacters.length,
  }));
}

export function normalizeSyncPayload(payload: unknown, sourceLabel = "JSON"): SyncDocument {
  const rawLines = extractRawLines(payload);
  if (rawLines.length === 0) throw new Error("동기화 구간을 찾을 수 없습니다.");

  const lines: SyncLine[] = [];
  for (const candidate of rawLines) {
    if (!isRecord(candidate)) continue;
    const text = textValue(candidate.text).trim();
    const start = finiteNumber(candidate.start ?? candidate.start_time);
    const end = finiteNumber(candidate.end ?? candidate.end_time);
    if (!text || start === undefined || end === undefined || start < 0 || end <= start) continue;
    const confidence = finiteNumber(candidate.confidence);
    lines.push({
      text,
      start,
      end,
      atoms: normalizeAtoms(candidate, text, start, end),
      ...(confidence === undefined ? {} : { confidence }),
      ...(typeof candidate.translation === "string" ? { translation: candidate.translation } : {}),
      ...(typeof candidate.pronunciation === "string" ? { pronunciation: candidate.pronunciation } : {}),
    });
  }

  lines.sort((a, b) => a.start - b.start || a.end - b.end);
  if (lines.length === 0) throw new Error("유효한 동기화 구간이 없습니다.");

  let language = "auto";
  if (isRecord(payload) && isRecord(payload.metadata) && typeof payload.metadata.language === "string") {
    language = payload.metadata.language;
  } else if (isRecord(payload) && typeof payload.language === "string") {
    language = payload.language;
  }

  return {
    lines,
    language,
    sourceLabel,
    duration: Math.max(...lines.map((line) => line.end)),
  };
}

function lineTokens(line: SyncLine): TokenSpan[] {
  const matches = Array.from(line.text.matchAll(/\S+/g));
  if (matches.length <= 1) {
    const characters = Array.from(line.text).filter((char) => !/\s/.test(char));
    if (characters.length === 0) return [];
    const atoms = line.atoms.length > 0 ? line.atoms : normalizeAtoms({}, line.text, line.start, line.end);
    const result: TokenSpan[] = [];
    let atomIndex = 0;
    let chunk = "";
    let chunkStart = atoms[0]?.start ?? line.start;
    for (let index = 0; index < characters.length; index += 1) {
      const char = characters[index] ?? "";
      if (!chunk) chunkStart = atoms[atomIndex]?.start ?? line.start;
      chunk += char;
      atomIndex += 1;
      const punctuation = PUNCTUATION_END.test(char);
      if (chunk.length >= 4 || punctuation || index === characters.length - 1) {
        result.push({
          text: chunk,
          start: chunkStart,
          end: atoms[Math.max(0, atomIndex - 1)]?.end ?? line.end,
          leadingSpace: false,
        });
        chunk = "";
      }
    }
    return result;
  }

  const compactAtoms = line.atoms.filter((atom) => atom.text.trim() !== "");
  const totalVisible = matches.reduce((sum, match) => sum + Array.from(match[0]).length, 0);
  let consumedVisible = 0;
  let atomCursor = 0;
  return matches.map((match, tokenIndex) => {
    const token = match[0];
    const tokenChars = Array.from(token).length;
    let count = tokenChars;
    if (compactAtoms.length !== totalVisible && tokenIndex === matches.length - 1) {
      count = Math.max(1, compactAtoms.length - atomCursor);
    }
    const startAtom = compactAtoms[atomCursor];
    atomCursor = Math.min(compactAtoms.length, atomCursor + count);
    consumedVisible += tokenChars;
    const endAtom = compactAtoms[Math.max(0, atomCursor - 1)];
    const ratioStart = (consumedVisible - tokenChars) / Math.max(1, totalVisible);
    const ratioEnd = consumedVisible / Math.max(1, totalVisible);
    return {
      text: token,
      start: startAtom?.start ?? line.start + (line.end - line.start) * ratioStart,
      end: endAtom?.end ?? line.start + (line.end - line.start) * ratioEnd,
      leadingSpace: (match.index ?? 0) > 0,
    };
  });
}

function densityLimits(
  options: PlannerOptions,
): { minChars: number; targetChars: number; maxChars: number; maxTokens: number } {
  const targetChars = Math.max(3, Math.min(24, Math.round(options.phraseTargetChars)));
  return {
    minChars: Math.max(2, Math.round(targetChars * 0.55)),
    targetChars,
    maxChars: Math.max(targetChars + 2, Math.round(targetChars * 1.65)),
    maxTokens: Math.max(1, Math.min(8, Math.round(options.maxTokensPerBlock))),
  };
}

function joinTokens(tokens: TokenSpan[]): string {
  return tokens.reduce((text, token, index) => {
    if (index === 0) return token.text;
    if (CLOSING_PUNCTUATION.test(token.text)) return text + token.text;
    return `${text} ${token.text}`;
  }, "");
}

function splitIntoBlockTokens(tokens: TokenSpan[], options: PlannerOptions): TokenSpan[][] {
  const limits = densityLimits(options);
  const groups: TokenSpan[][] = [];
  let current: TokenSpan[] = [];
  let chars = 0;

  const flush = (): void => {
    if (current.length > 0) groups.push(current);
    current = [];
    chars = 0;
  };

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token) continue;
    const previous = current[current.length - 1];
    const pauseBefore = previous ? token.start - previous.end : 0;
    const projected = chars + Array.from(token.text).length + (current.length > 0 ? 1 : 0);
    if (
      current.length > 0 &&
      ((pauseBefore >= options.pauseThreshold && chars >= limits.minChars) ||
        projected > limits.maxChars ||
        current.length >= limits.maxTokens)
    ) {
      flush();
    }
    current.push(token);
    chars += Array.from(token.text).length + (current.length > 1 ? 1 : 0);

    const hardStop = PUNCTUATION_END.test(token.text);
    const next = tokens[index + 1];
    const nextPause = next ? next.start - token.end : 0;
    if (
      hardStop ||
      chars >= limits.targetChars ||
      (nextPause >= options.pauseThreshold && chars >= limits.minChars)
    ) {
      flush();
    }
  }
  flush();

  for (let index = groups.length - 1; index >= 0 && groups.length > 1; index -= 1) {
    const group = groups[index];
    if (!group || joinTokens(group).length >= limits.minChars) continue;
    const previous = groups[index - 1];
    const next = groups[index + 1];
    if (previous && joinTokens([...previous, ...group]).length <= limits.maxChars) {
      previous.push(...group);
      groups.splice(index, 1);
    } else if (next && joinTokens([...group, ...next]).length <= limits.maxChars) {
      next.unshift(...group);
      groups.splice(index, 1);
    }
  }
  return groups;
}

function chooseLayout(layout: LayoutPreset, blockCount: number, cardIndex: number): Exclude<LayoutPreset, "auto"> {
  if (layout !== "auto") return layout;
  if (blockCount <= 1) return "center";
  const cycle: Array<Exclude<LayoutPreset, "auto">> = ["editorial", "split", "diagonal", "center"];
  return cycle[cardIndex % cycle.length] ?? "center";
}

function blockVisual(
  preset: Exclude<LayoutPreset, "auto">,
  index: number,
  count: number,
  text: string,
  options: PlannerOptions,
): Pick<TypographyBlock, "position" | "fontSize" | "rotation" | "justification" | "color" | "emphasis"> {
  const w = options.width;
  const h = options.height;
  const safeX = w * 0.1;
  const safeY = h * 0.12;
  const longFactor = Math.max(0.66, Math.min(1, 10 / Math.max(6, Array.from(text).length)));
  const base = Math.round(options.fontSize * longFactor);
  const emphasis = index === 0 ? 1 : Math.max(0.72, 1 - index * 0.08);
  const color: [number, number, number] = index === 0 ? [0.96, 0.96, 0.93] : [0.78, 0.8, 0.82];

  if (preset === "editorial") {
    const y = safeY + ((h - safeY * 2) * (index + 0.65)) / Math.max(1, count);
    return {
      position: [index % 2 === 0 ? safeX : w - safeX, y],
      fontSize: Math.round(base * (index === 0 ? 1.18 : 0.88)),
      rotation: 0,
      justification: index % 2 === 0 ? "left" : "right",
      color,
      emphasis,
    };
  }
  if (preset === "split") {
    const columns = count === 2 ? 2 : Math.min(2, count);
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      position: [column === 0 ? w * 0.28 : w * 0.72, h * (0.42 + row * 0.2)],
      fontSize: Math.round(base * (index === 0 ? 1.12 : 0.92)),
      rotation: 0,
      justification: "center",
      color,
      emphasis,
    };
  }
  if (preset === "diagonal") {
    const ratio = count <= 1 ? 0.5 : index / (count - 1);
    return {
      position: [safeX + (w - safeX * 2) * ratio, h * (0.3 + ratio * 0.42)],
      fontSize: Math.round(base * (1.08 - index * 0.06)),
      rotation: -4 + index * 3,
      justification: "center",
      color,
      emphasis,
    };
  }
  return {
    position: [w / 2, h * (0.5 + (index - (count - 1) / 2) * 0.13)],
    fontSize: Math.round(base * (index === 0 ? 1.12 : 0.94)),
    rotation: 0,
    justification: "center",
    color,
    emphasis,
  };
}

function uniqueGroupId(): string {
  return `EV2-${Date.now().toString(36).toUpperCase()}`;
}

export function planTypography(document: SyncDocument, options: PlannerOptions, groupId = uniqueGroupId()): TypographyPlan {
  const cards: TypographyCard[] = [];
  const warnings: string[] = [];
  const frame = 1 / Math.max(1, options.frameRate);
  let cardCounter = 0;

  for (const line of document.lines) {
    const tokens = lineTokens(line);
    if (tokens.length === 0) {
      warnings.push(`빈 구간을 건너뜀: ${line.text}`);
      continue;
    }
    const groups = splitIntoBlockTokens(tokens, options);
    for (let offset = 0; offset < groups.length; offset += options.maxBlocksPerCard) {
      const cardGroups = groups.slice(offset, offset + options.maxBlocksPerCard);
      if (cardGroups.length === 0) continue;
      cardCounter += 1;
      const cardId = `C${String(cardCounter).padStart(2, "0")}`;
      const firstToken = cardGroups[0]?.[0];
      const lastGroup = cardGroups[cardGroups.length - 1];
      const lastToken = lastGroup?.[lastGroup.length - 1];
      if (!firstToken || !lastToken) continue;
      const nextCardStart = groups[offset + options.maxBlocksPerCard]?.[0]?.start;
      const naturalEnd = Math.min(line.end, nextCardStart ?? line.end);
      const cardEnd = Math.max(lastToken.end + options.postRollFrames * frame, naturalEnd);
      const preset = chooseLayout(options.layout, cardGroups.length, cardCounter - 1);
      const cardStart = Math.max(0, firstToken.start - options.preRollFrames * frame);
      const blocks: TypographyBlock[] = cardGroups.map((group, blockIndex) => {
        const first = group[0] ?? firstToken;
        const last = group[group.length - 1] ?? lastToken;
        const text = joinTokens(group);
        const visual = blockVisual(preset, blockIndex, cardGroups.length, text, options);
        return {
          id: `${cardId}-B${String(blockIndex + 1).padStart(2, "0")}`,
          cardId,
          text,
          start: options.revealMode === "simultaneous"
            ? cardStart
            : Math.max(0, first.start - options.preRollFrames * frame),
          end: Math.max(cardEnd, last.end),
          ...visual,
        };
      });
      cards.push({
        id: cardId,
        start: blocks[0]?.start ?? line.start,
        end: cardEnd,
        sourceText: cardGroups.map(joinTokens).join(" "),
        blocks,
      });
    }
  }

  const blocks = cards.flatMap((card) => card.blocks);
  if (blocks.length > 250) warnings.push(`레이어 ${blocks.length}개가 생성됩니다. 밀도를 낮추는 것을 권장합니다.`);
  return { groupId, cards, blocks, warnings };
}

export function planLineLyrics(document: SyncDocument, options: PlannerOptions, groupId = uniqueGroupId()): TypographyPlan {
  const frame = 1 / Math.max(1, options.frameRate);
  const cards: TypographyCard[] = [];
  const warnings: string[] = [];
  document.lines.forEach((line, index) => {
    const cardId = `C${String(index + 1).padStart(2, "0")}`;
    const text = line.text.trim();
    if (!text) return;
    const start = Math.max(0, line.start - options.preRollFrames * frame);
    const end = Math.max(start + frame, line.end + options.postRollFrames * frame);
    const visual = blockVisual("center", 0, 1, text, options);
    const block: TypographyBlock = {
      id: `${cardId}-L01`,
      cardId,
      text,
      start,
      end,
      ...visual,
    };
    cards.push({
      id: cardId,
      start,
      end,
      sourceText: text,
      blocks: [block],
    });
  });
  const blocks = cards.flatMap((card) => card.blocks);
  if (blocks.length > 250) warnings.push(`레이어 ${blocks.length}개가 생성됩니다. 밀도를 낮추는 것을 권장합니다.`);
  return { groupId, cards, blocks, warnings };
}

function tokenStream(document: SyncDocument): TokenSpan[] {
  return document.lines.flatMap(lineTokens).sort((a, b) => a.start - b.start || a.end - b.end);
}

export function planLayerFill(
  document: SyncDocument,
  layers: TextLayerInfo[],
  includeKeyframed = false,
): FillAssignment[] {
  const sortedLayers = [...layers].sort((a, b) => a.inPoint - b.inPoint || a.index - b.index);
  const tokens = tokenStream(document);

  return sortedLayers.map((layer) => {
    let skippedReason: string | undefined;
    if (layer.locked) skippedReason = "잠긴 레이어";
    else if (!includeKeyframed && layer.sourceTextKeys > 0) skippedReason = "Source Text 키프레임 있음";
    const bucket = tokens.filter((token) => {
      const midpoint = (token.start + token.end) / 2;
      return midpoint >= layer.inPoint && midpoint < layer.outPoint;
    });
    return {
      layerIndex: layer.index,
      layerName: layer.name,
      text: skippedReason ? layer.text : joinTokens(bucket),
      inPoint: layer.inPoint,
      outPoint: layer.outPoint,
      ...(skippedReason ? { skippedReason } : {}),
    };
  });
}
