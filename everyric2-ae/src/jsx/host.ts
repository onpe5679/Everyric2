declare var app: any;
declare var $: any;
declare var File: any;
declare var CompItem: any;
declare var TextLayer: any;
declare var ParagraphJustification: any;

type HostObject = Record<string, any>;

function response(value: HostObject): string {
  try {
    return JSON.stringify(value);
  } catch (error) {
    return '{"ok":false,"error":"JSON response failed"}';
  }
}

function parsePayload(payloadJson?: string): HostObject {
  if (!payloadJson) return {};
  return JSON.parse(payloadJson);
}

function activeComp(): any | null {
  var item = app.project && app.project.activeItem;
  return item && item instanceof CompItem ? item : null;
}

function textProperty(layer: any): any | null {
  try {
    return layer.property("ADBE Text Properties").property("ADBE Text Document");
  } catch (error) {
    return null;
  }
}

function isTextLayer(layer: any): boolean {
  return Boolean(layer && (layer instanceof TextLayer || textProperty(layer)));
}

function textLayerData(layer: any): HostObject {
  var sourceText = textProperty(layer);
  var currentText = "";
  if (sourceText) {
    try {
      currentText = sourceText.value.text || "";
    } catch (error) {}
  }
  return {
    index: layer.index,
    name: layer.name,
    inPoint: layer.inPoint,
    outPoint: layer.outPoint,
    text: currentText,
    sourceTextKeys: sourceText ? sourceText.numKeys : 0,
    locked: Boolean(layer.locked),
  };
}

function getAudioLayers(comp: any): HostObject[] {
  var result: HostObject[] = [];
  for (var index = 1; index <= comp.numLayers; index += 1) {
    var layer = comp.layer(index);
    try {
      if (!layer.hasAudio || !layer.source) continue;
      var filePath: string | undefined;
      if (layer.source.file) filePath = layer.source.file.fsName;
      result.push({
        index: layer.index,
        name: layer.name,
        inPoint: layer.inPoint,
        outPoint: layer.outPoint,
        filePath: filePath,
      });
    } catch (error) {}
  }
  return result;
}

function getSelectedTextLayers(comp: any): HostObject[] {
  var result: HostObject[] = [];
  var selected = comp.selectedLayers || [];
  for (var index = 0; index < selected.length; index += 1) {
    if (isTextLayer(selected[index])) result.push(textLayerData(selected[index]));
  }
  return result;
}

function countGeneratedLayers(comp: any): number {
  var count = 0;
  for (var index = 1; index <= comp.numLayers; index += 1) {
    try {
      if (String(comp.layer(index).comment || "").indexOf("EV2|") === 0) count += 1;
    } catch (error) {}
  }
  return count;
}

function countEveryricMarkers(comp: any): number {
  var count = 0;
  try {
    var markers = comp.markerProperty;
    for (var keyIndex = 1; keyIndex <= markers.numKeys; keyIndex += 1) {
      if (isEveryricMarker(markers.keyValue(keyIndex))) count += 1;
    }
  } catch (error) {}
  for (var layerIndex = 1; layerIndex <= comp.numLayers; layerIndex += 1) {
    try {
      var layerMarkers = comp.layer(layerIndex).property("ADBE Marker");
      for (var layerKey = 1; layerKey <= layerMarkers.numKeys; layerKey += 1) {
        if (isEveryricMarker(layerMarkers.keyValue(layerKey))) count += 1;
      }
    } catch (error) {}
  }
  return count;
}

function everyricGetCompInfo(): string {
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: true, hasComp: false });
    return response({
      ok: true,
      hasComp: true,
      name: comp.name,
      width: comp.width,
      height: comp.height,
      duration: comp.duration,
      frameRate: comp.frameRate,
      time: comp.time,
      selectedTextLayers: getSelectedTextLayers(comp),
      generatedLayerCount: countGeneratedLayers(comp),
      everyricMarkerCount: countEveryricMarkers(comp),
      audioLayers: getAudioLayers(comp),
    });
  } catch (error) {
    return response({ ok: false, hasComp: false, error: String(error) });
  }
}

function everyricGetSelectedTextLayers(): string {
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, layers: [], error: "활성 컴포지션이 없습니다." });
    var layers = getSelectedTextLayers(comp);
    return response({ ok: true, layers: layers });
  } catch (error) {
    return response({ ok: false, layers: [], error: String(error) });
  }
}

function everyricApplyTextAssignments(payloadJson?: string): string {
  var undoStarted = false;
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, error: "활성 컴포지션이 없습니다." });
    var payload = parsePayload(payloadJson);
    var assignments = payload.assignments || [];
    if (!assignments.length) return response({ ok: false, error: "적용할 레이어가 없습니다." });

    app.beginUndoGroup("Everyric Studio - Fill selected layers");
    undoStarted = true;
    var updated = 0;
    var skipped = 0;
    var warnings: string[] = [];
    for (var index = 0; index < assignments.length; index += 1) {
      var assignment = assignments[index];
      var layer = comp.layer(Number(assignment.layerIndex));
      if (!layer || !isTextLayer(layer) || layer.locked) {
        skipped += 1;
        warnings.push("레이어를 건너뜀: " + String(assignment.layerName || assignment.layerIndex));
        continue;
      }
      var sourceText = textProperty(layer);
      if (!sourceText || sourceText.numKeys > 0) {
        skipped += 1;
        warnings.push("Source Text 키프레임 레이어를 건너뜀: " + layer.name);
        continue;
      }
      var documentValue = sourceText.value;
      documentValue.text = String(assignment.text || "");
      sourceText.setValue(documentValue);
      updated += 1;
    }
    return response({ ok: true, updated: updated, skipped: skipped, warnings: warnings });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  } finally {
    if (undoStarted) app.endUndoGroup();
  }
}

function justificationValue(name: string): any {
  if (name === "left") return ParagraphJustification.LEFT_JUSTIFY;
  if (name === "right") return ParagraphJustification.RIGHT_JUSTIFY;
  return ParagraphJustification.CENTER_JUSTIFY;
}

function styleSeed(comp: any): HostObject | null {
  var selected = comp.selectedLayers || [];
  for (var index = 0; index < selected.length; index += 1) {
    var sourceText = textProperty(selected[index]);
    if (!sourceText) continue;
    try {
      var documentValue = sourceText.value;
      return {
        font: documentValue.font,
        applyFill: documentValue.applyFill,
        fillColor: documentValue.fillColor,
        applyStroke: documentValue.applyStroke,
        strokeColor: documentValue.strokeColor,
        strokeWidth: documentValue.strokeWidth,
        fauxBold: documentValue.fauxBold,
        fauxItalic: documentValue.fauxItalic,
        tracking: documentValue.tracking,
        autoLeading: documentValue.autoLeading,
        leading: documentValue.leading,
      };
    } catch (error) {}
  }
  return null;
}

function applyDocumentStyle(documentValue: any, block: HostObject, seed: HostObject | null): void {
  documentValue.text = String(block.text || "");
  documentValue.fontSize = Math.max(1, Number(block.fontSize) || 72);
  documentValue.justification = justificationValue(String(block.justification || "center"));
  documentValue.applyFill = true;
  documentValue.fillColor = block.color || [1, 1, 1];
  if (!seed) return;
  try { if (seed.font) documentValue.font = seed.font; } catch (error) {}
  try { documentValue.applyStroke = seed.applyStroke; } catch (error) {}
  try { if (seed.strokeColor) documentValue.strokeColor = seed.strokeColor; } catch (error) {}
  try { if (seed.strokeWidth !== undefined) documentValue.strokeWidth = seed.strokeWidth; } catch (error) {}
  try { documentValue.fauxBold = seed.fauxBold; } catch (error) {}
  try { documentValue.fauxItalic = seed.fauxItalic; } catch (error) {}
  try { documentValue.tracking = seed.tracking; } catch (error) {}
  try { documentValue.autoLeading = seed.autoLeading; } catch (error) {}
  try { if (!seed.autoLeading && seed.leading) documentValue.leading = seed.leading; } catch (error) {}
}

function setTextAnchor(layer: any, block: HostObject): void {
  try {
    var sampleTime = Math.max(layer.inPoint, 0);
    var rect = layer.sourceRectAtTime(sampleTime, false);
    var anchorX = rect.left + rect.width / 2;
    if (block.justification === "left") anchorX = rect.left;
    if (block.justification === "right") anchorX = rect.left + rect.width;
    var anchorY = rect.top + rect.height / 2;
    layer.property("ADBE Transform Group").property("ADBE Anchor Point").setValue([anchorX, anchorY]);
  } catch (error) {}
}

function isEveryricMarker(marker: any): boolean {
  try {
    if (String(marker.comment || "").indexOf("EV2|") === 0) return true;
  } catch (error) {}
  try {
    if (String(marker.chapter || "").indexOf("EV2|") === 0) return true;
  } catch (error) {}
  try {
    if (String(marker.url || "").indexOf("EV2|") === 0) return true;
  } catch (error) {}
  return false;
}

function removeGeneratedMarkers(comp: any): number {
  var removed = 0;
  try {
    var markers = comp.markerProperty;
    for (var keyIndex = markers.numKeys; keyIndex >= 1; keyIndex -= 1) {
      var marker = markers.keyValue(keyIndex);
      if (isEveryricMarker(marker)) {
        markers.removeKey(keyIndex);
        removed += 1;
      }
    }
  } catch (error) {}
  for (var layerIndex = 1; layerIndex <= comp.numLayers; layerIndex += 1) {
    var layer = comp.layer(layerIndex);
    try {
      var layerMarkers = layer.property("ADBE Marker");
      for (var layerKey = layerMarkers.numKeys; layerKey >= 1; layerKey -= 1) {
        var layerMarker = layerMarkers.keyValue(layerKey);
        if (isEveryricMarker(layerMarker)) {
          layerMarkers.removeKey(layerKey);
          removed += 1;
        }
      }
    } catch (error) {}
  }
  return removed;
}

function removeGeneratedLayers(comp: any): number {
  var removed = 0;
  removeGeneratedMarkers(comp);
  for (var index = comp.numLayers; index >= 1; index -= 1) {
    var layer = comp.layer(index);
    try {
      if (String(layer.comment || "").indexOf("EV2|") === 0) {
        layer.remove();
        removed += 1;
      }
    } catch (error) {}
  }
  return removed;
}

function everyricCreateTypography(payloadJson?: string): string {
  var undoStarted = false;
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, error: "활성 컴포지션이 없습니다." });
    var payload = parsePayload(payloadJson);
    var plan = payload.plan;
    if (!plan || !plan.blocks || !plan.blocks.length) return response({ ok: false, error: "생성 계획이 비어 있습니다." });
    if (plan.blocks.length > 1000) return response({ ok: false, error: "안전을 위해 한 번에 1000개 이상의 레이어를 만들 수 없습니다." });

    app.beginUndoGroup("Everyric Studio - Build typography");
    undoStarted = true;
    var removed = payload.replacePrevious ? removeGeneratedLayers(comp) : 0;
    if (!payload.replacePrevious) removeGeneratedMarkers(comp);
    var seed = styleSeed(comp);
    var created = 0;
    var bottomToTop = String(payload.layerOrder || "bottom-to-top") === "bottom-to-top";
    var startIndex = bottomToTop ? 0 : plan.blocks.length - 1;
    var endIndex = bottomToTop ? plan.blocks.length : -1;
    var step = bottomToTop ? 1 : -1;
    for (var index = startIndex; index !== endIndex; index += step) {
      var block = plan.blocks[index];
      var start = Math.max(0, Math.min(comp.duration - comp.frameDuration, Number(block.start)));
      var end = Math.max(start + comp.frameDuration, Math.min(comp.duration, Number(block.end)));
      var layer = comp.layers.addText(String(block.text || ""));
      layer.name = "EV2 " + block.id + " · " + String(block.text || "").slice(0, 24);
      layer.comment = "EV2|" + plan.groupId + "|" + block.cardId + "|" + block.id;
      if (payload.autoLabelColors) {
        try {
          var cardNumber = Number(String(block.cardId || "").replace(/[^0-9]/g, ""));
          layer.label = 1 + (cardNumber % 16);
        } catch (error) {}
      }
      layer.startTime = 0;
      layer.inPoint = start;
      layer.outPoint = end;
      var sourceText = textProperty(layer);
      var documentValue = sourceText.value;
      applyDocumentStyle(documentValue, block, seed);
      sourceText.setValue(documentValue);
      setTextAnchor(layer, block);
      layer.property("ADBE Transform Group").property("ADBE Position").setValue(block.position);
      layer.property("ADBE Transform Group").property("ADBE Rotate Z").setValue(Number(block.rotation) || 0);
      created += 1;
    }
    return response({ ok: true, created: created, removed: removed });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  } finally {
    if (undoStarted) app.endUndoGroup();
  }
}

function everyricRemoveGeneratedLayers(): string {
  var undoStarted = false;
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, error: "활성 컴포지션이 없습니다." });
    app.beginUndoGroup("Everyric Studio - Remove generated layers");
    undoStarted = true;
    return response({
      ok: true,
      removed: removeGeneratedLayers(comp),
      markerCount: countEveryricMarkers(comp),
    });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  } finally {
    if (undoStarted) app.endUndoGroup();
  }
}

function everyricCreateLineMarkers(payloadJson?: string): string {
  var undoStarted = false;
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, error: "활성 컴포지션이 없습니다." });
    var payload = parsePayload(payloadJson);
    var documentValue = payload.document;
    var lines = documentValue && documentValue.lines ? documentValue.lines : [];
    if (!lines.length) return response({ ok: false, error: "마커로 만들 싱크 라인이 없습니다." });

    app.beginUndoGroup("Everyric Studio - Add line markers");
    undoStarted = true;
    removeGeneratedMarkers(comp);
    var created = 0;
    for (var index = 0; index < lines.length; index += 1) {
      var line = lines[index];
      var start = Math.max(0, Math.min(comp.duration, Number(line.start) || 0));
      var end = Math.max(start, Math.min(comp.duration, Number(line.end) || start));
      var lyric = String(line.text || "Line " + String(index + 1)).slice(0, 180);
      var metadata = "EV2|LINE|" + String(index + 1) + "|" + start.toFixed(3) + "|" + end.toFixed(3);
      var marker = new MarkerValue(lyric);
      marker.comment = lyric;
      try { marker.chapter = metadata; } catch (error) {}
      try { marker.url = metadata; } catch (error) {}
      try { marker.duration = Math.max(comp.frameDuration, end - start); } catch (error) {}
      comp.markerProperty.setValueAtTime(start, marker);
      created += 1;
    }
    return response({ ok: true, created: created });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  } finally {
    if (undoStarted) app.endUndoGroup();
  }
}

function everyricPickFile(payloadJson?: string): string {
  try {
    var payload = parsePayload(payloadJson);
    var kind = String(payload.kind || "json");
    var filter = kind === "json" ? "JSON:*.json" : "All files:*.*";
    var file = File.openDialog("Everyric Studio - 파일 선택", filter, false);
    if (!file) return response({ ok: false });
    return response({ ok: true, path: file.fsName });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  }
}

$.global.everyricGetCompInfo = everyricGetCompInfo;
$.global.everyricGetSelectedTextLayers = everyricGetSelectedTextLayers;
$.global.everyricApplyTextAssignments = everyricApplyTextAssignments;
$.global.everyricCreateTypography = everyricCreateTypography;
$.global.everyricRemoveGeneratedLayers = everyricRemoveGeneratedLayers;
$.global.everyricCreateLineMarkers = everyricCreateLineMarkers;
$.global.everyricRemoveGeneratedMarkers = function (): string {
  try {
    var comp = activeComp();
    if (!comp) return response({ ok: false, error: "활성 컴포지션이 없습니다." });
    return response({ ok: true, removed: removeGeneratedMarkers(comp) });
  } catch (error) {
    return response({ ok: false, error: String(error) });
  }
};
$.global.everyricPickFile = everyricPickFile;
