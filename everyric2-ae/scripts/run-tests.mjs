import assert from "assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { pathToFileURL, fileURLToPath } from "url";
import { build } from "esbuild";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "everyric-ae-tests-"));
const outfile = path.join(temp, "planner.mjs");

try {
  await build({
    entryPoints: [path.join(root, "src/panel/planner.ts")],
    outfile,
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node18",
  });
  const { normalizeSyncPayload, planTypography, planLineLyrics, planLayerFill } = await import(
    `${pathToFileURL(outfile).href}?v=${Date.now()}`
  );

  const document = normalizeSyncPayload({
    metadata: { language: "ko" },
    results: [
      {
        text: "아무렇지 않은 척 너를 보내고 돌아섰지만",
        start_time: 1,
        end_time: 6,
        word_segments: Array.from("아무렇지않은척너를보내고돌아섰지만").map((word, index) => ({
          word,
          start: 1 + index * 0.2,
          end: 1.18 + index * 0.2,
        })),
      },
    ],
  });
  assert.equal(document.lines.length, 1);
  assert.equal(document.language, "ko");
  assert.ok(document.lines[0].atoms.length > 10);

  const options = {
    density: "balanced",
    layout: "auto",
    width: 1920,
    height: 1080,
    frameRate: 30,
    fontSize: 94,
    preRollFrames: 3,
    postRollFrames: 8,
    pauseThreshold: 0.32,
    maxBlocksPerCard: 4,
    phraseTargetChars: 9,
    maxTokensPerBlock: 4,
    revealMode: "cumulative",
  };
  const plan = planTypography(document, options, "TEST-GROUP");
  assert.ok(plan.blocks.length >= 2, "readable blocks should be generated");
  assert.ok(plan.blocks.every((block) => block.text.length > 1), "single character blocks are avoided");
  for (const card of plan.cards) {
    assert.ok(card.blocks.every((block) => block.end === card.end), "blocks accumulate to card end");
    assert.ok(card.blocks.every((block) => block.position[0] >= 0 && block.position[0] <= 1920));
    assert.ok(card.blocks.every((block) => block.position[1] >= 0 && block.position[1] <= 1080));
  }

  const phraseDocument = normalizeSyncPayload({
    results: [
      {
        text: "시작의 첫걸음이 꿈을 정한 그날의 마음이 저기 널 부르고 있는걸",
        start_time: 1,
        end_time: 11,
        word_segments: Array.from("시작의첫걸음이꿈을정한그날의마음이저기널부르고있는걸").map((word, index) => ({
          word,
          start: 1 + index * 0.32,
          end: 1.2 + index * 0.32,
        })),
      },
    ],
  });
  const phrasePlan = planTypography(phraseDocument, options, "PHRASE-GROUP");
  assert.ok(phrasePlan.blocks.length <= 4, "balanced mode should keep one lyric line in a readable card");
  assert.ok(
    phrasePlan.blocks.every((block) => block.text.includes(" ") || block.text.length >= 5),
    "balanced mode should avoid isolated short words",
  );

  const finePlan = planTypography(phraseDocument, {
    ...options,
    phraseTargetChars: 5,
    maxTokensPerBlock: 2,
  }, "FINE-GROUP");
  const coarsePlan = planTypography(phraseDocument, {
    ...options,
    phraseTargetChars: 16,
    maxTokensPerBlock: 7,
  }, "COARSE-GROUP");
  assert.ok(finePlan.blocks.length > coarsePlan.blocks.length, "phrase controls should change split granularity");

  const togetherPlan = planTypography(phraseDocument, {
    ...options,
    revealMode: "simultaneous",
  }, "TOGETHER-GROUP");
  for (const card of togetherPlan.cards) {
    assert.ok(card.blocks.every((block) => block.start === card.start), "together reveal should share one in-point");
  }

  const linePlan = planLineLyrics(phraseDocument, options, "LINE-GROUP");
  assert.equal(linePlan.blocks.length, phraseDocument.lines.length, "line lyric mode should create one text layer per sync line");
  assert.equal(linePlan.blocks[0].text, phraseDocument.lines[0].text, "line lyric mode should keep the original line text intact");

  const assignments = planLayerFill(document, [
    { index: 3, name: "A", inPoint: 1, outPoint: 3, text: "old", sourceTextKeys: 0, locked: false },
    { index: 2, name: "B", inPoint: 3, outPoint: 5, text: "old", sourceTextKeys: 1, locked: false },
    { index: 1, name: "C", inPoint: 3.35, outPoint: 7, text: "old", sourceTextKeys: 0, locked: false },
  ]);
  assert.equal(assignments.length, 3);
  assert.equal(assignments[1].skippedReason, "Source Text 키프레임 있음");
  assert.equal(assignments[1].text, "old");
  const assignedText = assignments.filter((item) => !item.skippedReason).map((item) => item.text).join(" ");
  assert.ok(assignedText.includes("아무렇지"));
  assert.ok(assignedText.includes("돌아섰지만"));

  const bounded = planLayerFill(document, [
    { index: 1, name: "Short", inPoint: 1, outPoint: 2.8, text: "old", sourceTextKeys: 0, locked: false },
  ]);
  assert.ok(!bounded[0].text.includes("돌아섰지만"), "tokens beyond the selected range must not be crammed into the final layer");

  const duplicatedTiming = planLayerFill(document, [
    { index: 10, name: "Main", inPoint: 1, outPoint: 3, text: "old", sourceTextKeys: 0, locked: false },
    { index: 11, name: "Shadow", inPoint: 1, outPoint: 3, text: "old", sourceTextKeys: 0, locked: false },
  ]);
  assert.equal(duplicatedTiming[0].text, duplicatedTiming[1].text);
  assert.ok(duplicatedTiming[0].text.length > 0, "duplicated layers with identical timing should receive the same lyric text");

  const overlappingTiming = planLayerFill(document, [
    { index: 20, name: "Wide", inPoint: 1, outPoint: 5, text: "old", sourceTextKeys: 0, locked: false },
    { index: 21, name: "Inset", inPoint: 2, outPoint: 4, text: "old", sourceTextKeys: 0, locked: false },
  ]);
  assert.ok(overlappingTiming[0].text.includes(overlappingTiming[1].text), "overlapping layers should not steal text from each other");

  const hostSource = fs.readFileSync(path.join(root, "src/jsx/host.ts"), "utf8");
  assert.ok(hostSource.includes('layer.comment = "EV2|"'), "generated ownership metadata must remain in layer comments");
  assert.ok(hostSource.includes('layer.name = "EV2 " + block.id + " · "'), "generated layer names should expose card-block ids and lyric text");
  assert.ok(hostSource.includes("if (payload.autoLabelColors)"), "label color cycling must be optional");
  assert.ok(hostSource.includes("layer.label = 1 + (cardNumber % 16)"), "optional label color cycling should still be available");
  assert.ok(hostSource.includes("function everyricRemoveGeneratedLayers"), "cleanup tool must remove generated Everyric layers explicitly");
  assert.ok(hostSource.includes("function everyricCreateLineMarkers"), "line timing markers must be explicit opt-in");
  assert.ok(hostSource.includes("marker.comment = lyric"), "line marker names should display lyric text");
  assert.ok(hostSource.includes("marker.chapter = metadata"), "line marker metadata should not replace visible lyric text");
  assert.ok(!/function everyricCreateTypography[\\s\\S]*new MarkerValue/.test(hostSource), "typography generation must not add timeline markers");

  const versionOutfile = path.join(temp, "version.mjs");
  await build({
    entryPoints: [path.join(root, "src/panel/version.ts")],
    outfile: versionOutfile,
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node18",
  });
  const { compareVersions, isNewerVersion, satisfiesRange, PANEL_VERSION } = await import(
    `${pathToFileURL(versionOutfile).href}?v=${Date.now()}`
  );
  assert.equal(compareVersions("2.1.0", "2.0.9"), 1);
  assert.equal(compareVersions("2.0.0", "2.0.0"), 0);
  assert.equal(compareVersions("0.9.0", "0.10.0"), -1);
  assert.ok(isNewerVersion("2.1.0", "2.0.0"));
  assert.ok(!isNewerVersion("2.0.0", "2.1.0"));
  assert.ok(!isNewerVersion("available", "2.0.0"), "unparseable versions must never suggest an update");
  assert.ok(satisfiesRange("0.4.2", ">=0.1.0 <1.0.0"));
  assert.ok(!satisfiesRange("1.0.0", ">=0.1.0 <1.0.0"));
  assert.ok(!satisfiesRange("weird", ">=0.1.0"), "unparseable versions never satisfy a range");
  const panelPackage = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
  assert.equal(PANEL_VERSION, panelPackage.version, "PANEL_VERSION must match package.json version");

  console.log("Everyric Studio planner tests passed");
} finally {
  fs.rmSync(temp, { recursive: true, force: true });
}
