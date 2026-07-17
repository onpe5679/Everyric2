# Everyric Studio for After Effects — Product 1-Pager

## Background

Everyric2 already produces line and character/word timing data, but the existing AE prototype only
creates subtitle-like layers. A practical typography workflow needs to keep the timing precision while
generating readable, editable text layers that can be handed off to motion tools such as Animation
Composer or Shuffle It.

## Problem

Rendering alignment atoms directly produces one-character flashes. Rendering complete lyric lines
removes most of the rhythmic response. The plugin therefore needs separate concepts for alignment
atoms, readable typography blocks, and screen cards.

## Goal

- Replace the contents of selected text layers without changing their timing, transforms, style,
  effects, or parenting.
- Generate static, editable AE text layers arranged as typography cards. The plugin may set layer
  in/out points and static text/transform values, but it must not add text animators or motion keys.
- Preserve exact lyric order and timing provenance.
- Preview every destructive layer operation before applying it.
- Work with local Everyric2 JSON immediately and optionally run the local Everyric2 CLI.
- Keep the Everyric2 server and alignment core unchanged unless a missing integration contract makes
  a narrowly scoped change unavoidable.

## Non-goals

- Automatic motion design, text animators, easing, or transition presets.
- Replacing specialist animation plugins.
- Generating a final art-directed composition without user editing.
- Storing API credentials inside the extension bundle.

## Constraints

- After Effects 2025 and 2026 on Windows.
- Generated output must be ordinary AE text layers. Ownership metadata stays in layer comments so
  the composition and layer marker tracks remain clean.
- A single AE undo operation must revert each apply/generate action.
- Existing layers with Source Text keyframes are skipped by default.
- External alignment output is treated as untrusted input and validated before it reaches ExtendScript.

## Core model

1. **Atom** — character, syllable, or word timestamp. Used as a timing anchor only.
2. **Block** — a readable semantic/prosodic unit. Its target character count and maximum word count
   are user-controlled because there is no universal split size.
3. **Card** — 2–4 blocks that share a layout and a common exit time.

Blocks can use cumulative reveal (each phrase enters near its first atom) or simultaneous reveal (all
phrases enter at the card start). Both modes retain a common card exit. Readable/Balanced/Rhythmic
presets only seed the editable phrase-size, word-count, and pause sensitivity controls.

## Mode A — Fill selected layers

Selected text layers retain their in/out points and all visual properties. Timed lyric tokens are
assigned once, in order, to the layer whose interval best contains each token. Empty intervals are
reported. Keyframed Source Text layers are skipped unless the user explicitly opts in.

## Mode B — Build typography

Each input lyric line becomes one or more cards. Boundaries combine punctuation, acoustic gaps,
length limits, and minimum readable duration. A deterministic layout preset assigns normalized
positions, scale, rotation, and justification. All generated layers receive Everyric comment metadata
and deterministic names so a later generation can safely replace only its own output.

## Definition of done

- Type checking, planner unit tests, and production build pass.
- The extension installs into the user CEP extensions directory with all required files.
- After Effects opens the panel without a loading error.
- In a real composition, Mode A updates selected text layers without changing their bounds.
- In a real composition, Mode B creates readable cumulative or simultaneous blocks with no text
  animators and a single undo step.
- Generated layer timing, count, text order, safe-area placement, and metadata are inspected in AE.
