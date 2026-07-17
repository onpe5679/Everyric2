# Everyric Studio for After Effects

Everyric Studio converts Everyric2 alignment data into editable After Effects typography. It creates
ordinary text layers, static transforms, layer in/out points, and layer-comment metadata only. It does not
create Text Animators or motion keyframes.

## Modes

- **Fill selected layers** keeps each selected layer's bounds and visual design, replacing only its
  Source Text with the timed lyric content that best fits the interval.
- **Build typography** converts alignment atoms into readable blocks and screen cards. Blocks reveal
  cumulatively or together and share a card exit time.

## Split and timing controls

`Readable`, `Balanced`, and `Rhythmic` are starting presets, not fixed answers. Mode B exposes the
underlying controls so each video can choose its own pacing:

- phrase target length (characters)
- maximum words per block
- pause-cut sensitivity
- cumulative phrase reveal or simultaneous line reveal
- pre-roll, post-roll, and maximum blocks per card

The current `Balanced` defaults are 9 characters, 4 words, a 0.32 second pause threshold, and
cumulative reveal.

## Development

```powershell
cd everyric2-ae
npm install
npm run build
npm run install-plugin
```

Open After Effects and choose **Window → Extensions → Everyric Studio**. Use **구조 테스트용
컴포지션 만들기** from the settings drawer for a self-contained structural check. A real workflow
still requires a file-backed vocal audio layer and its lyrics.

For local alignment, set the Python executable to the environment where Everyric2 is installed. The
panel runs:

```text
python -m everyric2.cli sync <audio> <lyrics> --output <json> --format json
```

The plugin can also load CLI arrays, server `{segments: [...]}` responses, and `.everyric.json`
project files. No credential is embedded in the extension.

## Release

```powershell
npm run release:zxp   # build + signed .zxp + manual zip + SHA256SUMS into release/
```

Signing resolves a certificate in this order: `EVERYRIC_CERT_PATH`/`EVERYRIC_CERT_PASSWORD` env vars →
`../secrets/ElysianCert.p12` (password from env or `../secrets/cert-password.txt`) → an auto-generated
10-year self-signed certificate in `../secrets/`. ZXPSignCmd is downloaded to `scripts/.tools/` on first use.

Publishing: bump the version in `package.json`, `CSXS/manifest.xml`, and `src/panel/version.ts`
(the build fails if they disagree), then push a `ae-v<version>` tag. GitHub Actions builds, signs
(secrets `ZXP_CERT_BASE64` + `ZXP_CERT_PASSWORD`, otherwise self-signed), attaches release assets, and
updates the root `latest.json` that deployed panels poll for update badges and managed engine installs.
Engine releases follow the same flow with `engine-v<version>` tags (version source: `pyproject.toml`).
