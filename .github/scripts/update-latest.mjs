import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const manifestPath = path.join(repoRoot, "latest.json");
const REPO_URL = "https://github.com/onpe5679/Everyric2";

const tag = process.argv[2];
if (!tag) {
  console.error("usage: node update-latest.mjs <tag>  (engine-v0.1.0 | chrome-v1.1.0)");
  process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

if (tag.startsWith("engine-v")) {
  const version = tag.slice("engine-v".length);
  manifest.engine = {
    ...manifest.engine,
    version,
    wheelUrl: `${REPO_URL}/releases/download/${tag}/everyric2-${version}-py3-none-any.whl`,
    releaseUrl: `${REPO_URL}/releases/tag/${tag}`,
  };
} else if (tag.startsWith("chrome-v")) {
  const version = tag.slice("chrome-v".length);
  manifest.chrome = {
    ...manifest.chrome,
    version,
    zipUrl: `${REPO_URL}/releases/download/${tag}/Everyric-Chrome-${version}.zip`,
    releaseUrl: `${REPO_URL}/releases/tag/${tag}`,
  };
} else {
  console.error(`unknown tag prefix: ${tag}`);
  process.exit(1);
}

fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
console.log(`latest.json updated for ${tag}`);
