import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
fs.rmSync(path.join(root, "dist"), { recursive: true, force: true });
