/**
 * react-native-css metro plugin emits watcher events that don't match Metro 0.83+
 * FileMap's ChangeEvent shape (`changes.addedFiles`), which crashes with:
 *   Cannot read properties of undefined (reading 'addedFiles')
 *
 * Strip the broken emit so bundling works; after editing global CSS you may need to reload the app.
 */
const fs = require("fs");
const path = require("path");

const targets = [
  "node_modules/react-native-css/dist/commonjs/metro/index.js",
  "node_modules/react-native-css/dist/module/metro/index.js",
];

const root = path.join(__dirname, "..");

function stripWatcherEmitCall(source) {
  const needle = 'watcher.emit("change",';
  let out = source;
  let from = 0;

  while (true) {
    const i = out.indexOf(needle, from);
    if (i === -1) break;

    const braceOpen = out.indexOf("{", i);
    if (braceOpen === -1) break;

    let depth = 0;
    let k = braceOpen;
    for (; k < out.length; k++) {
      const ch = out[k];
      if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) {
          k++;
          while (k < out.length && /\s/.test(out[k])) k++;
          if (out[k] === ")") k++;
          while (k < out.length && /\s/.test(out[k])) k++;
          if (out[k] === ";") k++;
          let start = i;
          while (start > 0 && out[start - 1] === " ") start--;
          if (out[start - 1] === "\n") start--;
          out = out.slice(0, start) + out.slice(k);
          from = start;
          break;
        }
      }
    }
    if (k >= out.length) break;
  }

  out = out.replace(/\n\s*const watcher = bundler\.getWatcher\(\);\n/g, "\n");

  return out;
}

for (const rel of targets) {
  const file = path.join(root, rel);
  if (!fs.existsSync(file)) continue;
  const before = fs.readFileSync(file, "utf8");
  const after = stripWatcherEmitCall(before);
  if (after !== before) {
    fs.writeFileSync(file, after);
    console.log(`[patch-react-native-css-metro] updated ${rel}`);
  }
}
