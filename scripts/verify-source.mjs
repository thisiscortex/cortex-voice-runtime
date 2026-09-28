import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pinned = new Map([
  ['packages/stt/stt_daemon.py', '136cfc68d492f886b16090406e3640ff9abdad2799ba7ed7067442ef38648133'],
  ['packages/tts/tts_pocket.py', '8d931c084f58cc577e5088d6325818b5000c4983a88e4ac7dd862c3dfc7d21b0'],
  ['packages/tts/pocket-model/english.yaml', '41c53705299f8e682a18cd93d0c32a6f5a02d4332f2d68e09b56134eba98ccb5'],
]);
for (const [file, expected] of pinned) {
  const actual = createHash('sha256').update(readFileSync(path.join(root, file))).digest('hex');
  if (actual !== expected) throw new Error(`${file}: source hash changed without a new reviewed pin`);
}
const forbidden = /\.(?:safetensors|pt|wav|mp3)$|(?:^|\/)\.env(?:\.|$)|(?:^|\/)\.venv(?:\/|$)/i;
function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (['.git', '__pycache__', 'node_modules', 'dist'].includes(entry.name)) continue;
    const relative = path.relative(root, path.join(dir, entry.name));
    if (forbidden.test(relative)) throw new Error(`Unsafe source asset: ${relative}`);
    if (entry.isSymbolicLink()) throw new Error(`Symlink in source release: ${relative}`);
    if (entry.isDirectory()) walk(path.join(dir, entry.name));
    else if (!entry.isFile() || statSync(path.join(dir, entry.name)).size > 2 * 1024 * 1024) throw new Error(`Unexpected source file: ${relative}`);
  }
}
walk(root);
console.log('Voice source hashes and public asset boundary verified');
