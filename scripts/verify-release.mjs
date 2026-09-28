import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createReadStream } from 'node:fs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';

const archive = path.resolve(process.argv[2] || '');
if (!archive.endsWith('.tar.gz') || !fs.statSync(archive).isFile()) {
  throw new Error('Pass one built .tar.gz artifact');
}
const release = JSON.parse(fs.readFileSync(`${archive}.release.json`, 'utf8'));
assert.equal(release.archive, path.basename(archive));
assert.equal(release.archiveBytes, fs.statSync(archive).size);
assert.equal(release.id, 'cortex-voice-runtime');
assert.ok(['stt', 'tts'].includes(release.component));
assert.match(release.target, /^darwin-arm64-macos\d+$/);
assert.equal(release.minimumMacos, `${Number(release.target.split('macos')[1])}.0`);
const digest = createHash('sha256');
await new Promise((resolve, reject) => {
  const stream = createReadStream(archive);
  stream.on('data', (chunk) => digest.update(chunk));
  stream.on('error', reject);
  stream.on('end', resolve);
});
assert.equal(digest.digest('hex'), release.archiveSha256);

const listing = execFileSync('tar', ['-tf', archive], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
for (const raw of listing.split('\n').filter(Boolean)) {
  const cleaned = raw.replace(/^\.\//, '');
  if (path.posix.isAbsolute(cleaned) || path.posix.normalize(cleaned).startsWith('../')) {
    throw new Error(`Unsafe archive entry: ${raw}`);
  }
}
const extracted = fs.mkdtempSync(path.join(os.tmpdir(), 'cortex-voice-archive-'));
try {
  execFileSync('tar', ['-xzf', archive, '-C', extracted], { timeout: 20 * 60_000 });
  const manifest = JSON.parse(fs.readFileSync(path.join(extracted, 'manifest.json'), 'utf8'));
  for (const key of ['schemaVersion', 'id', 'component', 'version', 'target', 'minimumMacos', 'python', 'script', 'scriptSha256', 'lockSha256']) {
    assert.equal(manifest[key], release[key], `Manifest mismatch: ${key}`);
  }
  const script = path.join(extracted, manifest.script);
  assert.equal(createHash('sha256').update(fs.readFileSync(script)).digest('hex'), manifest.scriptSha256);
  assert.equal(createHash('sha256').update(fs.readFileSync(path.join(extracted, 'requirements.lock.txt'))).digest('hex'), manifest.lockSha256);
  const python = path.join(extracted, manifest.python);
  const probe = manifest.component === 'stt'
    ? 'import mlx_whisper, mlx.core, numpy; print("stt runtime ready")'
    : 'import pocket_tts, torch, numpy; print("tts runtime ready")';
  execFileSync(python, ['-c', probe], { stdio: 'inherit', timeout: 120_000 });
  const argumentsForScript = manifest.component === 'stt' ? ['--self-test'] : ['--help'];
  execFileSync(python, [script, ...argumentsForScript], { stdio: 'inherit', timeout: 120_000 });
  console.log(JSON.stringify({ verified: true, component: manifest.component, target: manifest.target, bytes: release.archiveBytes, extracted }));
} finally {
  fs.rmSync(extracted, { recursive: true, force: true });
}
