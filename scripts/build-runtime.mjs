import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createReadStream } from 'node:fs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const component = process.argv[2];
if (!['stt', 'tts'].includes(component)) throw new Error('Use stt or tts');
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error('This reviewed artifact target is darwin-arm64 only');
}
const version = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version;
const macosMajor = Number(execFileSync('/usr/bin/sw_vers', ['-productVersion'], { encoding: 'utf8' }).trim().split('.')[0]);
if (!Number.isSafeInteger(macosMajor) || macosMajor < 14) throw new Error('Unqualified macOS build host');
// Current MLX wheels are compiled for the build host's macOS major. Package
// metadata must never claim compatibility with older systems by default.
const target = `darwin-arm64-macos${macosMajor}`;
const artifactName = `cortex-voice-${component}-${version}-${target}`;
const dist = path.join(root, 'dist');
const staging = path.join(dist, `${artifactName}.staging`);
const relocated = path.join(dist, `${artifactName}.relocated`);
const output = path.join(dist, `${artifactName}.tar.gz`);
const inputs = path.join(root, 'packages', component);
const lock = path.join(inputs, 'requirements-macos-arm64.lock.txt');
if (!fs.existsSync(lock)) throw new Error(`Missing reviewed requirements lock: ${lock}`);
fs.mkdirSync(dist, { recursive: true });
if (fs.existsSync(staging) || fs.existsSync(relocated)) {
  throw new Error('Prior staging directory exists; inspect it before retrying');
}
const pythonExecutable = execFileSync('uv', ['python', 'find', '3.12', '--python-preference', 'only-managed'], { encoding: 'utf8' }).trim();
const pythonRoot = path.dirname(path.dirname(fs.realpathSync(pythonExecutable)));
const pythonBinary = path.join(staging, 'python', 'bin', 'python3.12');
const expectedScript = component === 'stt' ? 'stt_daemon.py' : 'tts_pocket.py';
const sourceScript = path.join(inputs, expectedScript);
const sourceHash = createHash('sha256').update(fs.readFileSync(sourceScript)).digest('hex');
const lockHash = createHash('sha256').update(fs.readFileSync(lock)).digest('hex');
try {
  fs.cpSync(pythonRoot, path.join(staging, 'python'), { recursive: true, verbatimSymlinks: true });
  // Managed Python's pip/idle/2to3 scripts carry absolute build-machine
  // shebangs and are unnecessary in a release that installs no client wheels.
  for (const entry of fs.readdirSync(path.join(staging, 'python', 'bin'))) {
    if (!['python', 'python3', 'python3.12'].includes(entry)) {
      fs.rmSync(path.join(staging, 'python', 'bin', entry), { recursive: true, force: true });
    }
  }
  fs.mkdirSync(path.join(staging, 'runtime'), { recursive: true });
  fs.copyFileSync(sourceScript, path.join(staging, 'runtime', expectedScript));
  fs.copyFileSync(lock, path.join(staging, 'requirements.lock.txt'));
  if (component === 'tts') {
    fs.cpSync(path.join(inputs, 'voice-assets'), path.join(staging, 'runtime', 'voice-assets'), { recursive: true });
  }
  const targetSite = path.join(staging, 'python', 'lib', 'python3.12', 'site-packages');
  execFileSync('uv', ['pip', 'install', '--python', pythonBinary,
    '--target', targetSite, '--require-hashes', '-r', lock], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, UV_LINK_MODE: 'copy' },
    timeout: 20 * 60_000,
  });
  const manifest = {
    schemaVersion: 1,
    id: 'cortex-voice-runtime',
    component,
    version,
    target,
    minimumMacos: `${macosMajor}.0`,
    python: 'python/bin/python3.12',
    script: `runtime/${expectedScript}`,
    scriptSha256: sourceHash,
    lockSha256: lockHash,
    modelIncluded: false,
    voiceCloneIncluded: false,
  };
  fs.writeFileSync(path.join(staging, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  fs.renameSync(staging, relocated);
  const movedPython = path.join(relocated, 'python', 'bin', 'python3.12');
  const imported = component === 'stt'
    ? 'import mlx_whisper, mlx.core, numpy; print("stt imports ready")'
    : 'import pocket_tts, torch, numpy; print("tts imports ready")';
  execFileSync(movedPython, ['-c', imported], { stdio: 'inherit', timeout: 120_000 });
  if (component === 'stt') {
    execFileSync(movedPython, [path.join(relocated, 'runtime', expectedScript), '--self-test'], { stdio: 'inherit', timeout: 120_000 });
  } else {
    execFileSync(movedPython, [path.join(relocated, 'runtime', expectedScript), '--help'], { stdio: 'inherit', timeout: 120_000 });
  }
  const metadata = [];
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      const relative = path.relative(relocated, file);
      if (entry.isSymbolicLink()) {
        const link = fs.readlinkSync(file);
        if (path.isAbsolute(link) || !path.resolve(path.dirname(file), link).startsWith(`${relocated}${path.sep}`)) {
          throw new Error(`External symlink in runtime artifact: ${relative}`);
        }
      } else if (entry.isDirectory()) walk(file);
      else {
        assert.equal(entry.isFile(), true, `Unexpected artifact entry: ${relative}`);
        metadata.push({ name: relative, bytes: fs.statSync(file).size });
      }
    }
  }
  walk(relocated);
  const physicalBytes = metadata.reduce((sum, entry) => sum + entry.bytes, 0);
  execFileSync('tar', ['-czf', output, '-C', relocated, '.'], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, COPYFILE_DISABLE: '1' },
    timeout: 20 * 60_000,
  });
  const hash = createHash('sha256');
  await new Promise((resolve, reject) => {
    const stream = createReadStream(output);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', resolve);
  });
  fs.writeFileSync(`${output}.release.json`, `${JSON.stringify({
    ...manifest,
    archive: path.basename(output),
    archiveBytes: fs.statSync(output).size,
    archiveSha256: hash.digest('hex'),
    installedBytes: physicalBytes,
    fileCount: metadata.length,
    qualified: false,
    qualificationNote: 'Source/import/relocation checks passed; host integration and platform release qualification are separate gates.',
  }, null, 2)}\n`);
  console.log(`Built ${output} from ${metadata.length} regular files (${physicalBytes} installed bytes)`);
} catch (error) {
  // Keep the staging tree as evidence for a failed qualification attempt.
  throw error;
}
