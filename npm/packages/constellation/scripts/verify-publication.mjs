import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';

const directory = resolve(process.argv[2] || 'release-candidate');
const seal = JSON.parse(await readFile(join(directory, 'seal.json'), 'utf8'));
if (seal.name !== '@ruvector/constellation' || !/^[a-zA-Z0-9._-]+\.tgz$/.test(seal.filename) || !/^[0-9a-f]{40}$/.test(seal.sourceSha) || !Number.isSafeInteger(seal.tests) || seal.tests < 1 || seal.packedConsumer !== 'passed') throw new Error('Invalid package seal');
const sourceManifest = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
if (seal.name !== sourceManifest.name || seal.version !== sourceManifest.version) throw new Error('Seal identity differs from source manifest');
const packedManifest = JSON.parse(execFileSync('tar', ['-xOf', join(directory, seal.filename), 'package/package.json'], { encoding: 'utf8', maxBuffer: 1024 * 1024 }));
if (packedManifest.name !== seal.name || packedManifest.version !== seal.version) throw new Error('Tarball identity differs from seal');
const local = await readFile(join(directory, seal.filename));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const integrity = bytes => `sha512-${createHash('sha512').update(bytes).digest('base64')}`;
if (digest(local) !== seal.sha256 || integrity(local) !== seal.integrity) throw new Error('Local tarball differs from seal');
if (process.env.EXPECTED_SHA && seal.sourceSha !== process.env.EXPECTED_SHA) throw new Error('Sealed source differs from approved SHA');
if (process.argv.includes('--local-only')) {
  console.log(`Verified sealed tarball ${seal.name}@${seal.version}`);
} else {
  const url = `https://registry.npmjs.org/${encodeURIComponent(seal.name)}/${encodeURIComponent(seal.version)}`;
  let verified = false;
  for (let attempt = 0; attempt < 6; attempt++) {
    const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!response.ok) {
      if (![404, 429, 500, 502, 503, 504].includes(response.status)) throw new Error(`Registry HTTP ${response.status}`);
    } else {
      const metadata = await response.json();
      const tarballUrl = new URL(metadata.dist?.tarball);
      if (tarballUrl.origin !== 'https://registry.npmjs.org') throw new Error('Unexpected tarball origin');
      if (metadata.name !== seal.name || metadata.version !== seal.version || metadata.dist.integrity !== seal.integrity) throw new Error('Published metadata differs from seal');
      const download = await fetch(tarballUrl, { redirect: 'error', signal: AbortSignal.timeout(30_000) });
      if (!download.ok) throw new Error(`Tarball HTTP ${download.status}`);
      const bytes = Buffer.from(await download.arrayBuffer());
      if (digest(bytes) !== seal.sha256 || integrity(bytes) !== seal.integrity || !bytes.equals(local)) throw new Error('Published bytes differ from tested tarball');
      verified = true;
      break;
    }
    if (attempt < 5) await new Promise(resolve => setTimeout(resolve, 5000));
  }
  if (!verified) throw new Error('Published version unavailable after bounded retries');
  const consumer = await mkdtemp(join(tmpdir(), 'constellation-public-consumer-'));
  const cleanEnv = {...process.env, NPM_CONFIG_USERCONFIG: '/dev/null'};
  for (const key of ['NPM_TOKEN','NODE_AUTH_TOKEN','GH_TOKEN','GITHUB_TOKEN']) delete cleanEnv[key];
  try {
    await writeFile(join(consumer,'package.json'), JSON.stringify({private:true,type:'module'}));
    execFileSync('npm',['install','--ignore-scripts','--workspaces=false','--no-audit','--no-fund',`${seal.name}@${seal.version}`],{cwd:consumer,env:cleanEnv,stdio:'pipe'});
    execFileSync(process.execPath,['node_modules/@ruvector/constellation/src/cli.mjs','--help'],{cwd:consumer,env:cleanEnv,stdio:'pipe'});
    execFileSync(process.execPath,['--input-type=module','-e',"const m=await import('@ruvector/constellation');if(typeof m.sweep!=='function')throw Error('Missing sweep')"],{cwd:consumer,env:cleanEnv,stdio:'pipe'});
    await writeFile(join(directory,'publication-receipt.json'),JSON.stringify({name:seal.name,version:seal.version,sourceSha:seal.sourceSha,sha256:seal.sha256,integrity:seal.integrity,publicBytesMatch:true,publicConsumer:'passed',verifiedAt:new Date().toISOString()},null,2)+'\n',{flag:'wx'});
  } finally { await rm(consumer,{recursive:true,force:true}); }
  console.log(`Verified published bytes ${seal.name}@${seal.version} SHA256 ${seal.sha256}`);
}
