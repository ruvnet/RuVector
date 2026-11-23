#!/usr/bin/env node
/**
 * decompileUrl checks the address it actually connects to, not only the
 * result of an earlier, separate lookup: the pre-flight resolver is stubbed to
 * answer with a public address, and a request to a loopback server must
 * still be refused. With RUVECTOR_ALLOW_PRIVATE_FETCH=1 the same request path
 * fetches and follows redirects normally.
 */

'use strict';

const assert = require('assert');
const dns = require('dns');
const http = require('http');

delete process.env.RUVECTOR_ALLOW_PRIVATE_FETCH;

async function main() {
  let hits = 0;
  const server = http.createServer((req, res) => {
    hits++;
    res.writeHead(200, { 'content-type': 'application/javascript' });
    res.end('module.exports = { value: 1 };\n');
  });
  // '::' with ipv6Only=false accepts both 127.0.0.1 and ::1, whichever
  // localhost resolves to first.
  await new Promise((resolve) => server.listen({ port: 0, host: '::', ipv6Only: false }, resolve));
  const { port } = server.address();

  // First answer (the pre-flight check) is public; later answers are real.
  const realLookup = dns.promises.lookup;
  dns.promises.lookup = async (host, opts) => {
    dns.promises.lookup = realLookup;
    return opts && opts.all ? [{ address: '93.184.216.34', family: 4 }] : { address: '93.184.216.34', family: 4 };
  };

  const { decompileUrl } = require('../src/decompiler/index.js');
  let err = null;
  try {
    await decompileUrl(`http://localhost:${port}/bundle.js`, { useRust: false, witness: false });
  } catch (e) {
    err = e;
  } finally {
    dns.promises.lookup = realLookup;
    server.close();
  }

  assert.strictEqual(hits, 0, `request reached the loopback server (${hits} hit(s)): the connection used an unchecked address`);
  assert.ok(err, 'decompileUrl must fail for a host whose connect-time address is private');
  assert.match(String(err.message), /private address/i, `unexpected error: ${err && err.message}`);

  // Public-address behaviour is unchanged: a literal private IP is still refused up front.
  await assert.rejects(decompileUrl('http://169.254.169.254/latest/meta-data/'), /private address/i);

  // With the explicit opt-in, the pinned request path still fetches and
  // follows a redirect to the body.
  process.env.RUVECTOR_ALLOW_PRIVATE_FETCH = '1';
  const okServer = http.createServer((req, res) => {
    if (req.url === '/start') {
      res.writeHead(302, { location: '/bundle.js' });
      return res.end();
    }
    res.writeHead(200, { 'content-type': 'application/javascript' });
    res.end('function hello() { return 42; }\nmodule.exports = { hello };\n');
  });
  await new Promise((resolve) => okServer.listen(0, '127.0.0.1', resolve));
  try {
    const out = await decompileUrl(`http://127.0.0.1:${okServer.address().port}/start`, { useRust: false, witness: false });
    assert.ok(out && typeof out === 'object', 'decompileUrl should return a result object');
  } finally {
    delete process.env.RUVECTOR_ALLOW_PRIVATE_FETCH;
    okServer.close();
  }

  console.log('decompile-url-pinned-dns: OK');
}

main().catch((e) => {
  console.error('decompile-url-pinned-dns: FAIL');
  console.error(e && e.stack || e);
  process.exit(1);
});
