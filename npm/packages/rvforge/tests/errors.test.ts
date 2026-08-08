import { EXIT_CODES, ForgeError, ForgeErrorCode, REDACTED, exitCodeFor, redact, redactDetails, toForgeError } from '../src/errors';

describe('error codes', () => {
  it('assigns a distinct exit code to every error code', () => {
    const codes = Object.values(ForgeErrorCode);
    const exits = codes.map((c) => EXIT_CODES[c]);
    expect(new Set(exits).size).toBe(codes.length);
    expect(exits.every((e) => Number.isInteger(e) && e > 0)).toBe(true);
  });

  it('never uses 0, which is reserved for success', () => {
    expect(Object.values(EXIT_CODES)).not.toContain(0);
  });

  it('maps a ForgeError to its exit code', () => {
    const err = new ForgeError(ForgeErrorCode.UNSIGNED_SEGMENT, 'nope');
    expect(err.exitCode).toBe(EXIT_CODES[ForgeErrorCode.UNSIGNED_SEGMENT]);
    expect(exitCodeFor(ForgeErrorCode.NETWORK)).toBe(EXIT_CODES[ForgeErrorCode.NETWORK]);
  });

  it('serialises to a stable JSON shape', () => {
    const err = new ForgeError(ForgeErrorCode.MANIFEST, 'bad manifest', { field: 'app.name' });
    expect(err.toJSON()).toEqual({
      code: 'FORGE_E_MANIFEST',
      message: 'bad manifest',
      details: { field: 'app.name' },
    });
  });

  it('omits an empty detail bag', () => {
    expect(toForgeError(new Error('boom')).toJSON()).toEqual({
      code: 'FORGE_E_INTERNAL',
      message: 'boom',
    });
  });

  it('preserves an existing ForgeError through toForgeError', () => {
    const original = new ForgeError(ForgeErrorCode.IO, 'io');
    expect(toForgeError(original)).toBe(original);
  });
});

describe('redaction', () => {
  it('strips bearer tokens', () => {
    expect(redact('failed with Authorization: Bearer sk-abc123def456ghi789')).not.toContain('sk-abc123');
  });

  it('strips URL userinfo', () => {
    expect(redact('https://user:hunter2@forge.example/v1/builds')).toBe(
      `https://${REDACTED}@forge.example/v1/builds`,
    );
  });

  it('strips long opaque strings that look like credentials', () => {
    const token = 'ghp16CharsAnd0More0000000000000000000000';
    expect(redact(`token=${token}`)).toBe(`token=${REDACTED}`);
  });

  it('leaves hex digests and ordinary prose alone', () => {
    const digest = 'a'.repeat(64);
    expect(redact(`sha256 ${digest}`)).toBe(`sha256 ${digest}`);
    expect(redact('Cannot open agent.rvf: no such file')).toBe('Cannot open agent.rvf: no such file');
  });

  it('keeps filesystem paths readable even when a segment looks opaque', () => {
    const path = '/tmp/build-cce929e0-abbf-4a5c-8701-f6c2aaf69b9a/forge-out/agent.rvf';
    expect(redact(`Cannot stat ${path}`)).toBe(`Cannot stat ${path}`);
  });

  it('replaces secret-named detail keys wholesale', () => {
    expect(
      redactDetails({
        apiKey: 'value',
        signingKey: 'value',
        nested: { password: 'value', path: 'agent.rvf' },
      }),
    ).toEqual({
      apiKey: REDACTED,
      signingKey: REDACTED,
      nested: { password: REDACTED, path: 'agent.rvf' },
    });
  });

  it('redacts through the ForgeError constructor', () => {
    const err = new ForgeError(ForgeErrorCode.NETWORK, 'auth failed: Bearer abcdef0123456789abcdef0123456789', {
      token: 'super-secret',
    });
    expect(err.message).not.toContain('abcdef0123456789');
    expect(err.details.token).toBe(REDACTED);
  });
});
