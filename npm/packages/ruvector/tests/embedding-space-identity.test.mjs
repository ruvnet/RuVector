import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = path.resolve(here, '..');
const fixtures = JSON.parse(fs.readFileSync(
  path.resolve(pkg, '../../../schemas/fixtures/embedding-space-identity-v1.json'),
  'utf8',
));
const identity = await import(pathToFileURL(
  path.join(pkg, 'dist/core/embedding-provenance.js'),
).href);
const onnx = await import(pathToFileURL(
  path.join(pkg, 'dist/core/onnx-embedder.js'),
).href);

assert.equal(
  identity.embeddingSpaceId(fixtures.base.identity),
  fixtures.base.embedding_space_id,
);
assert.equal(
  identity.embeddingSpaceId(fixtures.changed_query_template.identity),
  fixtures.changed_query_template.embedding_space_id,
);
assert.notEqual(
  fixtures.base.embedding_space_id,
  fixtures.changed_query_template.embedding_space_id,
);

const access = identity.embeddingSpaceAccess(
  fixtures.base.identity,
  fixtures.changed_query_template.identity,
);
assert.equal(access.textEmbedding, false);
assert.equal(access.corpusMutation, false);
assert.equal(access.vectorRead, true);
assert.notEqual(
  identity.embeddingCacheKey(fixtures.base.identity, 'query', 'same text'),
  identity.embeddingCacheKey(fixtures.changed_query_template.identity, 'query', 'same text'),
);
assert.notEqual(
  identity.embeddingCacheKey(fixtures.base.identity, 'query', 'same text'),
  identity.embeddingCacheKey(fixtures.base.identity, 'passage', 'same text'),
);

const loadedA = onnx.deriveOnnxEmbeddingSpaceIdentity(
  'bge-small-en-v1.5',
  new Uint8Array([1, 2, 3]),
  '{"tokenizer":"a"}',
  512,
  true,
  384,
);
const loadedB = onnx.deriveOnnxEmbeddingSpaceIdentity(
  'bge-small-en-v1.5',
  new Uint8Array([1, 2, 4]),
  '{"tokenizer":"a"}',
  512,
  true,
  384,
);
assert.equal(loadedA.role_policy, 'asymmetric');
assert.equal(loadedA.prefix_policy, 'query-recommended');
assert.notEqual(identity.embeddingSpaceId(loadedA), identity.embeddingSpaceId(loadedB));
assert.notEqual(
  identity.embeddingCacheKey(loadedA, 'query', 'same text'),
  identity.embeddingCacheKey(loadedB, 'query', 'same text'),
);

console.log('embedding-space identity fixture: ok');
