import assert from 'node:assert/strict';
import { init, fit, predict } from '../clustering.mjs';

await init();
const values = new Float32Array([0, 0, 2, 2]);
const result = fit(values, 2, { clusters: 1, epochs: 1 });
assert.deepEqual([...result.centroids], [1, 1]);
assert.equal(result.mse, 2);
assert.deepEqual([...predict(values, 2, result.centroids)], [0, 0]);
console.log('RuVector shared e-means dependency passed');
