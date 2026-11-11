import { type EMeansOptions } from '@ruvector/e-means';
import {
  trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer,
  productQuantizerToJSON, productQuantizerFromJSON, type ProductQuantizer,
} from '@ruvector/e-means/quantization';
import { partition, postingList, type IVFPartition } from '@ruvector/e-means/ivf';

const data = new Float32Array([1, 2, 3, 4]);
const options: EMeansOptions = { clusters: 1 };
const pq: ProductQuantizer = trainProductQuantizer(data, 2, 2, options);
const codes: Uint8Array = encodeProductQuantizer(pq, data);
const restored: Float32Array = decodeProductQuantizer(productQuantizerFromJSON(productQuantizerToJSON(pq)), codes);
const ivf: IVFPartition = partition(restored, 2, options);
const posting: Uint32Array = postingList(ivf, 0);
void posting;
// @ts-expect-error byte codes are required
decodeProductQuantizer(pq, new Uint32Array());
// @ts-expect-error float32 input vectors are required
trainProductQuantizer([1, 2], 2, 1, options);
