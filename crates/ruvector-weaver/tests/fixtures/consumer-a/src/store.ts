import { RvfDatabase } from '@ruvector/rvf';
import * as path from 'node:path';
import pad from 'left-pad';
export const open = (p: string) => RvfDatabase.open(path.resolve(p), pad);
