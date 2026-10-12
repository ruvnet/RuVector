import { cp, access, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const source = new URL('../../../../skills/ruvector-constellation/', import.meta.url);
await access(source);
await mkdir(new URL('../skills/', import.meta.url),{recursive:true});
await cp(source,new URL('../skills/ruvector-constellation/',import.meta.url),{recursive:true});
await cp(new URL('../../../../LICENSE',import.meta.url),new URL('../LICENSE',import.meta.url));
console.log('Packed skill from '+fileURLToPath(source));
