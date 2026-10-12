#!/usr/bin/env node
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { sweep, reportMarkdown } from './index.mjs';
import { reportHtml } from './visualize.mjs';
const [command, ...args] = process.argv.slice(2);
if (command === 'mcp') {
  if (args.length !== 1) throw new Error('Usage: ruvector-constellation mcp <report.json>');
  if ((await stat(resolve(args[0]))).size > 32 * 1024 * 1024) throw new Error('Report exceeds 32 MiB');
  const { serve } = await import('./mcp.mjs');
  await serve(JSON.parse(await readFile(resolve(args[0]), 'utf8')));
} else if (command === 'sweep') {
  const options = {};
  for (let i=0; i<args.length; i+=2) {
    if (!['--output','--previous','--maintainer'].includes(args[i]) || !args[i+1]) throw new Error('Invalid sweep arguments');
    options[args[i].slice(2)] = args[i+1];
  }
  const output = resolve(options.output ?? 'constellation-report');
  const previous = options.previous ? JSON.parse(await readFile(options.previous,'utf8')) : null;
  const report = await sweep({maintainer:options.maintainer ?? 'ruvnet',previous});
  await mkdir(output,{recursive:true});
  await writeFile(`${output}/inventory.json`, JSON.stringify(report,null,2)+'\n');
  await writeFile(`${output}/inventory.md`,reportMarkdown(report));
  await writeFile(`${output}/inventory.html`,reportHtml(report));
  console.log(JSON.stringify({output,packages:report.packages.length,complete:report.complete,cycles:report.graph.runtimeCycles.length}));
  if (!report.complete) process.exitCode=2;
} else if (command === '--help' || command === 'help' || !command) {
  console.log('ruvector-constellation sweep [--output DIR] [--previous FILE] [--maintainer NAME]\nruvector-constellation mcp <report.json>\nRead-only public registry discovery. Does not install or publish packages.');
} else { throw new Error(`Unknown command: ${command}`); }
