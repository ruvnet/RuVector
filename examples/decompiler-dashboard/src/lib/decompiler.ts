import type { DecompileResult } from '../types';
import { fetchPackageInfo, fetchPackageFileList, fetchFileContent, findMainBundle } from './npm-fetch';
import { splitBundle } from './module-splitter';
import { beautifyJS } from './beautifier';

export type DecompileProgress = (message: string, pct: number) => void;

export async function decompilePackage(
  packageName: string,
  version: string,
  onProgress?: DecompileProgress,
): Promise<DecompileResult> {
  const report = onProgress || (() => {});

  report('Fetching package info...', 5);
  const info = await fetchPackageInfo(packageName);
  const resolvedVersion = version || info.latest;

  report(`Listing files for ${packageName}@${resolvedVersion}...`, 15);
  const files = await fetchPackageFileList(packageName, resolvedVersion);

  report('Finding main bundle...', 25);
  const mainFile = findMainBundle(files);
  if (!mainFile) {
    throw new Error(`No suitable JS bundle found in ${packageName}@${resolvedVersion}`);
  }

  report(`Downloading ${mainFile}...`, 35);
  const originalSource = await fetchFileContent(packageName, resolvedVersion, mainFile);

  report('Beautifying source...', 55);
  const beautifiedSource = beautifyJS(originalSource);

  report('Splitting into modules...', 75);
  const { metrics, modules } = splitBundle(originalSource, mainFile);

  report('Done!', 100);

  return {
    packageName,
    version: resolvedVersion,
    metrics,
    modules,
    originalSource,
    beautifiedSource,
  };
}
