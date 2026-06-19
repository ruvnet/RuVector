const { platform, arch } = process;
const path = require('path');

/**
 * Detect whether this Linux runtime is musl (Alpine) rather than glibc.
 * Uses the Node process report header, which exposes `glibcVersionRuntime`
 * only on glibc systems. No glibc runtime => musl.
 */
function isMusl() {
  if (platform !== 'linux') return false;
  try {
    const report = typeof process.report?.getReport === 'function'
      ? process.report.getReport()
      : null;
    const header = report && report.header;
    if (header && typeof header.glibcVersionRuntime === 'string') {
      return false; // glibc present
    }
    return true; // no glibc runtime reported => musl
  } catch {
    return false; // default to glibc on any detection failure
  }
}

// Platform mapping for @ruvector/tiny-dancer. Linux entries are functions so
// libc (glibc vs musl) is resolved at load time.
const platformMap = {
  linux: {
    x64: () => isMusl()
      ? { package: '@ruvector/tiny-dancer-linux-x64-musl', file: 'ruvector-tiny-dancer.linux-x64-musl.node' }
      : { package: '@ruvector/tiny-dancer-linux-x64-gnu', file: 'ruvector-tiny-dancer.linux-x64-gnu.node' },
    arm64: () => isMusl()
      ? { package: '@ruvector/tiny-dancer-linux-arm64-musl', file: 'ruvector-tiny-dancer.linux-arm64-musl.node' }
      : { package: '@ruvector/tiny-dancer-linux-arm64-gnu', file: 'ruvector-tiny-dancer.linux-arm64-gnu.node' },
  },
  darwin: {
    x64: () => ({ package: '@ruvector/tiny-dancer-darwin-x64', file: 'ruvector-tiny-dancer.darwin-x64.node' }),
    arm64: () => ({ package: '@ruvector/tiny-dancer-darwin-arm64', file: 'ruvector-tiny-dancer.darwin-arm64.node' }),
  },
  win32: {
    x64: () => ({ package: '@ruvector/tiny-dancer-win32-x64-msvc', file: 'ruvector-tiny-dancer.win32-x64-msvc.node' }),
    arm64: () => ({ package: '@ruvector/tiny-dancer-win32-arm64-msvc', file: 'ruvector-tiny-dancer.win32-arm64-msvc.node' }),
  },
};

function loadNativeModule() {
  const resolve = platformMap[platform]?.[arch];
  const platformInfo = typeof resolve === 'function' ? resolve() : null;

  if (!platformInfo) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}\n` +
      `@ruvector/tiny-dancer native module is available for:\n` +
      `- Linux x64/ARM64 (glibc and musl/Alpine)\n` +
      `- macOS x64/ARM64\n` +
      `- Windows x64/ARM64\n\n` +
      `Install the package for your platform:\n` +
      `  npm install @ruvector/tiny-dancer`
    );
  }

  // Try local .node file first (for development and bundled packages)
  try {
    const localPath = path.join(__dirname, platformInfo.file);
    return require(localPath);
  } catch (localError) {
    // Fall back to platform-specific package
    try {
      return require(platformInfo.package);
    } catch (error) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          `Native module not found for ${platform}-${arch}\n` +
          `Please install: npm install ${platformInfo.package}\n` +
          `Or reinstall @ruvector/tiny-dancer to get optional dependencies`
        );
      }
      throw error;
    }
  }
}

module.exports = loadNativeModule();
