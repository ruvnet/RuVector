const { platform, arch } = process;
const path = require('path');

const platformMap = {
  'linux': {
    'x64': { package: '@ruvector/diskann-linux-x64-gnu', file: 'ruvector-diskann.linux-x64-gnu.node' },
    'arm64': { package: '@ruvector/diskann-linux-arm64-gnu', file: 'ruvector-diskann.linux-arm64-gnu.node' }
  },
  'darwin': {
    'x64': { package: '@ruvector/diskann-darwin-x64', file: 'ruvector-diskann.darwin-x64.node' },
    'arm64': { package: '@ruvector/diskann-darwin-arm64', file: 'ruvector-diskann.darwin-arm64.node' }
  },
  'win32': {
    'x64': { package: '@ruvector/diskann-win32-x64-msvc', file: 'ruvector-diskann.win32-x64-msvc.node' }
  }
};

function loadNativeModule() {
  const platformInfo = platformMap[platform]?.[arch];

  if (!platformInfo) {
    throw new Error(
      `Unsupported platform: ${platform}-${arch}\n` +
      `@ruvector/diskann is available for: Linux (x64, ARM64), macOS (x64, ARM64), Windows (x64)\n` +
      `Install: npm install @ruvector/diskann`
    );
  }

  try {
    return require(path.join(__dirname, platformInfo.file));
  } catch {
    try {
      return require(platformInfo.package);
    } catch (error) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          `Native module not found for ${platform}-${arch}\n` +
          `Install: npm install ${platformInfo.package}`
        );
      }
      throw error;
    }
  }
}

const native = loadNativeModule();

module.exports = {
  DiskAnn: native.DiskAnn,
};
