// Mark dist/cjs as CommonJS so Node loads the require() build correctly inside a "type": "module" package.
require('fs').writeFileSync(require('path').join(__dirname, '..', 'dist', 'cjs', 'package.json'), '{"type":"commonjs"}\n');
