# @ruvector/scipix

[![npm version](https://img.shields.io/npm/v/@ruvector/scipix.svg)](https://www.npmjs.com/package/@ruvector/scipix)
[![npm downloads](https://img.shields.io/npm/dm/@ruvector/scipix.svg)](https://www.npmjs.com/package/@ruvector/scipix)
[![License](https://img.shields.io/npm/l/@ruvector/scipix.svg)](https://github.com/ruvnet/ruvector/blob/main/LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

**OCR client for scientific documents - extract LaTeX, MathML from equations and research papers.**

A TypeScript client for the SciPix OCR API with support for mathematical equations, tables, and diagrams.

## Features

- 📐 **LaTeX Extraction** - Convert equation images to LaTeX notation
- 🧮 **MathML Output** - Get MathML for web rendering
- 📄 **PDF Processing** - Extract from multi-page scientific PDFs
- 🖼️ **Multiple Formats** - PNG, JPEG, WebP, TIFF, BMP support
- 📦 **Batch Processing** - Process multiple images efficiently
- 🎯 **Region Detection** - Identify equations, tables, and diagrams
- 💪 **Type-Safe** - Full TypeScript support

## Installation

```bash
npm install @ruvector/scipix
```

## Quick Start

```typescript
import { SciPixClient, OutputFormat } from '@ruvector/scipix';

// Create client
const client = new SciPixClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key',
});

// OCR an image file
const result = await client.ocrFile('./equation.png', {
  formats: [OutputFormat.LaTeX, OutputFormat.MathML],
  detectEquations: true,
});

console.log('LaTeX:', result.latex);
console.log('MathML:', result.mathml);
console.log('Confidence:', result.confidence);

// Quick LaTeX extraction
const latex = await client.extractLatex('./math.png');
console.log('Extracted:', latex);
// Output: \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
```

## Tutorials

### Basic OCR Usage

```typescript
import { SciPixClient, OutputFormat } from '@ruvector/scipix';
import { readFile } from 'fs/promises';

const client = new SciPixClient({
  baseUrl: process.env.SCIPIX_URL || 'http://localhost:8080',
  apiKey: process.env.SCIPIX_API_KEY,
  timeout: 30000,
});

// From file path
const result1 = await client.ocrFile('./document.png');

// From Buffer
const buffer = await readFile('./equation.png');
const result2 = await client.ocr(buffer);

// From base64
const base64 = buffer.toString('base64');
const result3 = await client.ocr(base64);

// From data URL
const dataUrl = `data:image/png;base64,${base64}`;
const result4 = await client.ocr(dataUrl);
```

### Extracting Equations from Research Papers

```typescript
import { SciPixClient, OutputFormat, ContentType } from '@ruvector/scipix';

const client = new SciPixClient({ baseUrl: 'http://localhost:8080' });

// Process a research paper PDF
const result = await client.ocrFile('./paper.pdf', {
  formats: [OutputFormat.LaTeX, OutputFormat.Text],
  detectEquations: true,
  detectTables: true,
  pages: [1, 2, 3], // First 3 pages only
});

// Filter equation regions
const equations = result.regions.filter(r => r.contentType === ContentType.Equation);

console.log(`Found ${equations.length} equations:`);
equations.forEach((eq, i) => {
  console.log(`\nEquation ${i + 1}:`);
  console.log(`  LaTeX: ${eq.latex}`);
  console.log(`  Confidence: ${(eq.confidence * 100).toFixed(1)}%`);
  console.log(`  Position: (${eq.bbox.x}, ${eq.bbox.y})`);
});

// Get all LaTeX in document order
const allLatex = equations.map(eq => eq.latex).join('\n\n');
```

### Batch Processing Multiple Images

```typescript
import { SciPixClient, OutputFormat } from '@ruvector/scipix';
import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

const client = new SciPixClient({ baseUrl: 'http://localhost:8080' });

// Get all PNG files in directory
const files = await readdir('./equations');
const pngFiles = files.filter(f => f.endsWith('.png'));

// Prepare batch request
const images = await Promise.all(
  pngFiles.map(async (file) => ({
    id: file,
    source: (await readFile(join('./equations', file))).toString('base64'),
  }))
);

// Process batch
const batchResult = await client.batchOcr({
  images,
  defaultOptions: {
    formats: [OutputFormat.LaTeX],
    detectEquations: true,
    minConfidence: 0.8,
  },
});

console.log(`Processed: ${batchResult.successful}/${batchResult.totalImages}`);
console.log(`Total time: ${batchResult.totalProcessingTime}ms`);

// Handle results
for (const item of batchResult.results) {
  if (item.success) {
    console.log(`${item.id}: ${item.result?.latex}`);
  } else {
    console.error(`${item.id} failed: ${item.error}`);
  }
}
```

### Building an AI Tutor with Equation Recognition

```typescript
import { SciPixClient } from '@ruvector/scipix';
import Anthropic from '@anthropic-ai/sdk';

const scipix = new SciPixClient({ baseUrl: 'http://localhost:8080' });
const anthropic = new Anthropic();

async function solveEquation(imagePath: string) {
  // Extract LaTeX from image
  const latex = await scipix.extractLatex(imagePath);
  console.log('Detected equation:', latex);

  // Ask Claude to solve it
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    messages: [{
      role: 'user',
      content: `Solve this equation step by step: ${latex}`
    }]
  });

  return {
    equation: latex,
    solution: response.content[0].text,
  };
}

// Usage
const result = await solveEquation('./homework-problem.png');
console.log(result.solution);
```

### Handling Different Output Formats

```typescript
import { SciPixClient, OutputFormat } from '@ruvector/scipix';

const client = new SciPixClient({ baseUrl: 'http://localhost:8080' });

// Get all formats
const result = await client.ocr(imageBuffer, {
  formats: [
    OutputFormat.LaTeX,
    OutputFormat.MathML,
    OutputFormat.AsciiMath,
    OutputFormat.Text,
  ],
});

// LaTeX for documents
console.log('LaTeX:', result.latex);
// \int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}

// MathML for web rendering
console.log('MathML:', result.mathml);
// <math><msubsup><mo>∫</mo><mn>0</mn><mo>∞</mo></msubsup>...

// Plain text fallback
console.log('Text:', result.text);
// ∫₀^∞ e^(-x²) dx = √π/2

// Use in HTML
const html = `
  <div class="equation">
    ${result.mathml}
  </div>
`;
```

### Error Handling and Retries

```typescript
import { SciPixClient, SciPixError, SciPixErrorCode } from '@ruvector/scipix';

const client = new SciPixClient({
  baseUrl: 'http://localhost:8080',
  timeout: 30000,
  maxRetries: 3,
});

async function safeOcr(imagePath: string) {
  try {
    return await client.ocrFile(imagePath);
  } catch (error) {
    if (error instanceof SciPixError) {
      switch (error.code) {
        case SciPixErrorCode.Timeout:
          console.error('Request timed out, try a smaller image');
          break;
        case SciPixErrorCode.InvalidImage:
          console.error('Invalid image format');
          break;
        case SciPixErrorCode.RateLimited:
          console.error('Rate limited, wait before retrying');
          await new Promise(r => setTimeout(r, 60000));
          return safeOcr(imagePath); // Retry
        case SciPixErrorCode.Unauthorized:
          console.error('Invalid API key');
          break;
        default:
          console.error('OCR failed:', error.message);
      }
    }
    throw error;
  }
}
```

### Health Monitoring

```typescript
import { SciPixClient } from '@ruvector/scipix';

const client = new SciPixClient({ baseUrl: 'http://localhost:8080' });

// Check API health
const health = await client.health();

console.log('Status:', health.status);
console.log('Version:', health.version);
console.log('Uptime:', health.uptime, 'seconds');
console.log('Models:');
health.models.forEach(model => {
  console.log(`  - ${model.name} v${model.version}: ${model.loaded ? '✓' : '✗'}`);
});
```

## API Reference

### SciPixClient

| Method | Description |
|--------|-------------|
| `ocr(image, options?)` | OCR an image (Buffer, base64, or path) |
| `ocrFile(path, options?)` | OCR from file path |
| `batchOcr(request)` | Process multiple images |
| `extractLatex(image)` | Quick LaTeX extraction |
| `extractMathML(image)` | Quick MathML extraction |
| `health()` | Check API health status |

### OCROptions

| Option | Type | Description |
|--------|------|-------------|
| `formats` | `OutputFormat[]` | Desired output formats |
| `languages` | `string[]` | Language hints |
| `detectEquations` | `boolean` | Enable equation detection |
| `detectTables` | `boolean` | Enable table detection |
| `detectDiagrams` | `boolean` | Enable diagram detection |
| `minConfidence` | `number` | Minimum confidence (0-1) |
| `preprocess` | `boolean` | Enable preprocessing |
| `dpi` | `number` | DPI hint for scanned docs |
| `pages` | `number[]` | Pages to process (PDF) |

### OCRResult

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Plain text output |
| `latex` | `string?` | LaTeX output |
| `mathml` | `string?` | MathML output |
| `regions` | `OCRRegion[]` | Detected regions |
| `confidence` | `number` | Overall confidence |
| `processingTime` | `number` | Time in milliseconds |

## Running the SciPix Server

This package is a client for the SciPix OCR server. To run the server:

```bash
# From the ruvector repository
cd examples/scipix
cargo run --release -- serve --port 8080

# Or with Docker
docker run -p 8080:8080 ruvector/scipix
```

See [SciPix documentation](https://github.com/ruvnet/ruvector/tree/main/examples/scipix) for server setup.

## Related Packages

- [@ruvector/raft](https://www.npmjs.com/package/@ruvector/raft) - Raft consensus for distributed systems
- [@ruvector/replication](https://www.npmjs.com/package/@ruvector/replication) - Data replication with vector clocks
- [ruvector](https://www.npmjs.com/package/ruvector) - High-performance vector database

## License

MIT OR Apache-2.0
