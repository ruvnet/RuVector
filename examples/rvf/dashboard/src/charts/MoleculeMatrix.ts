import { scaleLinear } from 'd3-scale';
import { select } from 'd3-selection';

export interface MoleculeEntry {
  molecule: string;
  confidence: number;
  wavelength: number;
  snr: number;
}

const MOLECULES = ['O2', 'H2O', 'CH4', 'CO2', 'O3', 'N2O', 'NH3', 'DMS'];
const METRICS = ['Confidence', 'SNR', 'Wavelength'];

export class MoleculeMatrix {
  private container: HTMLElement;
  private wrapper: HTMLElement;
  private svg: SVGSVGElement;

  constructor(container: HTMLElement) {
    this.container = container;

    this.wrapper = document.createElement('div');
    this.wrapper.className = 'chart-container';
    this.container.appendChild(this.wrapper);

    const title = document.createElement('h3');
    title.textContent = 'Molecule Detection Matrix';
    this.wrapper.appendChild(title);

    this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    this.wrapper.appendChild(this.svg);
  }

  update(entries: MoleculeEntry[]): void {
    const entryMap = new Map<string, MoleculeEntry>();
    for (const e of entries) entryMap.set(e.molecule, e);

    const cellW = 70;
    const cellH = 28;
    const labelW = 50;
    const headerH = 24;
    const w = labelW + METRICS.length * cellW;
    const h = headerH + MOLECULES.length * cellH;

    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    const sel = select(this.svg);
    sel.selectAll('*').remove();

    const colorScale = scaleLinear<string>()
      .domain([0, 0.5, 1])
      .range(['#0D1117', '#0E4D40', '#00E5FF']);

    // Column headers
    for (let c = 0; c < METRICS.length; c++) {
      sel.append('text')
        .attr('x', labelW + c * cellW + cellW / 2)
        .attr('y', headerH - 6)
        .attr('text-anchor', 'middle')
        .attr('fill', '#8B949E').attr('font-size', '10')
        .attr('font-family', 'var(--font-mono)')
        .text(METRICS[c]);
    }

    // Rows
    for (let r = 0; r < MOLECULES.length; r++) {
      const mol = MOLECULES[r];
      const entry = entryMap.get(mol);
      const y = headerH + r * cellH;

      // Row label
      sel.append('text')
        .attr('x', labelW - 6).attr('y', y + cellH / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', '#C9D1D9').attr('font-size', '10')
        .attr('font-family', 'var(--font-mono)')
        .text(mol);

      const values = entry
        ? [entry.confidence, Math.min(entry.snr / 50, 1), Math.min(entry.wavelength / 25, 1)]
        : [0, 0, 0];
      const rawValues = entry
        ? [entry.confidence.toFixed(2), entry.snr.toFixed(1), `${entry.wavelength.toFixed(1)}um`]
        : ['--', '--', '--'];

      for (let c = 0; c < METRICS.length; c++) {
        const x = labelW + c * cellW;

        sel.append('rect')
          .attr('x', x + 1).attr('y', y + 1)
          .attr('width', cellW - 2).attr('height', cellH - 2)
          .attr('rx', 3)
          .attr('fill', colorScale(values[c]));

        sel.append('text')
          .attr('x', x + cellW / 2).attr('y', y + cellH / 2 + 4)
          .attr('text-anchor', 'middle')
          .attr('fill', values[c] > 0.5 ? '#0D1117' : '#C9D1D9')
          .attr('font-size', '10').attr('font-weight', '600')
          .attr('font-family', 'var(--font-mono)')
          .text(rawValues[c]);
      }
    }
  }

  destroy(): void {
    this.wrapper.remove();
  }
}
