import type * as THREE from 'three';

export type LODLevel = 'boundary' | 'topk' | 'full';

export interface LODNode {
  id: string;
  weight: number;
  isBoundary: boolean;
}

const TOP_K = 50;

export class LODController {
  private nodes: LODNode[] = [];
  private boundaryIds: string[] = [];
  private topkIds: string[] = [];
  private allIds: string[] = [];

  setData(nodes: LODNode[]): void {
    this.nodes = nodes;
    this.allIds = nodes.map((n) => n.id);
    this.boundaryIds = nodes.filter((n) => n.isBoundary).map((n) => n.id);

    const boundarySet = new Set(this.boundaryIds);
    const nonBoundary = nodes
      .filter((n) => !n.isBoundary)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, TOP_K)
      .map((n) => n.id);

    this.topkIds = [...this.boundaryIds, ...nonBoundary];
  }

  getVisibleIds(level: LODLevel, _camera?: THREE.Camera): string[] {
    switch (level) {
      case 'boundary':
        return this.boundaryIds;
      case 'topk':
        return this.topkIds;
      case 'full':
        return this.allIds;
    }
  }

  autoLevel(cameraDistance: number): LODLevel {
    if (cameraDistance > 20) return 'boundary';
    if (cameraDistance > 8) return 'topk';
    return 'full';
  }
}
