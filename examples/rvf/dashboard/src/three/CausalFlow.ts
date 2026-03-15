import * as THREE from 'three';

export interface FlowEdge {
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  weight: number;
}

const PARTICLES_PER_EDGE = 3;
const PARTICLE_COLOR = new THREE.Color(0x00E5FF);

interface ParticleState {
  edgeIndex: number;
  phase: number;
  speed: number;
}

export class CausalFlow {
  private scene: THREE.Scene;
  private points: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.PointsMaterial | null = null;
  private edges: FlowEdge[] = [];
  private particles: ParticleState[] = [];

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  setEdges(edges: FlowEdge[]): void {
    this.dispose();
    this.edges = edges;

    const count = edges.length * PARTICLES_PER_EDGE;
    if (count === 0) return;

    this.particles = [];
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i];
      for (let p = 0; p < PARTICLES_PER_EDGE; p++) {
        const phase = p / PARTICLES_PER_EDGE;
        const speed = 0.3 + edge.weight * 0.7;
        this.particles.push({ edgeIndex: i, phase, speed });

        const idx = (i * PARTICLES_PER_EDGE + p) * 3;
        const src = edge.sourcePos;
        const tgt = edge.targetPos;
        positions[idx] = src.x + (tgt.x - src.x) * phase;
        positions[idx + 1] = src.y + (tgt.y - src.y) * phase;
        positions[idx + 2] = src.z + (tgt.z - src.z) * phase;
      }
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

    this.material = new THREE.PointsMaterial({
      color: PARTICLE_COLOR,
      size: 0.15,
      transparent: true,
      opacity: 0.85,
      sizeAttenuation: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.scene.add(this.points);
  }

  animate(dt: number): void {
    if (!this.geometry || this.edges.length === 0) return;

    const positions = this.geometry.attributes.position as THREE.BufferAttribute;
    const arr = positions.array as Float32Array;

    for (let i = 0; i < this.particles.length; i++) {
      const ps = this.particles[i];
      ps.phase = (ps.phase + ps.speed * dt) % 1;

      const edge = this.edges[ps.edgeIndex];
      const src = edge.sourcePos;
      const tgt = edge.targetPos;
      const t = ps.phase;

      const idx = i * 3;
      arr[idx] = src.x + (tgt.x - src.x) * t;
      arr[idx + 1] = src.y + (tgt.y - src.y) * t;
      arr[idx + 2] = src.z + (tgt.z - src.z) * t;
    }

    positions.needsUpdate = true;
  }

  dispose(): void {
    if (this.points) {
      this.scene.remove(this.points);
    }
    if (this.geometry) {
      this.geometry.dispose();
      this.geometry = null;
    }
    if (this.material) {
      this.material.dispose();
      this.material = null;
    }
    this.points = null;
    this.particles = [];
    this.edges = [];
  }
}
