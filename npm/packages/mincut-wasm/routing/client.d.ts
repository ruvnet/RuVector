export interface RequestOptions { signal?: AbortSignal; timeoutMs?: number; budget?: number }
export interface RouteOptions extends RequestOptions { landmarks?: boolean }
export interface RoadRoute { cost: number; nodes: number[]; arcs: number[]; settled: number }
export interface RoadSnap { node: number; distanceM: number }
/** Abort/timeout terminates this worker and rejects all pending requests. */
export class RoutingWorker {
  private constructor(worker: unknown);
  static create(nodes: number, endpoints: Uint32Array, costs: Uint32Array, turns?: Uint32Array, options?: RequestOptions): Promise<RoutingWorker>;
  static createRuField(nodes: number, endpoints: Uint32Array, costs: Uint32Array, turns?: Uint32Array, policy?: RuFieldPolicy, options?: RequestOptions): Promise<RoutingWorker>;
  route(source: number, target: number, options?: RouteOptions): Promise<RoadRoute | undefined>;
  prepare(landmarks: Uint32Array, options?: RequestOptions): Promise<void>;
  /** 0xffffffff closes an arc; other costs are integers <= 1e9. */
  update(ids: Uint32Array, costs: Uint32Array, options?: RequestOptions): Promise<void>;
  setCoordinates(latLon: Float64Array, options?: RequestOptions): Promise<void>;
  nearest(lat: number, lon: number, radiusM: number, options?: RequestOptions): Promise<RoadSnap | undefined>;
  close(reason?: Error): void;
}
export interface RuFieldPolicy { maxPenalty?: number; closeAtMillionths?: number; ttlNs?: number; maxLatenessNs?: number }
export interface RuFieldUpdate { node: number; riskMillionths: number; changedArcs: number; duplicate: boolean }
export interface RoutingWorker {
  bindZone(zone: string, node: number, options?: RequestOptions): Promise<void>;
  bindCell(x: number, y: number, z: number, node: number, options?: RequestOptions): Promise<void>;
  /** verified must only be true after checking the detached RuField receipt. */
  ingestRuField(event: string | object, verified: boolean, nowNs: number, options?: RequestOptions): Promise<RuFieldUpdate>;
  expire(nowNs: number, options?: RequestOptions): Promise<number>;
}
