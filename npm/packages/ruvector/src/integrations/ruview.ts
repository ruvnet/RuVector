import { IntegrationError, loadOptionalPackage } from './optional-package';

const PACKAGE = '@ruvnet/ruview';

/** Structured results are preserved, including honest negative and dry-run outcomes. */
export interface RuViewToolResult {
  ok: boolean;
  reason?: string;
  detail?: string;
  remedy?: string;
  dryRun?: boolean;
  [key: string]: unknown;
}

export interface RuViewTrainingReport {
  metric?: string;
  model_score: number;
  baseline_score?: number;
  split?: 'chronological' | 'blocked-gap' | 'grouped-subject' | 'grouped-session' | 'random-frame';
  train_subjects?: string[];
  test_subjects?: string[];
  train_end?: string;
  test_start?: string;
  n_test?: number;
  reproducer?: string;
  data?: 'real' | 'synthetic';
}

export interface RuViewGateResult extends RuViewToolResult {
  verdict: 'PASS' | 'FAIL';
  evidence: 'MEASURED' | 'CLAIMED' | 'SYNTHETIC' | null;
  deltaPp: number | null;
  findings: Array<{ severity: 'fail' | 'warn'; code: string; message: string }>;
  statement: string | null;
}

/** A dependency-free type surface over RuView's real SDK. Raw `call` retains
 * access to the device/operator registry without pretending to implement it. */
export interface RuViewIntegration {
  call(tool: string, args?: Record<string, unknown>): Promise<RuViewToolResult>;
  tools(): Array<{ name: string; description: string; inputSchema: object; annotations: object }>;
  doctor(options?: {
    groups?: Array<'runtime' | 'harness' | 'hosts' | 'repo' | 'rust' | 'python' | 'firmware' | 'serial' | 'devices' | 'remote' | 'sensing' | 'kernel'>;
    port?: string;
    probe_port?: boolean;
    sensing_url?: string;
  }): Promise<{
    ok: boolean;
    summary: { pass: number; warn: number; fail: number; skip: number };
    checks: Array<{ group: string; id: string; status: 'pass' | 'warn' | 'fail' | 'skip'; detail: string; remedy?: string }>;
    nextSteps: string[];
  }>;
  training: {
    modes: readonly string[];
    /** Evaluates a submitted report, not a new measurement or model execution. */
    gate(report: RuViewTrainingReport): Promise<RuViewGateResult>;
  };
  devices: {
    scan(): Promise<RuViewToolResult>;
    esp32(args?: {
      udp_port?: number; bind?: '0.0.0.0' | '127.0.0.1' | '::' | '::1'; seconds?: number;
      max_packets?: number; analyze?: boolean; node_id?: number;
      backend?: 'wasm' | 'napi' | 'auto'; analyze_max_frames?: number;
    }): Promise<RuViewToolResult>;
    mmwave(args: { port: string; model?: 'auto' | 'mr60bha2' | 'ld2410'; seconds?: number }): Promise<RuViewToolResult>;
    lidar(args: { source: 'rplidar'; port: string; baud?: 115200 | 256000 | 460800 | 921600; seconds?: number }
      | { source: 'iphone'; url: string; seconds?: number }): Promise<RuViewToolResult>;
  };
}

/** Explicitly load @ruvnet/ruview/sdk. Merely importing ruvector does not load
 * RuView, scan devices, start a server, or confirm a hardware operation.
 * `strict` is forwarded unchanged: upstream errors and confirmations remain in
 * effect. No operation is executed by this factory.
 */
export async function createRuViewIntegration(
  options: { strict?: boolean } = {},
): Promise<RuViewIntegration> {
  if (options.strict !== undefined && typeof options.strict !== 'boolean') {
    throw new TypeError('RuView strict must be a boolean');
  }
  const sdk = await loadOptionalPackage(PACKAGE, `${PACKAGE}/sdk`);
  if (typeof sdk.createRuView !== 'function') {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE,
      'Installed @ruvnet/ruview/sdk does not expose createRuView; tested with 0.9.1.');
  }
  const client = sdk.createRuView({ strict: options.strict });
  const methods = [client?.call, client?.tools, client?.doctor, client?.training?.gate,
    client?.devices?.scan, client?.devices?.esp32, client?.devices?.mmwave, client?.devices?.lidar];
  if (methods.some(method => typeof method !== 'function')) {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE,
      'Installed @ruvnet/ruview/sdk has an incompatible client API; tested with 0.9.1.');
  }
  return client as RuViewIntegration;
}
