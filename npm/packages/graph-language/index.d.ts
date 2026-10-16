export type Effect = 'graph.read' | 'model.predict' | 'decision.emit';
export type Operation =
  | {op:'nodes';label:string}
  | {op:'select';ids:string[]}
  | {op:'traverse';input:string;relation:string;depth:number}
  | {op:'similar';input:string;vector:number[];k:number}
  | {op:'observe';input:string;predicate:string;within:number}
  | {op:'infer';inputs:string[];rules:string[]}
  | {op:'predict';input:string;relation:string;model:string;k:number}
  | {op:'assess';input:string};
export interface Program {version:1;name:string;effects:Effect[];budget:number;steps:({bind:string}&Operation)[];emit:string}
export interface Node {id:string;label:string;properties?:Record<string,unknown>;vector?:number[]|null}
export interface Edge {id:string;source:string;target:string;relation:string}
export interface Observation {id:string;subject:string;predicate:string;value:boolean;source:string;observed_at:number;expires_at:number}
export interface Condition {predicate:string;value:boolean}
export interface Rule {id:string;requires:Condition[];conclusion:Condition}
export interface Snapshot {version:1;revision:string;nodes:Node[];edges:Edge[];observations:Observation[];rules?:Rule[]}
export interface Context {now:number;capabilities:Effect[];max_steps:number;max_results:number}
export interface RunRequest {program:Program;snapshot:Snapshot;context:Context}
export interface Evidence {subject:string;predicate:string;value:boolean;kind:'observed'|'derived';observation_ids:string[];sources:string[];rule_ids:string[];expires_at:number}
export interface Decision {subject:string;predicate:string;status:'supported'|'refuted'|'unknown'|'conflicting';evidence:Evidence[]}
export interface Prediction {subject:string;relation:string;object:string;score:number}
export interface PredictionBatch {model:string;revision:string;model_hash:string;exact:boolean;work:number;candidates:Prediction[]}
export type Output = {kind:'decisions';items:Decision[]}|{kind:'predictions';items:PredictionBatch};
export interface Receipt {version:1;program_hash:string;snapshot_hash:string;snapshot_revision:string;context_hash:string;now:number;work:number;output:Output;receipt_hash:string}
/** All integer values must stay within JavaScript's exact safe integer range. */
export class RuGraphError extends Error {
  constructor(code:string,message:string,position?:number);
  code:string;position?:number;
}
export class RuGraph {
  constructor(options?:{binary?:string;timeoutMs?:number});
  check(source:string):Promise<{valid:true;name:string;version:1}>;
  compile(source:string):Promise<Program>;
  run(request:RunRequest,options?:{modelsPath?:string}):Promise<Receipt>;
  audit(request:RunRequest,options:AuditOptions):Promise<AuditReport>;
  replay(request:RunRequest,receipt:Receipt,options?:{modelsPath?:string}):Promise<Receipt>;
}

export type Intervention = {kind:'observation'|'source'|'edge';id:string};
export interface AuditOptions {interventions:Intervention[];max_execution_work:number}
export interface AuditChange {subject:string;predicate:string;before:Decision['status']|null;after:Decision['status']|null}
export interface AuditTrial {intervention:Intervention;removed_records:number;receipt_hash:string;snapshot_hash:string;work:number;changes:AuditChange[];changed_decisions:number}
export interface AuditReport {version:1;baseline_receipt_hash:string;program_hash:string;snapshot_hash:string;context_hash:string;reserved_execution_work:number;execution_work:number;baseline_decisions:number;trials:AuditTrial[]}
