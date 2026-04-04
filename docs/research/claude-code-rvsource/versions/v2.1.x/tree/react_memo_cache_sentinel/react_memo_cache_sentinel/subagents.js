// Module: subagents

/* original: C */ let Backgroundtasksdialogdismissed=kY(),F=[n8({
  content:B78(G,z,Backgroundtasksdialogdismissed,void 0,!1),isCompactSummary:!0,isVisibleInTranscriptOnly:!0
} /* confidence: 65% */

/* original: Y */ let composed_value=N8(),$=kY(); /* confidence: 30% */

/* original: Y */ let composed_value=N8(),$=kY(); /* confidence: 30% */

/* original: W */ let composed_value=kY(); /* confidence: 30% */

/* original: _ */ let composed_value=Df(q); /* confidence: 30% */

/* original: Y */ let composed_value=N8(),$=kY(); /* confidence: 30% */

/* original: O */ let composed_value=await PY7($); /* confidence: 30% */

/* original: R96 */ function utility_fn(){
  return G8.sessionProjectDir
} /* confidence: 40% */

/* original: kY */ function helper_fn(){
  let q=R96()??rj(z7());
   /* confidence: 35% */

/* original: Df */ function helper_fn(q){
  if(q===N8())return kY();
   /* confidence: 35% */

/* original: fW */ function subagents_subagents(q){
  let K=R96()??rj(z7()),_=N8(),z=tz7.get(q),Y=z?I0(K,_,"subagents",z):I0(K,_,"subagents");
   /* confidence: 65% */

/* original: WQK */ function helper_fn(q){
  return fW(q).replace(/\.jsonl$/,".meta.json")
} /* confidence: 35% */

/* original: DQK */ function helper_fn(){
  let q=R96()??rj(z7());
   /* confidence: 35% */

/* original: ez7 */ function helper_fn(q){
  return I0(DQK(),`remote-agent-${q}.meta.json`)
} /* confidence: 35% */

/* original: fs1 */ function helper_fn(q,K){
  let _=ez7(q);
  await QC6(MK8(_),{
    recursive:!0
  }),await dC6(_,JSON.stringify(K))
} /* confidence: 35% */

/* original: DxY */ function helper_fn(q){
  let K=ez7(q);
  try{
    let _=await nC6(K,"utf-8");
    return JSON.parse(_)
  }catch(_){
    if(S9(_))return null;
    throw _
  }
} /* confidence: 35% */

/* original: o88 */ function helper_fn(q){
  let K=ez7(q);
  try{
    await HxY(K)
  }catch(_){
    if(S9(_))return;
    throw _
  }
} /* confidence: 35% */

/* original: AY7 */ function helper_fn(q,K){
  Gf(Df(q),{
    type:"ai-title",aiTitle:K,sessionId:q
  } /* confidence: 35% */

/* original: xxY */ function helper_fn(q,K){
  Gf(Df(q),{
    type:"task-summary",summary:K,sessionId:q,timestamp:new Date().toISOString()
  } /* confidence: 35% */

/* original: TQK */ function helper_fn(q){
  let K=I0(R96()??rj(z7()),`${q}.jsonl`);
  return la(K)
} /* confidence: 35% */

/* original: Rq8 */ function named_entity(){
  let q=I0(R96()??rj(z7()),N8(),"subagents"),K;
  try{
    K=await lC6(q,{
      withFileTypes:!0
    })
  }catch{
    return{
      
    }
  }let _=K.filter((z)=>z.isFile()&&z.name.startsWith("agent-")&&z.name.endsWith(".jsonl")).map((z)=>z.name.slice(6,-6));
  return Ic8(_)
} /* confidence: 70% */

/* original: PY7 */ function ToolUseHandler(q){
  try{
    let K=kY(),{
      messages:_
    }=await la(K),z=null;
    for(let Y of _.values())if(Y.type==="assistant"){
      let $=Y.message.content;
      if(Array.isArray($)){
        for(let O of $)if(O.type==="tool_use"&&O.id===q){
          z=Y;
          break
        }
      }
    }else if(Y.type==="user"){
      let $=Y.message.content;
      if(Array.isArray($)){
        for(let O of $)if(O.type==="tool_result"&&O.tool_use_id===q)return null
      }
    }return z
  }catch{
    return null
  }
} /* confidence: 93% */

/* original: AnK */ function string_string_transcript_trans(q,K,_){
  let[z,Y]=await Promise.all([K.readMain(),K.readSubagents()]),$=new Set;
  for(let j of z??[]){
    let H=j.payload.uuid;
    if(typeof H==="string")$.add(H)
  }for(let j of Y??[]){
    let H=j.payload.uuid;
    if(typeof H==="string")$.add(H)
  }N(`[persistence-sync] Server has ${$.size} events since compaction`);
  let O=(j)=>{
    N(`[persistence-sync] Write failed: ${j}`)
  },A=await OnK(Df(N8()),$);
  for(let j of A)q("transcript",j,{
    ...pJ(j)&&{
      isCompaction:!0
    }
  }).catch(O);
  let w=0;
  for(let j of _){
    let H=await OnK(fW(j),$);
    for(let J of H)q("transcript",J,{
      ...pJ(J)&&{
        isCompaction:!0
      },agentId:j
    }).catch(O);
    w+=H.length
  }return N(`[persistence-sync] Uploaded ${A.length} main + ${w} subagent entries`),{
    uploadedMain:A.length,uploadedSubagents:w
  }
} /* confidence: 65% */

