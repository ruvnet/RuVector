// Module: config_json

/* original: z */ let composed_value=QH(q); /* confidence: 30% */

/* original: Y */ let composed_value=ZL6(q); /* confidence: 30% */

/* original: $ */ let composed_value=jd(K),O=!_; /* confidence: 30% */

/* original: $ */ let composed_value=Y?.agentId,O=jd(K); /* confidence: 30% */

/* original: ZP6 */ function utility_fn(){
  return $M7(q7(),"teams")
} /* confidence: 40% */

/* original: ZL6 */ function helper_fn(q){
  return q88(ZP6(),uK6(q))
} /* confidence: 35% */

/* original: jd */ function config(q){
  return q88(ZL6(q),"config.json")
} /* confidence: 95% */

/* original: QH */ function helper_fn(q){
  try{
    let K=Ddz(jd(q),"utf-8");
    return l8(K)
  } /* confidence: 35% */

/* original: Ij6 */ function helper_fn(q,K){
  let _=ZL6(q);
   /* confidence: 35% */

/* original: mK6 */ function helper_fn(q,K){
  let _=ZL6(q);
  await Zdz(_,{
    recursive:!0
  }),await Gdz(jd(q),g6(K,null,2))
} /* confidence: 35% */

/* original: Tdz */ function helper_fn(q,K){
  let _=QH(q);
   /* confidence: 35% */

/* original: kdz */ function helper_fn(q,K){
  let _=QH(q);
   /* confidence: 35% */

/* original: Fo1 */ function helper_fn(q,K){
  let _=QH(q);
   /* confidence: 35% */

/* original: Uo1 */ function helper_fn(q,K){
  let _=QH(q);
   /* confidence: 35% */

/* original: LL6 */ function helper_fn(q,K,_){
  let z=QH(q);
   /* confidence: 35% */

/* original: do1 */ function helper_fn(q,K){
  let _=QH(q);
   /* confidence: 35% */

/* original: y3Y */ function helper_fn(q){
  if(!QH(q))return q;
  return oL6()
} /* confidence: 35% */

/* original: ncY */ function helper_fn(q,K,_){
  LL6(K,q,_);
   /* confidence: 35% */

/* original: KeK */ function helper_fn(){
  let q=pl6();
  if(!q?.teamName||!q?.agentName){
    N("[Reconnection] computeInitialTeamContext: No teammate context set (not a teammate)");
    return
  }let{
    teamName:K,agentId:_,agentName:z
  }=q,Y=QH(K);
  if(!Y){
    j6(Error(`[computeInitialTeamContext] Could not read team file for ${K}`));
    return
  }let $=jd(K),O=!_;
  return N(`[Reconnection] Computed initial team context for ${O?"leader":`teammate ${
    z
  }`} in team ${K}`),{
    teamName:K,teamFilePath:$,leadAgentId:Y.leadAgentId,selfAgentId:_,selfAgentName:z,isLeader:O,teammates:{
      
    }
  }
} /* confidence: 35% */

/* original: _eK */ function helper_fn(q,K,_){
  let z=QH(K);
   /* confidence: 35% */

