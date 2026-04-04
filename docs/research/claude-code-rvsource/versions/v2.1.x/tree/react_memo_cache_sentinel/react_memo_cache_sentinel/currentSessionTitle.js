// Module: currentSessionTitle

/* original: K */ let composed_value=oZ(q),_; /* confidence: 30% */

/* original: X */ let composed_value=Dz7(M); /* confidence: 30% */

/* original: _ */ let composed_value=$Y(); /* confidence: 30% */

/* original: _ */ let composed_value=$Y(),z=composed_value.getInternalEventReader(); /* confidence: 30% */

/* original: D */ let composed_value=0,f={
  onTransportPersistenceReady:(o,q6)=>{
    let t=++composed_value;
    (async()=>{
      try{
        let n=await Rq8();
        await AnK(o,q6,Object.keys(n))
      }catch(n){
        N(`[bridge:repl] Persistence sync failed: ${F6(n)}`,{
          level:"error"
        })
      }if(t!==composed_value){
        N("[bridge:repl] Transport torn down during sync â skipping writer install");
        return
      }XK8(o),N("[bridge:repl] Session persistence enabled â transcript entries forwarded as internal events")
    })()
  },onTransportPersistenceTeardown:()=>{
    composed_value++,qY7()
  }
}; /* confidence: 30% */

/* original: gJY */ function renametoaddaname_Version_repor(){
  let q=N8(),_=oZ(q)??_z.createElement(T,{
    dimColor:!0
  },"/rename to add a name");
  return[{
    label:"Version",value:{
      ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"
    }.VERSION
  },{
    label:"Session name",value:_
  },{
    label:"Session ID",value:q
  },{
    label:"cwd",value:Z8()
  },...Kx8(),..._x8()]
} /* confidence: 65% */

/* original: $Y */ function helper_fn(){
  if(!_M6){
    if(_M6=new ZQK,!jQK)gq(async()=>{
      await _M6?.flush();
      try{
        _M6?.reAppendSessionMetadata()
      }catch{
        
      }
    }),jQK=!0
  } /* confidence: 35% */

/* original: vxY */ function helper_fn(q){
  $Y().sessionFile=q
} /* confidence: 35% */

/* original: XK8 */ function helper_fn(q){
  $Y().setInternalEventWriter(q)
} /* confidence: 35% */

/* original: qY7 */ function helper_fn(){
  $Y().clearInternalEventWriter()
} /* confidence: 35% */

/* original: KY7 */ function helper_fn(q,K){
  $Y().setInternalEventReader(q),$Y().setInternalSubagentEventReader(K)
} /* confidence: 35% */

/* original: TxY */ function helper_fn(q){
  $Y().setRemoteIngressUrl(q)
} /* confidence: 35% */

/* original: Xd */ function helper_fn(q,K,_){
  await $Y().insertMessageChain(DK8(q),!0,K,_)
} /* confidence: 35% */

/* original: bp1 */ function helper_fn(q){
  await $Y().insertQueueOperation(q)
} /* confidence: 35% */

/* original: _Y7 */ function helper_fn(q){
  await $Y().removeMessageByUuid(q)
} /* confidence: 35% */

/* original: I88 */ function helper_fn(q,K,_){
  await $Y().insertFileHistorySnapshot(q,K,_)
} /* confidence: 35% */

/* original: kxY */ function helper_fn(q){
  await $Y().insertAttributionSnapshot(q)
} /* confidence: 35% */

/* original: aH6 */ function helper_fn(q,K){
  await $Y().insertContentReplacement(q,K)
} /* confidence: 35% */

/* original: jx */ function helper_fn(){
  $Y().resetSessionFile()
} /* confidence: 35% */

/* original: Mc */ function helper_fn(){
  let q=$Y();
   /* confidence: 35% */

/* original: VxY */ function helper_fn(q){
  let K=N8();
  if(!K)return;
  await $Y().appendEntry({
    type:"marble-origami-commit",sessionId:K,...q
  })
} /* confidence: 35% */

/* original: NxY */ function helper_fn(q){
  let K=N8();
  if(!K)return;
  await $Y().appendEntry({
    type:"marble-origami-snapshot",sessionId:K,...q
  })
} /* confidence: 35% */

/* original: ms */ function helper_fn(){
  await $Y().flush()
} /* confidence: 35% */

/* original: Dz7 */ function helper_fn(q){
  if(q===N8())return $Y().currentSessionTag;
   /* confidence: 35% */

/* original: oZ */ function helper_fn(q){
  if(q===N8())return $Y().currentSessionTitle;
   /* confidence: 35% */

/* original: wY7 */ function helper_fn(){
  return $Y().currentSessionAgentColor
} /* confidence: 35% */

/* original: Xc */ function helper_fn(q){
  let K=$Y();
   /* confidence: 35% */

/* original: Iq8 */ function helper_fn(){
  let q=$Y();
   /* confidence: 35% */

/* original: ig8 */ function helper_fn(){
  $Y().reAppendSessionMetadata()
} /* confidence: 35% */

/* original: PK8 */ function helper_fn(q){
  $Y().currentSessionAgentSetting=q
} /* confidence: 35% */

/* original: HY7 */ function helper_fn(q){
  $Y().currentSessionTitle=q,vQK.emit()
} /* confidence: 35% */

/* original: uxY */ function helper_fn(q){
  $Y().currentSessionMode=q
} /* confidence: 35% */

/* original: JY7 */ function helper_fn(q){
  $Y().currentSessionPermissionMode=q
} /* confidence: 35% */

/* original: ZQK */ class entity_class{
  currentSessionTag;
   /* confidence: 45% */

