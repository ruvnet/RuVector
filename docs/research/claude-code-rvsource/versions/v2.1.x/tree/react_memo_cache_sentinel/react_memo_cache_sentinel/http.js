// Module: http

/* original: V_4 */ var V_4={
  
};

/* original: j */ let composed_value=await SIY(),H=!composed_value&&JS()!==void 0&&!X86(q.url); /* confidence: 30% */

/* original: SIY */ function 127001_http(){
  let{
    SandboxManager:q
  }=await Promise.resolve().then(() => (W$(),V_4));
  if(!q.isSandboxingEnabled())return;
  await q.waitForNetworkInitialization();
  let K=q.getProxyPort();
  if(!K)return;
  return{
    host:"127.0.0.1",port:K,protocol:"http"
  }
} /* confidence: 65% */

