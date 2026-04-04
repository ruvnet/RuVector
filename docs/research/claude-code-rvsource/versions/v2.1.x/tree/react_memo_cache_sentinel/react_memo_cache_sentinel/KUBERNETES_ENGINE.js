// Module: KUBERNETES_ENGINE

/* original: Dk1 */ var __esModule_APP_ENGINE_KUBERNET=B((fCq)=>{
  Object.defineProperty(fCq,"__esModule",{
    value:!0
  });
  fCq.GCPEnv=void 0;
  fCq.clear=ge9;
  fCq.getEnv=Fe9;
  var DCq=rc6(),di;
  (function(q){
    q.APP_ENGINE="APP_ENGINE",q.KUBERNETES_ENGINE="KUBERNETES_ENGINE",q.CLOUD_FUNCTIONS="CLOUD_FUNCTIONS",q.COMPUTE_ENGINE="COMPUTE_ENGINE",q.CLOUD_RUN="CLOUD_RUN",q.NONE="NONE"
  })(di||(fCq.GCPEnv=di={
    
  }));
  var tc6;
  function ge9(){
    tc6=void 0
  }async function Fe9(){
    if(tc6)return tc6;
    return tc6=Ue9(),tc6
  }async function Ue9(){
    let q=di.NONE;
    if(Qe9())q=di.APP_ENGINE;
    else if(de9())q=di.CLOUD_FUNCTIONS;
    else if(await ne9())if(await le9())q=di.KUBERNETES_ENGINE;
    else if(ce9())q=di.CLOUD_RUN;
    else q=di.COMPUTE_ENGINE;
    else q=di.NONE;
    return q
  }function Qe9(){
    return!!(process.env.GAE_SERVICE||process.env.GAE_MODULE_NAME)
  }function de9(){
    return!!(process.env.FUNCTION_NAME||process.env.FUNCTION_TARGET)
  }function ce9(){
    return!!process.env.K_CONFIGURATION
  }async function le9(){
    try{
      return await DCq.instance("attributes/cluster-name"),!0
    }catch(q){
      return!1
    }
  }async function ne9(){
    return DCq.isAvailable()
  }
} /* confidence: 65% */

/* original: S8_ */ var composed_value=Dk1(); /* confidence: 30% */

/* original: de9 */ function utility_fn(){
  return!!(process.env.FUNCTION_NAME||process.env.FUNCTION_TARGET)
} /* confidence: 40% */

