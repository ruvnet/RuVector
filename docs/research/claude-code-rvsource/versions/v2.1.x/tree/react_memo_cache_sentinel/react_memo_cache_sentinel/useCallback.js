// Module: useCallback

/* original: z */ let composed_value=ub6.useCallback(async()=>{
  if(!q)return;
  try{
    HK8();
    let Y=await w0(q);
    K(Y)
  }catch(Y){
    if(Y instanceof Error)j6(Y)
  }
} /* confidence: 30% */

/* original: HK8 */ function utility_fn(){
  OQK.cache?.clear?.(),rC.cache?.clear?.(),jw6.cache?.clear?.(),YxY?.()
} /* confidence: 40% */

/* original: leK */ function helper_fn(q,K){
  let _=ub6.useCallback(async()=>{
    if(!q)return;
    try{
      Id();
      let Y=await w0(q);
      K(Y)
    }catch(Y){
      if(Y instanceof Error)j6(Y)
    }
  },[q,K]);
  ub6.useEffect(()=>Ib6.subscribe(_),[_]);
  let z=ub6.useCallback(async()=>{
    if(!q)return;
    try{
      HK8();
      let Y=await w0(q);
      K(Y)
    }catch(Y){
      if(Y instanceof Error)j6(Y)
    }
  },[q,K]);
  ub6.useEffect(()=>cl6(z),[z])
} /* confidence: 35% */

