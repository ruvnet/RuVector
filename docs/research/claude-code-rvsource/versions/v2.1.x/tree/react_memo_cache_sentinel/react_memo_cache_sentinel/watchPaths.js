// Module: watchPaths

/* original: oK6 */ var oK6=void 0,h_K,xu8;

/* original: aL6 */ function env_config(){
  N("Invalidating session environment cache"),oK6=void 0
} /* confidence: 95% */

/* original: GdK */ function helper_fn(q,K){
  let _=await Vx({
    hookInput:q,timeoutMs:K
  });
  if(_.length>0)aL6();
  let z=_.flatMap(($)=>$.watchPaths??[]),Y=_.map(($)=>$.systemMessage).filter(($)=>!!$);
  return{
    results:_,watchPaths:z,systemMessages:Y
  }
} /* confidence: 35% */

