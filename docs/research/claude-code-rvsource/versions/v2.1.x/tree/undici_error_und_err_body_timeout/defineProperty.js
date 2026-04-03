// Module: defineProperty

/* original: G28 */ var undiciglobalDispatcher1_functi=B((nd$,$F7)=>{
  var _F7=Symbol.for("undici.globalDispatcher.1"),{
    InvalidArgumentError:Hd5
  }=t$(),Jd5=vf6();
  if(YF7()===void 0)zF7(new Jd5);
  function zF7(q){
    if(!q||typeof q.dispatch!=="function")throw new Hd5("Argument agent must implement Agent");
    Object.defineProperty(globalThis,_F7,{
      value:q,writable:!0,enumerable:!1,configurable:!1
    })
  }function YF7(){
    return globalThis[_F7]
  }$F7.exports={
    setGlobalDispatcher:zF7,getGlobalDispatcher:YF7
  }
} /* confidence: 65% */

/* original: YF7 */ function utility_fn(){
  return globalThis[_F7]
} /* confidence: 40% */

