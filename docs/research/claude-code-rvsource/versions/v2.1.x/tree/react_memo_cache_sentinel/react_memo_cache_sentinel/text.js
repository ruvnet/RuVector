// Module: text

/* original: kEY */ function utility_fn(){
  return`thinkback@${m2}`
} /* confidence: 40% */

/* original: VEY */ function plugin_handler(){
  let q=jP(),K=kEY(),_=q.plugins[K];
  if(!_||_.length===0)return{
    type:"text",value:"Thinkback plugin not installed. Run /think-back first to install it."
  };
  let z=_[0];
  if(!z?.installPath)return{
    type:"text",value:"Thinkback plugin installation path not found."
  };
  let Y=vEY(z.installPath,"skills",TEY);
  return{
    type:"text",value:(await md8(Y)).message
  }
} /* confidence: 95% */

