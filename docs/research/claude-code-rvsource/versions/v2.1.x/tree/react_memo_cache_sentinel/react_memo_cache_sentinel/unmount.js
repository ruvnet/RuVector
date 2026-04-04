// Module: unmount

/* original: zi6 */ var zi6={
  
};

/* original: uc */ function helper_fn(q,K,_){
  return TtY(q,K,{
    color:"error",beforeExit:_
  })
} /* confidence: 35% */

/* original: TtY */ function InkRenderer(q,K,_){
  let{
    Text:z
  }=await Promise.resolve().then(() => (i6(),zi6)),Y=_?.color,$=_?.exitCode??1;
  q.render(Y?OE.default.createElement(z,{
    color:Y
  },K):OE.default.createElement(z,null,K)),q.unmount(),await _?.beforeExit?.(),process.exit($)
} /* confidence: 45% */

