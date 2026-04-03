// Module: useRef

/* original: _5 */ function utility_fn(){
  return G8.isRemoteMode
} /* confidence: 40% */

/* original: Cx */ function UseRefHook(q){
  let{
    addNotification:K
  }=JK(),_=m58.useRef(!1),z=m58.useRef(q);
  z.current=q,m58.useEffect(()=>{
    if(_5()||_.current)return;
    _.current=!0,Promise.resolve().then(()=>z.current()).then((Y)=>{
      if(!Y)return;
      for(let $ of Array.isArray(Y)?Y:[Y])K($)
    }).catch(j6)
  },[K])
} /* confidence: 46% */

/* original: V55 */ function helper_fn(){
  if(_5())return;
   /* confidence: 35% */

