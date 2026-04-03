// Module: seenAt

/* original: _ZY */ var plugin_handler="flagged-plugins.json",zZY=172800000,dy=null; /* confidence: 95% */

/* original: QhK */ function helper_fn(){
  return KZY(BX(),_ZY)
} /* confidence: 35% */

/* original: CQ8 */ function helper_fn(){
  try{
    let q=await sfY(QhK(),{
      encoding:"utf-8"
    });
    return YZY(q)
  }catch{
    return{
      
    }
  }
} /* confidence: 35% */

/* original: dhK */ function helper_fn(){
  let q=await CQ8(),K=Date.now(),_=!1;
  for(let[z,Y]of Object.entries(q))if(Y.seenAt&&K-new Date(Y.seenAt).getTime()>=zZY)delete q[z],_=!0;
  if(dy=q,_)await bQ8(q)
} /* confidence: 35% */

/* original: lhK */ function helper_fn(q){
  if(dy===null)dy=await CQ8();
  let K=new Date().toISOString(),_=!1,z={
    ...dy
  };
  for(let Y of q){
    let $=z[Y];
    if($&&!$.seenAt)z[Y]={
      ...$,seenAt:K
    },_=!0
  }if(_)await bQ8(z)
} /* confidence: 35% */

/* original: nhK */ function helper_fn(q){
  if(dy===null)dy=await CQ8();
  if(!(q in dy))return;
  let{
    [q]:K,..._
  }=dy;
  dy=_,await bQ8(_)
} /* confidence: 35% */

