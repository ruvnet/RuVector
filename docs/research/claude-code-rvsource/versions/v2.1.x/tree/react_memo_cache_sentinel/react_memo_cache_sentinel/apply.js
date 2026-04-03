// Module: apply

/* original: HR5 */ var HR5=800,JR5=16,MR5,Xh7;

/* original: Ph7 */ var composed_value=L(()=>{
  MR5=Date.now;
  Xh7=XR5
} /* confidence: 30% */

/* original: XR5 */ function helper_fn(q){
  var K=0,_=0;
  return function(){
    var z=MR5(),Y=JR5-(z-_);
    if(_=z,Y>0){
      if(++K>=HR5)return arguments[0]
    }else K=0;
    return q.apply(void 0,arguments)
  }
} /* confidence: 35% */

