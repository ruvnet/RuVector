// Module: writeUInt32BE

/* original: xS8 */ var xS8=Buffer.allocUnsafe(NU4);

/* original: VU4 */ function helper_fn(q){
  return function(){
    for(let _=0;
    _<q/4;
    _++)xS8.writeUInt32BE(Math.random()*4294967296>>>0,_*4);
    for(let _=0;
    _<q;
    _++)if(xS8[_]>0)break;
    else if(_===q-1)xS8[q-1]=1;
    return xS8.toString("hex",0,q)
  }
} /* confidence: 35% */

/* original: yU4 */ class entity_class{
  generateTraceId=VU4(NU4);
  generateSpanId=VU4(SGz)
} /* confidence: 45% */

