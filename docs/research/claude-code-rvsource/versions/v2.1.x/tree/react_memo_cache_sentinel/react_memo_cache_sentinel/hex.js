// Module: hex

/* original: hU4 */ var __esModule_hex=B((EU4)=>{
  Object.defineProperty(EU4,"__esModule",{
    value:!0
  });
  EU4.RandomIdGenerator=void 0;
  var SGz=8,NU4=16;
  class yU4{
    generateTraceId=VU4(NU4);
    generateSpanId=VU4(SGz)
  }EU4.RandomIdGenerator=yU4;
  var xS8=Buffer.allocUnsafe(NU4);
  function VU4(q){
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
  }
} /* confidence: 65% */

/* original: SGz */ var SGz=8,NU4=16;

/* original: bGz */ var composed_value=hU4(); /* confidence: 30% */

