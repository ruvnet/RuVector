// Module: default

/* original: qc6 */ var __esModule_____StringifiedUUID=B((Cvq)=>{
  Object.defineProperty(Cvq,"__esModule",{
    value:!0
  });
  Cvq.default=void 0;
  var tp9=ep9(ed6());
  function ep9(q){
    return q&&q.__esModule?q:{
      default:q
    }
  }var JZ=[];
  for(let q=0;
  q<256;
  ++q)JZ.push((q+256).toString(16).substr(1));
  function qB9(q,K=0){
    let _=(JZ[q[K+0]]+JZ[q[K+1]]+JZ[q[K+2]]+JZ[q[K+3]]+"-"+JZ[q[K+4]]+JZ[q[K+5]]+"-"+JZ[q[K+6]]+JZ[q[K+7]]+"-"+JZ[q[K+8]]+JZ[q[K+9]]+"-"+JZ[q[K+10]]+JZ[q[K+11]]+JZ[q[K+12]]+JZ[q[K+13]]+JZ[q[K+14]]+JZ[q[K+15]]).toLowerCase();
    if(!(0,tp9.default)(_))throw TypeError("Stringified UUID is invalid");
    return _
  }var KB9=qB9;
  Cvq.default=KB9
} /* confidence: 65% */

/* original: JZ */ var JZ=[];

/* original: KB9 */ var composed_value=qB9; /* confidence: 30% */

/* original: HB9 */ var composed_value=Fvq(qc6()),JB9=Fvq(GG1()); /* confidence: 30% */

/* original: z */ let composed_value=q.random||(q.rng||NB9.default)(); /* confidence: 30% */

/* original: UB9 */ var composed_value=hi(pvq()),QB9=hi(svq()),dB9=hi(KTq()),cB9=hi(wTq()),lB9=hi(JTq()),nB9=hi(PTq()),iB9=hi(ed6()),rB9=hi(qc6()),oB9=hi(GG1()); /* confidence: 30% */

/* original: qB9 */ function ____StringifiedUUIDisinvalid(q,K=0){
  let _=(JZ[q[K+0]]+JZ[q[K+1]]+JZ[q[K+2]]+JZ[q[K+3]]+"-"+JZ[q[K+4]]+JZ[q[K+5]]+"-"+JZ[q[K+6]]+JZ[q[K+7]]+"-"+JZ[q[K+8]]+JZ[q[K+9]]+"-"+JZ[q[K+10]]+JZ[q[K+11]]+JZ[q[K+12]]+JZ[q[K+13]]+JZ[q[K+14]]+JZ[q[K+15]]).toLowerCase();
  if(!(0,tp9.default)(_))throw TypeError("Stringified UUID is invalid");
  return _
} /* confidence: 65% */

