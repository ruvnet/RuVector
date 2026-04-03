// Module: length

/* original: QT1 */ var __esModule_crypto=B((ohq)=>{
  Object.defineProperty(ohq,"__esModule",{
    value:!0
  });
  ohq.default=ca9;
  var Qa9=da9(U6("crypto"));
  function da9(q){
    return q&&q.__esModule?q:{
      default:q
    }
  }var Lf8=new Uint8Array(256),Ef8=Lf8.length;
  function ca9(){
    if(Ef8>Lf8.length-16)Qa9.default.randomFillSync(Lf8),Ef8=0;
    return Lf8.slice(Ef8,Ef8+=16)
  }
} /* confidence: 65% */

/* original: Lf8 */ var Lf8=new Uint8Array(256),Ef8=Lf8.length;

/* original: _s9 */ var composed_value=Ys9(QT1()),zs9=cc6(); /* confidence: 30% */

/* original: NRq */ var composed_value=yRq(VRq()),Rs9=yRq(QT1()),Ss9=cc6(); /* confidence: 30% */

/* original: ca9 */ function helper_fn(){
  if(Ef8>Lf8.length-16)Qa9.default.randomFillSync(Lf8),Ef8=0;
  return Lf8.slice(Ef8,Ef8+=16)
} /* confidence: 35% */

/* original: Cs9 */ function helper_fn(q,K,_){
  if(NRq.default.randomUUID&&!K&&!q)return NRq.default.randomUUID();
   /* confidence: 35% */

