// Module: exports

/* original: C_ */ var C_=B((EsO,S24)=>{
  S24.exports={
    options:{
      usePureJavaScript:!1
    }
  }
}

/* original: fU */ var composed_value=B((IsO,i24)=>{
  var dV8=C_();
  i24.exports=dV8.md=dV8.md||{
    
  };
  dV8.md.algorithms=dV8.md.algorithms||{
    
  }
} /* confidence: 30% */

/* original: nk6 */ var string_Unknownhashalgorithm__s=B((usO,r24)=>{
  var or=C_();
  fU();
  D$();
  var OQ_=r24.exports=or.hmac=or.hmac||{
    
  };
  OQ_.create=function(){
    var q=null,K=null,_=null,z=null,Y={
      
    };
    return Y.start=function($,O){
      if($!==null)if(typeof $==="string")if($=$.toLowerCase(),$ in or.md.algorithms)K=or.md.algorithms[$].create();
      else throw Error('Unknown hash algorithm "'+$+'"');
      else K=$;
      if(O===null)O=q;
      else{
        if(typeof O==="string")O=or.util.createBuffer(O);
        else if(or.util.isArray(O)){
          var A=O;
          O=or.util.createBuffer();
          for(var w=0;
          w<A.length;
          ++w)O.putByte(A[w])
        }var j=O.length();
        if(j>K.blockLength)K.start(),K.update(O.bytes()),O=K.digest();
        _=or.util.createBuffer(),z=or.util.createBuffer(),j=O.length();
        for(var w=0;
        w<j;
        ++w){
          var A=O.at(w);
          _.putByte(54^A),z.putByte(92^A)
        }if(j<K.blockLength){
          var A=K.blockLength-j;
          for(var w=0;
          w<A;
          ++w)_.putByte(54),z.putByte(92)
        }q=O,_=_.bytes(),z=z.bytes()
      }K.start(),K.update(_)
    },Y.update=function($){
      K.update($)
    },Y.getMac=function(){
      var $=K.digest().bytes();
      return K.start(),K.update(z),K.update($),K.digest()
    },Y.digest=Y.getMac,Y
  }
} /* confidence: 65% */

/* original: dj4 */ var composed_value=B((tsO,Qj4)=>{
  var KN8=C_();
  II1();
  Qj4.exports=KN8.mgf=KN8.mgf||{
    
  };
  KN8.mgf.mgf1=KN8.mgf1
} /* confidence: 30% */

/* original: IH4 */ var composed_value=B((HtO,xH4)=>{
  xH4.exports=fU();
  lV8();
  ak6();
  ZI1();
  rI1()
} /* confidence: 30% */

/* original: UH4 */ var composed_value=B((XtO,FH4)=>{
  FH4.exports=C_();
  Jq6();
  jH4();
  wm();
  BV8();
  ar6();
  EH4();
  nk6();
  SH4();
  bH4();
  IH4();
  II1();
  oV8();
  oA6();
  LI1();
  pI1();
  pH4();
  gI1();
  RI1();
  GI1();
  _N8();
  gC();
  kI1();
  gH4();
  lI1();
  D$()
} /* confidence: 30% */

