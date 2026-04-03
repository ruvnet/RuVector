// Module: ENCRYPTED

/* original: gI1 */ var typed_entity=B((_tO,tj4)=>{
  var Dq6=C_();
  wm();
  Mq6();
  bI1();
  oA6();
  oV8();
  pI1();
  _N8();
  er6();
  D$();
  $N8();
  var BI1=Dq6.asn1,qV6=tj4.exports=Dq6.pki=Dq6.pki||{
    
  };
  qV6.pemToDer=function(q){
    var K=Dq6.pem.decode(q)[0];
    if(K.procType&&K.procType.type==="ENCRYPTED")throw Error("Could not convert PEM to DER; PEM is encrypted.");
    return Dq6.util.createBuffer(K.body)
  };
  qV6.privateKeyFromPem=function(q){
    var K=Dq6.pem.decode(q)[0];
    if(K.type!=="PRIVATE KEY"&&K.type!=="RSA PRIVATE KEY"){
      var _=Error('Could not convert private key from PEM; PEM header type is not "PRIVATE KEY" or "RSA PRIVATE KEY".');
      throw _.headerType=K.type,_
    }if(K.procType&&K.procType.type==="ENCRYPTED")throw Error("Could not convert private key from PEM; PEM is encrypted.");
    var z=BI1.fromDer(K.body);
    return qV6.privateKeyFromAsn1(z)
  };
  qV6.privateKeyToPem=function(q,K){
    var _={
      type:"RSA PRIVATE KEY",body:BI1.toDer(qV6.privateKeyToAsn1(q)).getBytes()
    };
    return Dq6.pem.encode(_,{
      maxline:K
    })
  };
  qV6.privateKeyInfoToPem=function(q,K){
    var _={
      type:"PRIVATE KEY",body:BI1.toDer(q).getBytes()
    };
    return Dq6.pem.encode(_,{
      maxline:K
    })
  }
} /* confidence: 70% */

/* original: BI1 */ var BI1=Dq6.asn1,qV6=tj4.exports=Dq6.pki=Dq6.pki||{
  
};

/* original: z */ var composed_value=BI1.fromDer(K.body); /* confidence: 30% */

