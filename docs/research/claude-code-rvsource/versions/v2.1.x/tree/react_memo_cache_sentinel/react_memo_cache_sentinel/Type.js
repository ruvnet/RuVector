// Module: Type

/* original: q1 */ var q1=X4.asn1,Ok=mH4.exports=X4.pkcs7=X4.pkcs7||{
  
};

/* original: z */ var composed_value=q1.fromDer(K.body); /* confidence: 30% */

/* original: Y */ var composed_value=q1.derToOid(K.contentType),$; /* confidence: 30% */

/* original: $ */ var composed_value=[],O=q1.create(q1.Class.CONTEXT_SPECIFIC,0,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.INTEGER,!1,q1.integerToDer(q.version).getBytes()),q1.create(q1.Class.UNIVERSAL,q1.Type.SET,!0,q.digestAlgorithmIdentifiers),q.contentInfo])]); /* confidence: 30% */

/* original: $ */ var composed_value=q1.derToOid(q.contentInfo.value[0].value),O=q1.toDer(Y); /* confidence: 30% */

/* original: J */ var composed_value=q1.create(q1.Class.UNIVERSAL,q1.Type.SET,!0,[]); /* confidence: 30% */

/* original: O */ var composed_value=q1.derToOid(z.contentType); /* confidence: 30% */

/* original: K */ function composed_value(){
  var z={
    
  };
  for(var Y=0;
  Y<q.signers.length;
  ++Y){
    var $=q.signers[Y],O=$.digestAlgorithm;
    if(!(O in z))z[O]=X4.md[X4.pki.oids[O]].create();
    if($.authenticatedAttributes.length===0)$.md=z[O];
    else $.md=X4.md[X4.pki.oids[O]].create()
  }q.digestAlgorithmIdentifiers=[];
  for(var O in z)q.digestAlgorithmIdentifiers.push(q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.OID,!1,q1.oidToDer(O).getBytes()),q1.create(q1.Class.UNIVERSAL,q1.Type.NULL,!1,"")]));
  return z
} /* confidence: 30% */

/* original: Sl_ */ function value_holder(q){
  var K={
    
  },_=[];
  if(!q1.validate(q,Ok.asn1.recipientInfoValidator,K,_)){
    var z=Error("Cannot read PKCS#7 RecipientInfo. ASN.1 object is not an PKCS#7 RecipientInfo.");
    throw z.errors=_,z
  }return{
    version:K.version.charCodeAt(0),issuer:X4.pki.RDNAttributesAsArray(K.issuer),serialNumber:X4.util.createBuffer(K.serial).toHex(),encryptedContent:{
      algorithm:q1.derToOid(K.encAlgorithm),parameter:K.encParameter?K.encParameter.value:void 0,content:K.encKey
    }
  }
} /* confidence: 70% */

/* original: Cl_ */ function McpServerConfig(q){
  return q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.INTEGER,!1,q1.integerToDer(q.version).getBytes()),q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[X4.pki.distinguishedNameToAsn1({
    attributes:q.issuer
  } /* confidence: 31% */

/* original: Il_ */ function McpServerConfig(q){
  var K=q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.INTEGER,!1,q1.integerToDer(q.version).getBytes()),q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[X4.pki.distinguishedNameToAsn1({
    attributes:q.issuer
  } /* confidence: 31% */

/* original: ml_ */ function WriteTool(q){
  return[q1.create(q1.Class.UNIVERSAL,q1.Type.OID,!1,q1.oidToDer(X4.pki.oids.data).getBytes()),q1.create(q1.Class.UNIVERSAL,q1.Type.SEQUENCE,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.OID,!1,q1.oidToDer(q.algorithm).getBytes()),!q.parameter?void 0:q1.create(q1.Class.UNIVERSAL,q1.Type.OCTETSTRING,!1,q.parameter.getBytes())]),q1.create(q1.Class.CONTEXT_SPECIFIC,0,!0,[q1.create(q1.Class.UNIVERSAL,q1.Type.OCTETSTRING,!1,q.content.getBytes())])]
} /* confidence: 31% */

