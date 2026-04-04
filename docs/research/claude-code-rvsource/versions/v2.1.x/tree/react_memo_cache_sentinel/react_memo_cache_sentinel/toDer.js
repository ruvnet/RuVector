// Module: toDer

/* original: z8 */ var z8=Z5.asn1,Uq=oj4.exports=Z5.pki=Z5.pki||{
  
}

/* original: nj4 */ var Certificate_CertificateTBSCert=Z5.pki.rsa.publicKeyValidator,Ec_={
  name:"Certificate",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,value:[{
    name:"Certificate.TBSCertificate",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,captureAsn1:"tbsCertificate",value:[{
      name:"Certificate.TBSCertificate.version",tagClass:z8.Class.CONTEXT_SPECIFIC,type:0,constructed:!0,optional:!0,value:[{
        name:"Certificate.TBSCertificate.version.integer",tagClass:z8.Class.UNIVERSAL,type:z8.Type.INTEGER,constructed:!1,capture:"certVersion"
      }]
    },{
      name:"Certificate.TBSCertificate.serialNumber",tagClass:z8.Class.UNIVERSAL,type:z8.Type.INTEGER,constructed:!1,capture:"certSerialNumber"
    },{
      name:"Certificate.TBSCertificate.signature",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,value:[{
        name:"Certificate.TBSCertificate.signature.algorithm",tagClass:z8.Class.UNIVERSAL,type:z8.Type.OID,constructed:!1,capture:"certinfoSignatureOid"
      },{
        name:"Certificate.TBSCertificate.signature.parameters",tagClass:z8.Class.UNIVERSAL,optional:!0,captureAsn1:"certinfoSignatureParams"
      }]
    },{
      name:"Certificate.TBSCertificate.issuer",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,captureAsn1:"certIssuer"
    },{
      name:"Certificate.TBSCertificate.validity",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,value:[{
        name:"Certificate.TBSCertificate.validity.notBefore (utc)",tagClass:z8.Class.UNIVERSAL,type:z8.Type.UTCTIME,constructed:!1,optional:!0,capture:"certValidity1UTCTime"
      },{
        name:"Certificate.TBSCertificate.validity.notBefore (generalized)",tagClass:z8.Class.UNIVERSAL,type:z8.Type.GENERALIZEDTIME,constructed:!1,optional:!0,capture:"certValidity2GeneralizedTime"
      },{
        name:"Certificate.TBSCertificate.validity.notAfter (utc)",tagClass:z8.Class.UNIVERSAL,type:z8.Type.UTCTIME,constructed:!1,optional:!0,capture:"certValidity3UTCTime"
      },{
        name:"Certificate.TBSCertificate.validity.notAfter (generalized)",tagClass:z8.Class.UNIVERSAL,type:z8.Type.GENERALIZEDTIME,constructed:!1,optional:!0,capture:"certValidity4GeneralizedTime"
      }]
    },{
      name:"Certificate.TBSCertificate.subject",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,captureAsn1:"certSubject"
    },Certificate_CertificateTBSCert,{
      name:"Certificate.TBSCertificate.issuerUniqueID",tagClass:z8.Class.CONTEXT_SPECIFIC,type:1,constructed:!0,optional:!0,value:[{
        name:"Certificate.TBSCertificate.issuerUniqueID.id",tagClass:z8.Class.UNIVERSAL,type:z8.Type.BITSTRING,constructed:!1,captureBitStringValue:"certIssuerUniqueId"
      }]
    },{
      name:"Certificate.TBSCertificate.subjectUniqueID",tagClass:z8.Class.CONTEXT_SPECIFIC,type:2,constructed:!0,optional:!0,value:[{
        name:"Certificate.TBSCertificate.subjectUniqueID.id",tagClass:z8.Class.UNIVERSAL,type:z8.Type.BITSTRING,constructed:!1,captureBitStringValue:"certSubjectUniqueId"
      }]
    },{
      name:"Certificate.TBSCertificate.extensions",tagClass:z8.Class.CONTEXT_SPECIFIC,type:3,constructed:!0,captureAsn1:"certExtensions",optional:!0
    }]
  },{
    name:"Certificate.signatureAlgorithm",tagClass:z8.Class.UNIVERSAL,type:z8.Type.SEQUENCE,constructed:!0,value:[{
      name:"Certificate.signatureAlgorithm.algorithm",tagClass:z8.Class.UNIVERSAL,type:z8.Type.OID,constructed:!1,capture:"certSignatureOid"
    },{
      name:"Certificate.TBSCertificate.signature.parameters",tagClass:z8.Class.UNIVERSAL,optional:!0,captureAsn1:"certSignatureParams"
    }]
  },{
    name:"Certificate.signatureValue",tagClass:z8.Class.UNIVERSAL,type:z8.Type.BITSTRING,constructed:!1,captureBitStringValue:"certSignature"
  }]
} /* confidence: 65% */

/* original: $ */ var composed_value=z8.fromDer(z.body,_); /* confidence: 30% */

/* original: z */ var composed_value=z8.fromDer(K.body); /* confidence: 30% */

/* original: $ */ var composed_value=z8.fromDer(z.body,_); /* confidence: 30% */

/* original: $ */ var composed_value=z8.toDer(q.tbsCertificate); /* confidence: 30% */

/* original: A */ var headers=K.tbsCertificate||Uq.getTBSCertificate(K),w=z8.toDer(headers); /* confidence: 70% */

/* original: $ */ var composed_value=z8.derToOid(_.publicKeyOid); /* confidence: 30% */

/* original: j */ var composed_value=z8.toDer(O.tbsCertificate); /* confidence: 30% */

/* original: H */ var composed_value=Z5.md.sha1.create(),J=z8.toDer(_.certIssuer); /* confidence: 30% */

/* original: M */ var composed_value=Z5.md.sha1.create(),X=z8.toDer(_.certSubject); /* confidence: 30% */

/* original: $ */ var composed_value=z8.derToOid(_.publicKeyOid); /* confidence: 30% */

/* original: A */ var headers=z8.toDer(O.certificationRequestInfo); /* confidence: 70% */

/* original: $ */ var composed_value=z8.toDer(q.certificationRequestInfo); /* confidence: 30% */

/* original: z */ var composed_value=q.certificationRequestInfo||Uq.getCertificationRequestInfo(q),Y=z8.toDer(composed_value); /* confidence: 30% */

/* original: O */ var composed_value=_.value,A=z8.Type.PRINTABLESTRING; /* confidence: 30% */

/* original: A */ var headers=q.value.value,W=z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,[]),D=z8.create(z8.Class.CONTEXT_SPECIFIC,0,!0,[]),j; /* confidence: 70% */

/* original: w */ var composed_value=z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.OID,!1,z8.oidToDer(Y.type).getBytes()),z8.create(z8.Class.UNIVERSAL,z8.Type.SET,!0,[z8.create(z8.Class.UNIVERSAL,O,A,$)])]); /* confidence: 30% */

/* original: w */ var composed_value=z8.toDer(Uq.certificateToAsn1(O)).getBytes(); /* confidence: 30% */

/* original: j */ var composed_value=z8.toDer(Uq.certificateToAsn1(O)).getBytes(); /* confidence: 30% */

/* original: uI1 */ function helper_fn(q,K){
  switch(q){
    case rY["RSASSA-PSS"]:var _=[];
    if(K.hash.algorithmOid!==void 0)_.push(z8.create(z8.Class.CONTEXT_SPECIFIC,0,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.OID,!1,z8.oidToDer(K.hash.algorithmOid).getBytes()),z8.create(z8.Class.UNIVERSAL,z8.Type.NULL,!1,"")])]));
    if(K.mgf.algorithmOid!==void 0)_.push(z8.create(z8.Class.CONTEXT_SPECIFIC,1,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.OID,!1,z8.oidToDer(K.mgf.algorithmOid).getBytes()),z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.OID,!1,z8.oidToDer(K.mgf.hash.algorithmOid).getBytes()),z8.create(z8.Class.UNIVERSAL,z8.Type.NULL,!1,"")])])]));
    if(K.saltLength!==void 0)_.push(z8.create(z8.Class.CONTEXT_SPECIFIC,2,!0,[z8.create(z8.Class.UNIVERSAL,z8.Type.INTEGER,!1,z8.integerToDer(K.saltLength).getBytes())]));
    return z8.create(z8.Class.UNIVERSAL,z8.Type.SEQUENCE,!0,_);
    default:return z8.create(z8.Class.UNIVERSAL,z8.Type.NULL,!1,"")
  } /* confidence: 35% */

/* original: Sc_ */ function helper_fn(q){
  var K=z8.create(z8.Class.CONTEXT_SPECIFIC,0,!0,[]);
   /* confidence: 35% */

