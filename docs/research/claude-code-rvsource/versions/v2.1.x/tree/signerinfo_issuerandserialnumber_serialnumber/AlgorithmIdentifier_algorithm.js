// Module: AlgorithmIdentifier_algorithm

/* original: PH4 */ var PrivateKeyInfo_PrivateKeyInfov=B((Pl_)=>{
  var Xl_=C_();
  wm();
  var CD=Xl_.asn1;
  Pl_.privateKeyValidator={
    name:"PrivateKeyInfo",tagClass:CD.Class.UNIVERSAL,type:CD.Type.SEQUENCE,constructed:!0,value:[{
      name:"PrivateKeyInfo.version",tagClass:CD.Class.UNIVERSAL,type:CD.Type.INTEGER,constructed:!1,capture:"privateKeyVersion"
    },{
      name:"PrivateKeyInfo.privateKeyAlgorithm",tagClass:CD.Class.UNIVERSAL,type:CD.Type.SEQUENCE,constructed:!0,value:[{
        name:"AlgorithmIdentifier.algorithm",tagClass:CD.Class.UNIVERSAL,type:CD.Type.OID,constructed:!1,capture:"privateKeyOid"
      }]
    },{
      name:"PrivateKeyInfo",tagClass:CD.Class.UNIVERSAL,type:CD.Type.OCTETSTRING,constructed:!1,capture:"privateKey"
    }]
  };
  Pl_.publicKeyValidator={
    name:"SubjectPublicKeyInfo",tagClass:CD.Class.UNIVERSAL,type:CD.Type.SEQUENCE,constructed:!0,captureAsn1:"subjectPublicKeyInfo",value:[{
      name:"SubjectPublicKeyInfo.AlgorithmIdentifier",tagClass:CD.Class.UNIVERSAL,type:CD.Type.SEQUENCE,constructed:!0,value:[{
        name:"AlgorithmIdentifier.algorithm",tagClass:CD.Class.UNIVERSAL,type:CD.Type.OID,constructed:!1,capture:"publicKeyOid"
      }]
    },{
      tagClass:CD.Class.UNIVERSAL,type:CD.Type.BITSTRING,constructed:!1,composed:!0,captureBitStringValue:"ed25519PublicKey"
    }]
  }
} /* confidence: 65% */

/* original: CD */ var CD=Xl_.asn1;

/* original: GH4 */ var composed_value=PH4(),fl_=composed_value.publicKeyValidator,Zl_=composed_value.privateKeyValidator; /* confidence: 30% */

