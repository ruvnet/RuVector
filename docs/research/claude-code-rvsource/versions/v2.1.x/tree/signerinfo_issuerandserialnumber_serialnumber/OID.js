// Module: OID

/* original: xI1 */ var auth_handler=B((asO,Fj4)=>{
  var sk6=C_();
  wm();
  D$();
  var y4=sk6.asn1,tk6=Fj4.exports=sk6.pkcs7asn1=sk6.pkcs7asn1||{
    
  };
  sk6.pkcs7=sk6.pkcs7||{
    
  };
  sk6.pkcs7.asn1=tk6;
  var Bj4={
    name:"ContentInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"ContentInfo.ContentType",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"contentType"
    },{
      name:"ContentInfo.content",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,constructed:!0,optional:!0,captureAsn1:"content"
    }]
  };
  tk6.contentInfoValidator=Bj4;
  var gj4={
    name:"EncryptedContentInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"EncryptedContentInfo.contentType",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"contentType"
    },{
      name:"EncryptedContentInfo.contentEncryptionAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
        name:"EncryptedContentInfo.contentEncryptionAlgorithm.algorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"encAlgorithm"
      },{
        name:"EncryptedContentInfo.contentEncryptionAlgorithm.parameter",tagClass:y4.Class.UNIVERSAL,captureAsn1:"encParameter"
      }]
    },{
      name:"EncryptedContentInfo.encryptedContent",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,capture:"encryptedContent",captureAsn1:"encryptedContentAsn1"
    }]
  };
  tk6.envelopedDataValidator={
    name:"EnvelopedData",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"EnvelopedData.Version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"version"
    },{
      name:"EnvelopedData.RecipientInfos",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SET,constructed:!0,captureAsn1:"recipientInfos"
    }].concat(gj4)
  };
  tk6.encryptedDataValidator={
    name:"EncryptedData",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"EncryptedData.Version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"version"
    }].concat(gj4)
  };
  var Vc_={
    name:"SignerInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"SignerInfo.version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1
    },{
      name:"SignerInfo.issuerAndSerialNumber",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
        name:"SignerInfo.issuerAndSerialNumber.issuer",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,captureAsn1:"issuer"
      },{
        name:"SignerInfo.issuerAndSerialNumber.serialNumber",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"serial"
      }]
    },{
      name:"SignerInfo.digestAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
        name:"SignerInfo.digestAlgorithm.algorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"digestAlgorithm"
      },{
        name:"SignerInfo.digestAlgorithm.parameter",tagClass:y4.Class.UNIVERSAL,constructed:!1,captureAsn1:"digestParameter",optional:!0
      }]
    },{
      name:"SignerInfo.authenticatedAttributes",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,constructed:!0,optional:!0,capture:"authenticatedAttributes"
    },{
      name:"SignerInfo.digestEncryptionAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,capture:"signatureAlgorithm"
    },{
      name:"SignerInfo.encryptedDigest",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OCTETSTRING,constructed:!1,capture:"signature"
    },{
      name:"SignerInfo.unauthenticatedAttributes",tagClass:y4.Class.CONTEXT_SPECIFIC,type:1,constructed:!0,optional:!0,capture:"unauthenticatedAttributes"
    }]
  };
  tk6.signedDataValidator={
    name:"SignedData",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"SignedData.Version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"version"
    },{
      name:"SignedData.DigestAlgorithms",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SET,constructed:!0,captureAsn1:"digestAlgorithms"
    },Bj4,{
      name:"SignedData.Certificates",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,optional:!0,captureAsn1:"certificates"
    },{
      name:"SignedData.CertificateRevocationLists",tagClass:y4.Class.CONTEXT_SPECIFIC,type:1,optional:!0,captureAsn1:"crls"
    },{
      name:"SignedData.SignerInfos",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SET,capture:"signerInfos",optional:!0,value:[Vc_]
    }]
  };
  tk6.recipientInfoValidator={
    name:"RecipientInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"RecipientInfo.version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"version"
    },{
      name:"RecipientInfo.issuerAndSerial",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
        name:"RecipientInfo.issuerAndSerial.issuer",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,captureAsn1:"issuer"
      },{
        name:"RecipientInfo.issuerAndSerial.serialNumber",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"serial"
      }]
    },{
      name:"RecipientInfo.keyEncryptionAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
        name:"RecipientInfo.keyEncryptionAlgorithm.algorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"encAlgorithm"
      },{
        name:"RecipientInfo.keyEncryptionAlgorithm.parameter",tagClass:y4.Class.UNIVERSAL,constructed:!1,captureAsn1:"encParameter",optional:!0
      }]
    },{
      name:"RecipientInfo.encryptedKey",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OCTETSTRING,constructed:!1,capture:"encKey"
    }]
  }
} /* confidence: 95% */

/* original: y4 */ var y4=sk6.asn1,tk6=Fj4.exports=sk6.pkcs7asn1=sk6.pkcs7asn1||{
  
};

/* original: Bj4 */ var ContentInfo_ContentInfoContent={
  name:"ContentInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
    name:"ContentInfo.ContentType",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"contentType"
  },{
    name:"ContentInfo.content",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,constructed:!0,optional:!0,captureAsn1:"content"
  }]
}; /* confidence: 65% */

/* original: gj4 */ var EncryptedContentInfo_Encrypted={
  name:"EncryptedContentInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
    name:"EncryptedContentInfo.contentType",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"contentType"
  },{
    name:"EncryptedContentInfo.contentEncryptionAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"EncryptedContentInfo.contentEncryptionAlgorithm.algorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"encAlgorithm"
    },{
      name:"EncryptedContentInfo.contentEncryptionAlgorithm.parameter",tagClass:y4.Class.UNIVERSAL,captureAsn1:"encParameter"
    }]
  },{
    name:"EncryptedContentInfo.encryptedContent",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,capture:"encryptedContent",captureAsn1:"encryptedContentAsn1"
  }]
}; /* confidence: 65% */

/* original: Vc_ */ var auth_handler={
  name:"SignerInfo",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
    name:"SignerInfo.version",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1
  },{
    name:"SignerInfo.issuerAndSerialNumber",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"SignerInfo.issuerAndSerialNumber.issuer",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,captureAsn1:"issuer"
    },{
      name:"SignerInfo.issuerAndSerialNumber.serialNumber",tagClass:y4.Class.UNIVERSAL,type:y4.Type.INTEGER,constructed:!1,capture:"serial"
    }]
  },{
    name:"SignerInfo.digestAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,value:[{
      name:"SignerInfo.digestAlgorithm.algorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OID,constructed:!1,capture:"digestAlgorithm"
    },{
      name:"SignerInfo.digestAlgorithm.parameter",tagClass:y4.Class.UNIVERSAL,constructed:!1,captureAsn1:"digestParameter",optional:!0
    }]
  },{
    name:"SignerInfo.authenticatedAttributes",tagClass:y4.Class.CONTEXT_SPECIFIC,type:0,constructed:!0,optional:!0,capture:"authenticatedAttributes"
  },{
    name:"SignerInfo.digestEncryptionAlgorithm",tagClass:y4.Class.UNIVERSAL,type:y4.Type.SEQUENCE,constructed:!0,capture:"signatureAlgorithm"
  },{
    name:"SignerInfo.encryptedDigest",tagClass:y4.Class.UNIVERSAL,type:y4.Type.OCTETSTRING,constructed:!1,capture:"signature"
  },{
    name:"SignerInfo.unauthenticatedAttributes",tagClass:y4.Class.CONTEXT_SPECIFIC,type:1,constructed:!0,optional:!0,capture:"unauthenticatedAttributes"
  }]
}; /* confidence: 95% */

