// Module: policyConstraints

/* original: Mq6 */ var env_config=B((bsO,c24)=>{
  var ir6=C_();
  ir6.pki=ir6.pki||{
    
  };
  var WI1=c24.exports=ir6.pki.oids=ir6.oids=ir6.oids||{
    
  };
  function qq(q,K){
    WI1[q]=K,WI1[K]=q
  }function gO(q,K){
    WI1[q]=K
  }qq("1.2.840.113549.1.1.1","rsaEncryption");
  qq("1.2.840.113549.1.1.4","md5WithRSAEncryption");
  qq("1.2.840.113549.1.1.5","sha1WithRSAEncryption");
  qq("1.2.840.113549.1.1.7","RSAES-OAEP");
  qq("1.2.840.113549.1.1.8","mgf1");
  qq("1.2.840.113549.1.1.9","pSpecified");
  qq("1.2.840.113549.1.1.10","RSASSA-PSS");
  qq("1.2.840.113549.1.1.11","sha256WithRSAEncryption");
  qq("1.2.840.113549.1.1.12","sha384WithRSAEncryption");
  qq("1.2.840.113549.1.1.13","sha512WithRSAEncryption");
  qq("1.3.101.112","EdDSA25519");
  qq("1.2.840.10040.4.3","dsa-with-sha1");
  qq("1.3.14.3.2.7","desCBC");
  qq("1.3.14.3.2.26","sha1");
  qq("1.3.14.3.2.29","sha1WithRSASignature");
  qq("2.16.840.1.101.3.4.2.1","sha256");
  qq("2.16.840.1.101.3.4.2.2","sha384");
  qq("2.16.840.1.101.3.4.2.3","sha512");
  qq("2.16.840.1.101.3.4.2.4","sha224");
  qq("2.16.840.1.101.3.4.2.5","sha512-224");
  qq("2.16.840.1.101.3.4.2.6","sha512-256");
  qq("1.2.840.113549.2.2","md2");
  qq("1.2.840.113549.2.5","md5");
  qq("1.2.840.113549.1.7.1","data");
  qq("1.2.840.113549.1.7.2","signedData");
  qq("1.2.840.113549.1.7.3","envelopedData");
  qq("1.2.840.113549.1.7.4","signedAndEnvelopedData");
  qq("1.2.840.113549.1.7.5","digestedData");
  qq("1.2.840.113549.1.7.6","encryptedData");
  qq("1.2.840.113549.1.9.1","emailAddress");
  qq("1.2.840.113549.1.9.2","unstructuredName");
  qq("1.2.840.113549.1.9.3","contentType");
  qq("1.2.840.113549.1.9.4","messageDigest");
  qq("1.2.840.113549.1.9.5","signingTime");
  qq("1.2.840.113549.1.9.6","counterSignature");
  qq("1.2.840.113549.1.9.7","challengePassword");
  qq("1.2.840.113549.1.9.8","unstructuredAddress");
  qq("1.2.840.113549.1.9.14","extensionRequest");
  qq("1.2.840.113549.1.9.20","friendlyName");
  qq("1.2.840.113549.1.9.21","localKeyId");
  qq("1.2.840.113549.1.9.22.1","x509Certificate");
  qq("1.2.840.113549.1.12.10.1.1","keyBag");
  qq("1.2.840.113549.1.12.10.1.2","pkcs8ShroudedKeyBag");
  qq("1.2.840.113549.1.12.10.1.3","certBag");
  qq("1.2.840.113549.1.12.10.1.4","crlBag");
  qq("1.2.840.113549.1.12.10.1.5","secretBag");
  qq("1.2.840.113549.1.12.10.1.6","safeContentsBag");
  qq("1.2.840.113549.1.5.13","pkcs5PBES2");
  qq("1.2.840.113549.1.5.12","pkcs5PBKDF2");
  qq("1.2.840.113549.1.12.1.1","pbeWithSHAAnd128BitRC4");
  qq("1.2.840.113549.1.12.1.2","pbeWithSHAAnd40BitRC4");
  qq("1.2.840.113549.1.12.1.3","pbeWithSHAAnd3-KeyTripleDES-CBC");
  qq("1.2.840.113549.1.12.1.4","pbeWithSHAAnd2-KeyTripleDES-CBC");
  qq("1.2.840.113549.1.12.1.5","pbeWithSHAAnd128BitRC2-CBC");
  qq("1.2.840.113549.1.12.1.6","pbewithSHAAnd40BitRC2-CBC");
  qq("1.2.840.113549.2.7","hmacWithSHA1");
  qq("1.2.840.113549.2.8","hmacWithSHA224");
  qq("1.2.840.113549.2.9","hmacWithSHA256");
  qq("1.2.840.113549.2.10","hmacWithSHA384");
  qq("1.2.840.113549.2.11","hmacWithSHA512");
  qq("1.2.840.113549.3.7","des-EDE3-CBC");
  qq("2.16.840.1.101.3.4.1.2","aes128-CBC");
  qq("2.16.840.1.101.3.4.1.22","aes192-CBC");
  qq("2.16.840.1.101.3.4.1.42","aes256-CBC");
  qq("2.5.4.3","commonName");
  qq("2.5.4.4","surname");
  qq("2.5.4.5","serialNumber");
  qq("2.5.4.6","countryName");
  qq("2.5.4.7","localityName");
  qq("2.5.4.8","stateOrProvinceName");
  qq("2.5.4.9","streetAddress");
  qq("2.5.4.10","organizationName");
  qq("2.5.4.11","organizationalUnitName");
  qq("2.5.4.12","title");
  qq("2.5.4.13","description");
  qq("2.5.4.15","businessCategory");
  qq("2.5.4.17","postalCode");
  qq("2.5.4.42","givenName");
  qq("1.3.6.1.4.1.311.60.2.1.2","jurisdictionOfIncorporationStateOrProvinceName");
  qq("1.3.6.1.4.1.311.60.2.1.3","jurisdictionOfIncorporationCountryName");
  qq("2.16.840.1.113730.1.1","nsCertType");
  qq("2.16.840.1.113730.1.13","nsComment");
  gO("2.5.29.1","authorityKeyIdentifier");
  gO("2.5.29.2","keyAttributes");
  gO("2.5.29.3","certificatePolicies");
  gO("2.5.29.4","keyUsageRestriction");
  gO("2.5.29.5","policyMapping");
  gO("2.5.29.6","subtreesConstraint");
  gO("2.5.29.7","subjectAltName");
  gO("2.5.29.8","issuerAltName");
  gO("2.5.29.9","subjectDirectoryAttributes");
  gO("2.5.29.10","basicConstraints");
  gO("2.5.29.11","nameConstraints");
  gO("2.5.29.12","policyConstraints");
  gO("2.5.29.13","basicConstraints");
  qq("2.5.29.14","subjectKeyIdentifier");
  qq("2.5.29.15","keyUsage");
  gO("2.5.29.16","privateKeyUsagePeriod");
  qq("2.5.29.17","subjectAltName");
  qq("2.5.29.18","issuerAltName");
  qq("2.5.29.19","basicConstraints");
  gO("2.5.29.20","cRLNumber");
  gO("2.5.29.21","cRLReason");
  gO("2.5.29.22","expirationDate");
  gO("2.5.29.23","instructionCode");
  gO("2.5.29.24","invalidityDate");
  gO("2.5.29.25","cRLDistributionPoints");
  gO("2.5.29.26","issuingDistributionPoint");
  gO("2.5.29.27","deltaCRLIndicator");
  gO("2.5.29.28","issuingDistributionPoint");
  gO("2.5.29.29","certificateIssuer");
  gO("2.5.29.30","nameConstraints");
  qq("2.5.29.31","cRLDistributionPoints");
  qq("2.5.29.32","certificatePolicies");
  gO("2.5.29.33","policyMappings");
  gO("2.5.29.34","policyConstraints");
  qq("2.5.29.35","authorityKeyIdentifier");
  gO("2.5.29.36","policyConstraints");
  qq("2.5.29.37","extKeyUsage");
  gO("2.5.29.46","freshestCRL");
  gO("2.5.29.54","inhibitAnyPolicy");
  qq("1.3.6.1.4.1.11129.2.4.2","timestampList");
  qq("1.3.6.1.5.5.7.1.1","authorityInfoAccess");
  qq("1.3.6.1.5.5.7.3.1","serverAuth");
  qq("1.3.6.1.5.5.7.3.2","clientAuth");
  qq("1.3.6.1.5.5.7.3.3","codeSigning");
  qq("1.3.6.1.5.5.7.3.4","emailProtection");
  qq("1.3.6.1.5.5.7.3.8","timeStamping")
} /* confidence: 95% */

/* original: WI1 */ var WI1=c24.exports=ir6.pki.oids=ir6.oids=ir6.oids||{
  
};

/* original: qq */ function helper_fn(q,K){
  WI1[q]=K,WI1[K]=q
} /* confidence: 35% */

/* original: gO */ function helper_fn(q,K){
  WI1[q]=K
} /* confidence: 35% */

