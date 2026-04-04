// Module: assign

/* original: d83 */ var d83="__smithy_context";

/* original: dJ3 */ var composed_value=(q)=>{
  let K=(0,BJ3.resolveAwsSdkSigV4Config)(q);
  return Object.assign(K,{
    authSchemePreference:(0,tA1.normalizeProvider)(q.authSchemePreference??[])
  })
}; /* confidence: 30% */

/* original: Jf3 */ var composed_value=(q)=>{
  let K=(0,Af3.resolveAwsSdkSigV4Config)(q);
  return Object.assign(K,{
    authSchemePreference:(0,pw1.normalizeProvider)(q.authSchemePreference??[])
  })
}; /* confidence: 30% */

/* original: GG3 */ var composed_value=(q)=>{
  let K=(0,PG3.resolveAwsSdkSigV4Config)(q);
  return Object.assign(K,{
    authSchemePreference:(0,K21.normalizeProvider)(q.authSchemePreference??[])
  })
}; /* confidence: 30% */

/* original: Wy3 */ var composed_value=(q)=>{
  let K=(0,Yj1.memoizeIdentityProvider)(q.token,Yj1.isIdentityExpired,Yj1.doesIdentityRequireRefresh),_=(0,Hy3.resolveAwsSdkSigV4Config)(q);
  return Object.assign(_,{
    authSchemePreference:(0,$j1.normalizeProvider)(q.authSchemePreference??[]),token:K
  })
}; /* confidence: 30% */

/* original: L99 */ var composed_value=(q)=>{
  let K=(0,LM1.memoizeIdentityProvider)(q.token,LM1.isIdentityExpired,LM1.doesIdentityRequireRefresh),_=(0,k99.resolveAwsSdkSigV4Config)(q);
  return Object.assign(_,{
    authSchemePreference:(0,hM1.normalizeProvider)(q.authSchemePreference??[]),token:K
  })
}; /* confidence: 30% */

/* original: qG9 */ var composed_value=(q)=>{
  let K=(0,a09.resolveAwsSdkSigV4Config)(q);
  return Object.assign(K,{
    authSchemePreference:(0,aP1.normalizeProvider)(q.authSchemePreference??[])
  })
}; /* confidence: 30% */

