// Module: protocolMode

/* original: v4 */ var v4={
  
};

/* original: eNq */ function helper_fn(q,K,_,z){
  let Y=rd6.getStandardAuthorizeRequestParameters({
    ...q.auth,authority:K,redirectUri:_.redirectUri||""
  },_,z);
  if(v4.addLibraryInfo(Y,{
    sku:uT.MSAL_SKU,version:vu,cpu:process.arch||"",os:process.platform||""
  }),q.auth.protocolMode!==CG.OIDC)v4.addApplicationTelemetry(Y,q.telemetry.application);
  if(v4.addResponseType(Y,B06.CODE),_.codeChallenge&&_.codeChallengeMethod)v4.addCodeChallengeParams(Y,_.codeChallenge,_.codeChallengeMethod);
  return v4.addExtraQueryParameters(Y,_.extraQueryParameters||{
    
  }),rd6.getAuthorizeUrl(K,Y,q.auth.encodeExtraQueryParams,_.extraQueryParameters)
} /* confidence: 35% */

