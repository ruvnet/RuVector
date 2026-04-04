// Module: timeout

/* original: dJz */ var urnietfparamsoauthgranttypetok=30000,cJz="urn:ietf:params:oauth:grant-type:token-exchange",lC4="urn:ietf:params:oauth:grant-type:jwt-bearer",cC4="urn:ietf:params:oauth:token-type:id-jag",lJz="urn:ietf:params:oauth:token-type:id_token",mh8,mo,nJz,iJz,rJz; /* confidence: 65% */

/* original: nC4 */ function BashTool(q){
  return(K,_)=>{
    let z=AbortSignal.timeout(dJz),Y=q?AbortSignal.any([z,q]):z;
    return fetch(K,{
      ..._,signal:Y
    })
  }
} /* confidence: 31% */

