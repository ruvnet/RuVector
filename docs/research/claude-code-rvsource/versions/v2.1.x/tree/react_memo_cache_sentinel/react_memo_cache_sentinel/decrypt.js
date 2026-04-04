// Module: decrypt

/* original: K */ let composed_value=iv()?void 0:process.env.ANTHROPIC_API_KEY; /* confidence: 30% */

/* original: iv */ function utility_fn(){
  return!1
} /* confidence: 40% */

/* original: p_6 */ function NoSubtleCryptoimplementationfo(q,K,_){
  if(K=K||"",_=_||globalThis.crypto&&globalThis.crypto.subtle||IE7.SubtleCrypto,!_)throw Error("No SubtleCrypto implementation found");
  try{
    let z=await _.importKey("raw",fK1(K),{
      name:"AES-CBC",length:128
    },!0,["encrypt","decrypt"]),[Y,$]=q.split("."),O=await _.decrypt({
      name:"AES-CBC",iv:fK1(Y)
    },z,fK1($));
    return new TextDecoder().decode(O)
  }catch(z){
    throw Error("Failed to decrypt")
  }
} /* confidence: 65% */

/* original: OL7 */ function helper_fn(q,K,_){
  if(q={
    ...q
  },q.encryptedFeatures){
    try{
      q.features=JSON.parse(await p_6(q.encryptedFeatures,K,_))
    }catch(z){
      console.error(z)
    }delete q.encryptedFeatures
  }if(q.encryptedExperiments){
    try{
      q.experiments=JSON.parse(await p_6(q.encryptedExperiments,K,_))
    }catch(z){
      console.error(z)
    }delete q.encryptedExperiments
  }if(q.encryptedSavedGroups){
    try{
      q.savedGroups=JSON.parse(await p_6(q.encryptedSavedGroups,K,_))
    }catch(z){
      console.error(z)
    }delete q.encryptedSavedGroups
  }return q
} /* confidence: 35% */

/* original: g65 */ function helper_fn(){
  Cx(yrY)
} /* confidence: 35% */

