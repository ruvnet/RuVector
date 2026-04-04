// Module: safeParse

/* original: vu1 */ var vu1={
  
};

/* original: Gn_ */ function helper_fn(q){
  let{
    McpbManifestSchema:K
  }=await Promise.resolve().then(() => (Tu1(),vu1)),_=K.safeParse(q);
  if(!_.success){
    let z=_.error.flatten(),Y=[...Object.entries(z.fieldErrors).map(([$,O])=>`${$}: ${O?.join(", ")}`),...z.formErrors||[]].filter(Boolean).join("; ");
    throw Error(`Invalid manifest: ${Y}`)
  }return _.data
} /* confidence: 35% */

/* original: vn_ */ function helper_fn(q){
  let K;
  try{
    K=l8(q)
  }catch(_){
    throw Error(`Invalid JSON in manifest.json: ${F6(_)}`)
  }return Gn_(K)
} /* confidence: 35% */

