// Module: facets

/* original: K */ let composed_value=wt(yc8(),`${q.session_id}.json`); /* confidence: 30% */

/* original: K */ let composed_value=wt(Uz7(),`${q.session_id}.json`); /* confidence: 30% */

/* original: b */ let composed_value=wt(Nc8(),"report.html"); /* confidence: 30% */

/* original: Nc8 */ function utility_fn(){
  return wt(q7(),"usage-data")
} /* confidence: 40% */

/* original: yc8 */ function helper_fn(){
  return wt(Nc8(),"facets")
} /* confidence: 35% */

/* original: Uz7 */ function helper_fn(){
  return wt(Nc8(),"session-meta")
} /* confidence: 35% */

/* original: FbY */ function helper_fn(q){
  let K=wt(Uz7(),`${q}.json`);
  try{
    let _=await pUK(K,{
      encoding:"utf-8"
    });
    return l8(_)
  }catch{
    return null
  }
} /* confidence: 35% */

