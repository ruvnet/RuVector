// Module: enabledPlugins

/* original: X */ let composed_value=t0(); /* confidence: 30% */

/* original: t0 */ function utility_fn(){
  return G8.additionalDirectoriesForClaudeMd
} /* confidence: 40% */

/* original: ZA6 */ function helper_fn(){
  let q={
    
  };
  for(let K of t0())for(let _ of o34){
    let{
      settings:z
    }=E66(r34(K,".claude",_));
    if(!z?.enabledPlugins)continue;
    Object.assign(q,z.enabledPlugins)
  }return q
} /* confidence: 35% */

