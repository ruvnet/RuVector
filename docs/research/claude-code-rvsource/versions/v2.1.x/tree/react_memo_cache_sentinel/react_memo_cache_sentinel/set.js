// Module: set

/* original: vnz */ var vnz=10,c$;

/* original: Pd */ function ConfigManager(q){
  let K=q??N8(),_=B96(),z=_.get(K);
  if(!z){
    let Y=c$();
    for(let $=0;
    $<vnz;
    $++){
      z=oL6();
      let O=rK6(Y,`${z}.md`);
      if(!M8().existsSync(O))break
    }_.set(K,z)
  }return z
} /* confidence: 34% */

