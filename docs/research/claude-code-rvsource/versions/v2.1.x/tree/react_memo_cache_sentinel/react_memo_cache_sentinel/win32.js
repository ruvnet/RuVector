// Module: win32

/* original: rS7 */ function utility_fn(){
  return process.platform==="win32"
} /* confidence: 40% */

/* original: Yg */ function helper_fn(q,K=[],_){
  if(rS7()){
    let z=HB6(q);
    if(z===null)throw Error(`Command '${q}' not found or is in an unsafe location (current directory)`);
    return zg(z,K,_)
  }return zg(q,K,_)
} /* confidence: 35% */

