// Module: some

/* original: RD6 */ function utility_fn(){
  return process.versions.bun!==void 0
} /* confidence: 40% */

/* original: y1$ */ function helper_fn(){
  let q=RD6(),K=process.execArgv.some((z)=>{
    if(q)return/--inspect(-brk)?/.test(z);
    else return/--inspect(-brk)?|--debug(-brk)?/.test(z)
  } /* confidence: 35% */

