// Module: statSync

/* original: Sl9 */ function env_config(){
  try{
    return myq.statSync("/.dockerenv"),!0
  }catch{
    return!1
  }
} /* confidence: 95% */

/* original: Tv1 */ function helper_fn(){
  if(vv1===void 0)vv1=Sl9()||Cl9();
   /* confidence: 35% */

