// Module: utf8

/* original: Byq */ var linux_microsoft_procversion_ut=()=>{
  if(gyq.platform!=="linux")return!1;
  if(Il9.release().toLowerCase().includes("microsoft")){
    if(WG6())return!1;
    return!0
  }try{
    return ul9.readFileSync("/proc/version","utf8").toLowerCase().includes("microsoft")?!WG6():!1
  }catch{
    return!1
  }
} /* confidence: 65% */

/* original: Cl9 */ function procselfcgroup_utf8_docker(){
  try{
    return myq.readFileSync("/proc/self/cgroup","utf8").includes("docker")
  } /* confidence: 65% */

/* original: WG6 */ function helper_fn(){
  if(kv1===void 0)kv1=xl9()||Tv1();
  return kv1
} /* confidence: 35% */

