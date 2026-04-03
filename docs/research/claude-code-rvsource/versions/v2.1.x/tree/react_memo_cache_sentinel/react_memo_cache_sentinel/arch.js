// Module: arch

/* original: xrY */ function darwin_win32_x64(){
  return process.platform==="darwin"||process.platform==="win32"&&process.arch==="x64"
} /* confidence: 65% */

/* original: r65 */ function helper_fn(){
  if(!xrY())return!1;
   /* confidence: 35% */

