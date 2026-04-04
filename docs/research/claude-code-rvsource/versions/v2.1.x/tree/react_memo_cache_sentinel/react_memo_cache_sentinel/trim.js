// Module: trim

/* original: q */ let composed_value=Fu.trim(); /* confidence: 30% */

/* original: Fu */ var Fu="",Uv6=!1,jn6=null;

/* original: Af_ */ function p_print_utf8_string_readable(){
  if(!process.stdin.isTTY||Uv6||process.argv.includes("-p")||process.argv.includes("--print"))return;
  Uv6=!0,Fu="";
  try{
    process.stdin.setEncoding("utf8"),process.stdin.setRawMode(!0),process.stdin.ref(),jn6=()=>{
      let q=process.stdin.read();
      while(q!==null){
        if(typeof q==="string")wf_(q);
        q=process.stdin.read()
      }
    },process.stdin.on("readable",jn6)
  }catch{
    Uv6=!1
  }
} /* confidence: 65% */

/* original: jf_ */ function StringTrimmer(){
  return Fu.trim().length>0
} /* confidence: 41% */

/* original: OL1 */ function helper_fn(q){
  Fu=q
} /* confidence: 35% */

