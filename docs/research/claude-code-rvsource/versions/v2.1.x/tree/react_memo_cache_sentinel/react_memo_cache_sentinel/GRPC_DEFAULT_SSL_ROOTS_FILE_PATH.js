// Module: GRPC_DEFAULT_SSL_ROOTS_FILE_PATH

/* original: Bc1 */ var __esModule_fs=B((bn4)=>{
  Object.defineProperty(bn4,"__esModule",{
    value:!0
  });
  bn4.CIPHER_SUITES=void 0;
  bn4.getDefaultRootsData=yNz;
  var NNz=U6("fs");
  bn4.CIPHER_SUITES=process.env.GRPC_SSL_CIPHER_SUITES;
  var Cn4=process.env.GRPC_DEFAULT_SSL_ROOTS_FILE_PATH,pc1=null;
  function yNz(){
    if(Cn4){
      if(pc1===null)pc1=NNz.readFileSync(Cn4);
      return pc1
    }return null
  }
} /* confidence: 65% */

/* original: Cn4 */ var Cn4=process.env.GRPC_DEFAULT_SSL_ROOTS_FILE_PATH,pc1=null;

/* original: Vn1 */ var composed_value=Bc1(); /* confidence: 30% */

/* original: yNz */ function helper_fn(){
  if(Cn4){
    if(pc1===null)pc1=NNz.readFileSync(Cn4);
    return pc1
  }return null
} /* confidence: 35% */

