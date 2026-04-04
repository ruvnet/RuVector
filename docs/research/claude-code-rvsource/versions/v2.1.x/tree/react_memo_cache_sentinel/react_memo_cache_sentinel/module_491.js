// Module: module_491

/* original: YJ6 */ var YJ6=null,TF8=null;

/* original: mGK */ function helper_fn(){
  if(!YJ6)return;
  try{
    await yq7(YJ6,{
      recursive:!0,force:!0
    }),N(`Cleaned up session plugin cache at ${YJ6}`)
  }catch(q){
    N(`Failed to clean up session plugin cache: ${q}`)
  }finally{
    YJ6=null,TF8=null
  }
} /* confidence: 35% */

