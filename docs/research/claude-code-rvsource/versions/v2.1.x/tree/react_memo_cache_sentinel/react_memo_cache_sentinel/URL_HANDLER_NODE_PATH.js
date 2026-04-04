// Module: URL_HANDLER_NODE_PATH

/* original: K38 */ var K38=null;

/* original: m6$ */ function url_handler(){
  if(K38)return K38;
  if(process.platform!=="darwin")return null;
  try{
    if(process.env.URL_HANDLER_NODE_PATH)K38=U6(process.env.URL_HANDLER_NODE_PATH);
    else{
      let q=u6$(I6$(x6$(import.meta.url)),"..","url-handler",`${process.arch}-darwin`,"url-handler.node");
      K38=b6$(import.meta.url)(q)
    }return K38
  }catch{
    return null
  }
} /* confidence: 70% */

/* original: p6$ */ function helper_fn(q){
  let K=m6$();
   /* confidence: 35% */

