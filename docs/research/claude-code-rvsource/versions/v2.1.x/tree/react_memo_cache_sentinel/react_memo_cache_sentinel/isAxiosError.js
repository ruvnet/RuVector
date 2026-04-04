// Module: isAxiosError

/* original: $ */ let composed_value=await w87(Y,K.signal,zMK); /* confidence: 30% */

/* original: kIq */ function utility_fn(){
  return`Claude-User (${M$()}; +https://support.anthropic.com/)`
} /* confidence: 40% */

/* original: w87 */ function response(q,K,_,z=0){
  if(z>eJK)throw Error(`Too many redirects (exceeded ${eJK})`);
  try{
    return await O1.get(q,{
      signal:K,timeout:p4Y,maxRedirects:0,responseType:"arraybuffer",maxContentLength:m4Y,headers:{
        Accept:"text/markdown, text/html, */*","User-Agent":kIq()
      }
    })
  }catch(Y){
    if(O1.isAxiosError(Y)&&Y.response&&[301,302,307,308].includes(Y.response.status)){
      let $=Y.response.headers.location;
      if(!$)throw Error("Redirect missing Location header");
      let O=new URL($,q).toString();
      if(_(q,O))return response(O,K,_,z+1);
      else return{
        type:"redirect",originalUrl:q,redirectUrl:O,statusCode:Y.response.status
      }
    }if(O1.isAxiosError(Y)&&Y.response?.status===403&&Y.response.headers["x-proxy-error"]==="blocked-by-allowlist"){
      let $=new URL(q).hostname;
      throw new qMK($)
    }throw Y
  }
} /* confidence: 70% */

