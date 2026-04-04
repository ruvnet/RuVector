// Module: contentLengthMiddleware

/* original: mn */ var request=B((xw3)=>{
  var Cw3=bs7(),xs7="content-length";
  function Is7(q){
    return(K)=>async(_)=>{
      let z=_.request;
      if(Cw3.HttpRequest.isInstance(z)){
        let{
          body:Y,headers:$
        }=z;
        if(Y&&Object.keys($).map((O)=>O.toLowerCase()).indexOf(xs7)===-1)try{
          let O=q(Y);
          z.headers={
            ...z.headers,[xs7]:String(O)
          }
        }catch(O){
          
        }
      }return K({
        ..._,request:z
      })
    }
  }var us7={
    step:"build",tags:["SET_CONTENT_LENGTH","CONTENT_LENGTH"],name:"contentLengthMiddleware",override:!0
  },bw3=(q)=>({
    applyToStack:(K)=>{
      K.add(Is7(q.bodyLengthChecker),us7)
    }
  });
  xw3.contentLengthMiddleware=Is7;
  xw3.contentLengthMiddlewareOptions=us7;
  xw3.getContentLengthPlugin=bw3
} /* confidence: 70% */

/* original: us7 */ var build_SET_CONTENT_LENGTH_CONTE={
  step:"build",tags:["SET_CONTENT_LENGTH","CONTENT_LENGTH"],name:"contentLengthMiddleware",override:!0
} /* confidence: 65% */

