// Module: getAccessToken

/* original: ebq */ var request=B((sbq)=>{
  Object.defineProperty(sbq,"__esModule",{
    value:!0
  });
  sbq.PassThroughClient=void 0;
  var L8_=CF();
  class AV1 extends L8_.AuthClient{
    async request(q){
      return this.transporter.request(q)
    }async getAccessToken(){
      return{
        
      }
    }async getRequestHeaders(){
      return{
        
      }
    }
  }sbq.PassThroughClient=AV1;
  var h8_=new AV1;
  h8_.getAccessToken()
} /* confidence: 70% */

/* original: h8_ */ var h8_=new AV1;

