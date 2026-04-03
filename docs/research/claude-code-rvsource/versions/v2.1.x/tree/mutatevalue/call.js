// Module: call

/* original: pe1 */ var composed_value=B((lfw,u2K)=>{
  var I2K=th6();
  u2K.exports=me1;
  function me1(){
    I2K.call(this),this.view=null,this.detail=0
  }me1.prototype=Object.create(I2K.prototype,{
    constructor:{
      value:me1
    },initUIEvent:{
      value:function(q,K,_,z,Y){
        this.initEvent(q,K,_),this.view=z,this.detail=Y
      }
    }
  })
} /* confidence: 30% */

/* original: b67 */ var composed_value=B((yZw,kHK)=>{
  kHK.exports={
    Event:th6(),UIEvent:pe1(),MouseEvent:ge1(),CustomEvent:THK()
  }
} /* confidence: 30% */

/* original: me1 */ function utility_fn(){
  I2K.call(this),this.view=null,this.detail=0
} /* confidence: 40% */

