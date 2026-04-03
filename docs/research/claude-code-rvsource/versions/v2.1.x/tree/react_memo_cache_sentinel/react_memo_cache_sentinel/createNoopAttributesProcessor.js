// Module: createNoopAttributesProcessor

/* original: NS8 */ var composed_value=B((tg4)=>{
  Object.defineProperty(tg4,"__esModule",{
    value:!0
  });
  tg4.createDenyListAttributesProcessor=tg4.createAllowListAttributesProcessor=tg4.createMultiAttributesProcessor=tg4.createNoopAttributesProcessor=void 0;
  class rg4{
    process(q,K){
      return q
    }
  }class og4{
    _processors;
    constructor(q){
      this._processors=q
    }process(q,K){
      let _=q;
      for(let z of this._processors)_=z.process(_,K);
      return _
    }
  }class ag4{
    _allowedAttributeNames;
    constructor(q){
      this._allowedAttributeNames=q
    }process(q,K){
      let _={
        
      };
      return Object.keys(q).filter((z)=>this._allowedAttributeNames.includes(z)).forEach((z)=>_[z]=q[z]),_
    }
  }class sg4{
    _deniedAttributeNames;
    constructor(q){
      this._deniedAttributeNames=q
    }process(q,K){
      let _={
        
      };
      return Object.keys(q).filter((z)=>!this._deniedAttributeNames.includes(z)).forEach((z)=>_[z]=q[z]),_
    }
  }function S0z(){
    return I0z
  }tg4.createNoopAttributesProcessor=S0z;
  function C0z(q){
    return new og4(q)
  }tg4.createMultiAttributesProcessor=C0z;
  function b0z(q){
    return new ag4(q)
  }tg4.createAllowListAttributesProcessor=b0z;
  function x0z(q){
    return new sg4(q)
  }tg4.createDenyListAttributesProcessor=x0z;
  var I0z=new rg4
} /* confidence: 30% */

/* original: x0z */ function helper_fn(q){
  return new sg4(q)
} /* confidence: 35% */

/* original: sg4 */ class entity_class{
  _deniedAttributeNames;
   /* confidence: 45% */

