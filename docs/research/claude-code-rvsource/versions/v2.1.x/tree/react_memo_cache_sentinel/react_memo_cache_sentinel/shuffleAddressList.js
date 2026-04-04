// Module: shuffleAddressList

/* original: pe6 */ var pe6="pick_first",Uxz=250;

/* original: dxz */ var composed_value=new pE6(!1); /* confidence: 30% */

/* original: cxz */ function helper_fn(){
  (0,mn1.registerLoadBalancerType)(pe6,Xb8,pE6),(0,mn1.registerDefaultLoadBalancerType)(pe6)
} /* confidence: 35% */

/* original: pE6 */ class config{
  constructor(q){
    this.shuffleAddressList=q
  }getLoadBalancerName(){
    return pe6
  }toJsonObject(){
    return{
      [pe6]:{
        shuffleAddressList:this.shuffleAddressList
      }
    }
  }getShuffleAddressList(){
    return this.shuffleAddressList
  }static createFromJson(q){
    if("shuffleAddressList"in q&&typeof q.shuffleAddressList!=="boolean")throw Error("pick_first config field shuffleAddressList must be a boolean if provided");
    return new config(q.shuffleAddressList===!0)
  }
} /* confidence: 95% */

