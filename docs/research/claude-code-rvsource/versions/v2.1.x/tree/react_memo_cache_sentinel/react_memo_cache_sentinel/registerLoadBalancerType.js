// Module: registerLoadBalancerType

/* original: fb8 */ var fb8="round_robin";

/* original: EIz */ function helper_fn(){
  (0,Ze4.registerLoadBalancerType)(fb8,ln1,Zb8)
} /* confidence: 35% */

/* original: Zb8 */ class entity_class{
  getLoadBalancerName(){
    return fb8
  }constructor(){
    
  }toJsonObject(){
    return{
      [fb8]:{
        
      }
    }
  }static createFromJson(q){
    return new entity_class
  }
} /* confidence: 45% */

