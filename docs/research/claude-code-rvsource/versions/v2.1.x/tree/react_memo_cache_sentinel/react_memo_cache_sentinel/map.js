// Module: map

/* original: JM */ var JM=Symbol("map");

/* original: K */ let composed_value=EG6(this[JM],q); /* confidence: 30% */

/* original: _ */ let composed_value=EG6(this[JM],q); /* confidence: 30% */

/* original: _ */ let composed_value=EG6(this[JM],q); /* confidence: 30% */

/* original: K */ let composed_value=EG6(this[JM],q); /* confidence: 30% */

/* original: mT1 */ function keyvalue_key_value__(q){
  let K=arguments.length>1&&arguments[1]!==void 0?arguments[1]:"key+value";
  return Object.keys(q[JM]).sort().map(K==="key"?function(z){
    return z.toLowerCase()
  }:K==="value"?function(z){
    return q[JM][z].join(", ")
  }:function(z){
    return[z.toLowerCase(),q[JM][z].join(", ")]
  })
} /* confidence: 65% */

