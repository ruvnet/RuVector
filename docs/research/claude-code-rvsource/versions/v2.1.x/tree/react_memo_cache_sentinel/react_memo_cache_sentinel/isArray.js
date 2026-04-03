// Module: isArray

/* original: ZZ5 */ var ref_recursiveRef_recursiveAnch=new Set(["$ref","$recursiveRef","$recursiveAnchor","$dynamicRef","$dynamicAnchor"]); /* confidence: 65% */

/* original: Pq1 */ function helper_fn(q){
  for(let K in q){
    if(ZZ5.has(K))return!0;
    let _=q[K];
    if(Array.isArray(_)&&_.some(helper_fn))return!0;
    if(typeof _=="object"&&helper_fn(_))return!0
  }return!1
} /* confidence: 35% */

