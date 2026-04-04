// Module: decode

/* original: ap5 */ var ap5=new TextDecoder;

/* original: sp5 */ function helper_fn(q){
  if(q.length===0)return"";
  if(q[0]===239&&q[1]===187&&q[2]===191)q=q.subarray(3);
  return ap5.decode(q)
} /* confidence: 35% */

