// Module: replace

/* original: g1Y */ var g1Y=/--!?>/;

/* original: r2K */ function helper_fn(q){
  if(!g1Y.test(q))return q;
  return q.replace(/(--\!?)>/g,"$1&gt;")
} /* confidence: 35% */

