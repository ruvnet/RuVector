// Module: getTime

/* original: ESz */ var ESz=2147483647;

/* original: LSz */ function helper_fn(q){
  let K=q instanceof Date?q.getTime():q,_=new Date().getTime(),z=K-_;
  if(z<0)return 0;
  else if(z>ESz)return 1/0;
  else return z
} /* confidence: 35% */

