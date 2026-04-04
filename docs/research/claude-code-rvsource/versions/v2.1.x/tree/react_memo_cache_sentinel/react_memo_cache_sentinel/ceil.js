// Module: ceil

/* original: NSz */ var m_S_M_H=[["m",1],["S",1000],["M",60000],["H",3600000]]; /* confidence: 65% */

/* original: ySz */ function helper_fn(q){
  let K=new Date().getTime();
  if(q instanceof Date)q=q.getTime();
  let _=Math.max(q-K,0);
  for(let[z,Y]of NSz){
    let $=_/Y;
    if($<1e8)return String(Math.ceil($))+z
  }throw Error("Deadline is too far in the future")
} /* confidence: 35% */

