// Module: parse

/* original: _ */ let composed_value=$O(q.command); /* confidence: 30% */

/* original: $ */ let composed_value=$O(_); /* confidence: 30% */

/* original: K */ let composed_value=ko().parse(q); /* confidence: 30% */

/* original: K */ let composed_value=ko().parse(q); /* confidence: 30% */

/* original: _ */ let composed_value=ko().parse(q); /* confidence: 30% */

/* original: O */ let composed_value=$O(q); /* confidence: 30% */

/* original: ko */ function utility_fn(){
  return p$z
} /* confidence: 40% */

/* original: bh4 */ function BashTool(q,K){
  let _=$O(q.command);
   /* confidence: 31% */

/* original: Dwz */ function helper_fn(q){
  let K=$O(q);
   /* confidence: 35% */

/* original: t9Y */ function helper_fn(q){
  let _=$O(q).at(-1)||q;
   /* confidence: 35% */

/* original: YzY */ function helper_fn(q){
  let K=$O(q);
   /* confidence: 35% */

/* original: p17 */ function helper_fn(q){
  let K=$O(q);
   /* confidence: 35% */

/* original: wzY */ function helper_fn(q){
  let K=$O(q);
   /* confidence: 35% */

/* original: _s6 */ function StringTrimmer(q){
  return $O(q).some((K)=>l17(K.trim()))
} /* confidence: 41% */

/* original: $O */ function state_manager(q){
  if(!q)return[];
  if(q.length>f47)return[q];
  let K=ko().parse(q);
  if(!K)return[q];
  let _=[],z=(Y)=>{
    if(x2Y.has(Y.type)||Y.type==="comment")return;
    if(Y.type==="redirected_statement"){
      for(let $ of Y.children)if(!$.type.endsWith("_redirect"))z($);
      return
    }if(b2Y.has(Y.type)){
      for(let $ of Y.children)z($);
      return
    }_.push(Y.text)
  };
  return z(K),_
} /* confidence: 95% */

