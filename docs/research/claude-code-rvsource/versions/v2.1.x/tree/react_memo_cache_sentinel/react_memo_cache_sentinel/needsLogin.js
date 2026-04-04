// Module: needsLogin

/* original: q_K */ function utility_fn(){
  return await ND6({
    ignoreUntracked:!0
  })
} /* confidence: 40% */

/* original: Ia1 */ function needsLogin_needsGitStash(){
  let q=new Set,[K,_]=await Promise.all([Gu8(),q_K()]);
  if(K)q.add("needsLogin");
  if(!_)q.add("needsGitStash");
  return q
} /* confidence: 65% */

