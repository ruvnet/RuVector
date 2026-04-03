// Module: sessionId

/* original: Y */ let composed_value=N8(); /* confidence: 30% */

/* original: z */ let composed_value=K??N8(); /* confidence: 30% */

/* original: _ */ let composed_value=await ca1(); /* confidence: 30% */

/* original: _ */ let composed_value=q.name??Pd(),z=await O78(N8(),composed_value); /* confidence: 30% */

/* original: H */ let composed_value=N8(); /* confidence: 30% */

/* original: z */ let composed_value={
  ...K,pastedContents:_,timestamp:Date.now(),project:iz(),sessionId:N8()
}; /* confidence: 30% */

/* original: q */ let composed_value=N8(); /* confidence: 30% */

/* original: Y */ let composed_value=N8(),$=yR8(q,composed_value); /* confidence: 30% */

/* original: K */ let composed_value=N8(); /* confidence: 30% */

/* original: _ */ let composed_value=N8(),z=K===composed_value,Y; /* confidence: 30% */

/* original: K */ let composed_value=q?.dir,_=q?.lockIdentity??N8(),z=await g15(composed_value); /* confidence: 30% */

/* original: E1 */ let helper_fn={
  type:"prompt_suggestion",suggestion:b8.suggestion,uuid:tM(),session_id:N8()
} /* confidence: 35% */

/* original: N8 */ function utility_fn(){
  return G8.sessionId
} /* confidence: 40% */

/* original: DN6 */ function helper_fn(){
  let q=qC(),K=N8(),_={
    "user.id":q
  };
   /* confidence: 35% */

/* original: PW */ function helper_fn(q){
  let K=Pd(N8());
   /* confidence: 35% */

/* original: ca1 */ function env_config(){
  let q=Iu8(q7(),"session-env",N8());
  return await Vnz(q,{
    recursive:!0
  }),q
} /* confidence: 95% */

/* original: S_K */ function helper_fn(q,K){
  let _=q.toLowerCase();
  return Iu8(await ca1(),`${_}-hook-${K}.sh`)
} /* confidence: 35% */

/* original: cVK */ function status(){
  let q=N8(),K=await FK7();
  if(K.status==="not-installed")return{
    success:!1,error:"Claude Desktop is not installed. Install it from https://claude.ai/download"
  };
  if(K.status==="version-too-old")return{
    success:!1,error:`Claude Desktop ${K.version} is too old to resume this session. Please update to ${RU8} or later.`
  };
  let _=WJY(q);
  if(!await ZJY(_))return{
    success:!1,error:"Failed to open Claude Desktop. Please try opening it manually.",deepLinkUrl:_
  };
  return{
    success:!0,deepLinkUrl:_
  }
} /* confidence: 70% */

/* original: TbK */ function Iapologize_Noprompt_Noprompt(){
  if(qd8)return qd8;
  let q=N8();
  return qd8=Kh6(10).then((K)=>{
    return L48=K.filter((_)=>{
      if(_.isSidechain)return!1;
      if(_.sessionId===q)return!1;
      if(_.summary?.includes("I apologize"))return!1;
      let z=_.summary&&_.summary!=="No prompt",Y=_.firstPrompt&&_.firstPrompt!=="No prompt";
      return z||Y
    }).slice(0,3),L48
  }).catch(()=>{
    return L48=[],L48
  }),qd8
} /* confidence: 65% */

/* original: Fc8 */ function MemoComponent(){
  return dk(rj(Z8()),N8(),"session-memory")+vf
} /* confidence: 87% */

/* original: ggY */ function helper_fn(){
  return hnK(q7(),"uploads",N8())
} /* confidence: 35% */

