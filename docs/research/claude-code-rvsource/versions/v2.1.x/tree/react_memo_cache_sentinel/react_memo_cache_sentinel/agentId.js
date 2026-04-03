// Module: agentId

/* original: R */ let composed_value=await j0K(K); /* confidence: 30% */

/* original: E */ let composed_value=await j0K(_); /* confidence: 30% */

/* original: _ */ let composed_value=PW(q.agentId),z=KP(q.agentId)!==null; /* confidence: 30% */

/* original: Y */ let composed_value=PW(K.agentId),$=KP(K.agentId),O=[]; /* confidence: 30% */

/* original: _ */ let composed_value=PW(q.agentId),z=KP(q.agentId)!==null; /* confidence: 30% */

/* original: _ */ let composed_value=K.map((z)=>{
  if(z.type!=="tool_use")return z;
  if(z.name===UX){
    let Y=KP();
    if(Y)return{
      ...z,input:{
        ...z.input,plan:Y
      }
    }
  }return z
} /* confidence: 30% */

/* original: A */ let headers=KP(),w=PW(); /* confidence: 70% */

/* original: A */ let headers=N8(),w=B96().get(headers); /* confidence: 70% */

/* original: B96 */ function utility_fn(){
  return G8.planSlugCache
} /* confidence: 40% */

/* original: da1 */ function helper_fn(q,K){
  B96().set(q,K)
} /* confidence: 35% */

/* original: N_K */ function helper_fn(){
  B96().clear()
} /* confidence: 35% */

/* original: dg8 */ function helper_fn(q){
  let K=KP(q);
   /* confidence: 35% */

/* original: j0K */ function plan_plan_mode_full(q){
  if(q.getAppState().toolPermissionContext.mode!=="plan")return null;
  let _=PW(q.agentId),z=KP(q.agentId)!==null;
  return P4({
    type:"plan_mode",reminderType:"full",isSubAgent:!!q.agentId,planFilePath:_,planExists:z
  })
} /* confidence: 65% */

