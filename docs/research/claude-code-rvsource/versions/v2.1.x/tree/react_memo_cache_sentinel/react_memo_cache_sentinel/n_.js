// Module: n_

/* original: SA */ var SA=600000,mIY=1500;

/* original: _ */ let composed_value=await gT6("skills",K[0]); /* confidence: 30% */

/* original: Vx */ function helper_fn({
  getAppState:q,hookInput:K,matchQuery:_,signal:z,timeoutMs:Y=SA
} /* confidence: 35% */

/* original: ZQ */ function helper_fn(q,K=SA){
  let{
    message:_,title:z,notificationType:Y
  }=q,$={
    ...n$(void 0),hook_event_name:"Notification",message:_,title:z,notification_type:Y
  };
  await Vx({
    hookInput:$,timeoutMs:K,matchQuery:Y
  })
} /* confidence: 35% */

/* original: pg8 */ function StopFailure_unknown_StopFailur(q,K,_=SA){
  let z=K?.getAppState(),Y=N8();
  if(!aC6("StopFailure",z,Y))return;
  let $=Z3(q.message.content,`
`).trim()||void 0,O=q.error??"unknown",A={
    ...n$(void 0,void 0,K),hook_event_name:"StopFailure",error:O,error_details:q.errorDetails,last_assistant_message:$
  };
  await Vx({
    getAppState:K?.getAppState,hookInput:A,timeoutMs:_,matchQuery:O
  })
} /* confidence: 65% */

/* original: _S6 */ function StringTrimmer(q,K,_=SA){
  let z={
    ...n$(void 0),hook_event_name:"PreCompact",trigger:q.trigger,custom_instructions:q.customInstructions
  },Y=await Vx({
    hookInput:z,matchQuery:q.trigger,signal:K,timeoutMs:_
  });
  if(Y.length===0)return{
    
  };
  let $=Y.filter((A)=>A.succeeded&&A.output.trim().length>0).map((A)=>A.output.trim()),O=[];
  for(let A of Y)if(A.succeeded)if(A.output.trim())O.push(`PreCompact [${A.command}] completed successfully: ${A.output.trim()}`);
  else O.push(`PreCompact [${A.command}] completed successfully`);
  else if(A.output.trim())O.push(`PreCompact [${A.command}] failed: ${A.output.trim()}`);
  else O.push(`PreCompact [${A.command}] failed`);
  return{
    newCustomInstructions:$.length>0?$.join(`

`):void 0,userDisplayMessage:O.length>0?O.join(`
`):void 0
  }
} /* confidence: 41% */

/* original: rg8 */ function StringTrimmer(q,K,_=SA){
  let z={
    ...n$(void 0),hook_event_name:"PostCompact",trigger:q.trigger,compact_summary:q.compactSummary
  },Y=await Vx({
    hookInput:z,matchQuery:q.trigger,signal:K,timeoutMs:_
  });
  if(Y.length===0)return{
    
  };
  let $=[];
  for(let O of Y)if(O.succeeded)if(O.output.trim())$.push(`PostCompact [${O.command}] completed successfully: ${O.output.trim()}`);
  else $.push(`PostCompact [${O.command}] completed successfully`);
  else if(O.output.trim())$.push(`PostCompact [${O.command}] failed: ${O.output.trim()}`);
  else $.push(`PostCompact [${O.command}] failed`);
  return{
    userDisplayMessage:$.length>0?$.join(`
`):void 0
  }
} /* confidence: 41% */

/* original: mq8 */ function helper_fn(q,K){
  let{
    getAppState:_,setAppState:z,signal:Y,timeoutMs:$=SA
  }=K||{
    
  },O={
    ...n$(void 0),hook_event_name:"SessionEnd",reason:q
  },A=await Vx({
    getAppState:_,hookInput:O,matchQuery:q,signal:Y,timeoutMs:$
  });
  for(let w of A)if(!w.succeeded&&w.output)process.stderr.write(`SessionEnd hook [${w.command}] failed: ${w.output}
`);
  if(z){
    let w=N8();
    xy6(z,w)
  }
} /* confidence: 35% */

/* original: gT6 */ function ConfigChange_policy_settings(q,K,_=SA){
  let z={
    ...n$(void 0),hook_event_name:"ConfigChange",source:q,file_path:K
  },Y=await Vx({
    hookInput:z,timeoutMs:_,matchQuery:q
  });
  if(q==="policy_settings")return Y.map(($)=>({
    ...$,blocked:!1
  }));
  return Y
} /* confidence: 65% */

/* original: sa1 */ function helper_fn(q,K,_=SA){
  let z={
    ...n$(void 0),hook_event_name:"CwdChanged",old_cwd:q,new_cwd:K
  };
   /* confidence: 35% */

/* original: ta1 */ function helper_fn(q,K,_=SA){
  let z={
    ...n$(void 0),hook_event_name:"FileChanged",file_path:q,event:K
  };
   /* confidence: 35% */

/* original: Fo6 */ function helper_fn(q,K,_,z){
  let{
    globs:Y,triggerFilePath:$,parentFilePath:O,timeoutMs:A=SA
  }=z??{
    
  },w={
    ...n$(void 0),hook_event_name:"InstructionsLoaded",file_path:q,memory_type:K,load_reason:_,globs:Y,trigger_file_path:$,parent_file_path:O
  };
  await Vx({
    hookInput:w,timeoutMs:A,matchQuery:_
  })
} /* confidence: 35% */

/* original: kK8 */ function StringTrimmer(q){
  let K={
    ...n$(void 0),hook_event_name:"WorktreeCreate",name:q
  },_=await Vx({
    hookInput:K,timeoutMs:SA
  }),z=_.find(($)=>$.succeeded&&$.output.trim().length>0);
  if(!z){
    let $=_.filter((O)=>!O.succeeded).map((O)=>`${O.command}: ${O.output.trim()||"no output"}`);
    throw Error(`WorktreeCreate hook failed: ${$.join("; ")||"no successful output"}`)
  }return{
    worktreePath:z.output.trim()
  }
} /* confidence: 41% */

/* original: ac8 */ function WorktreeRemove_error(q){
  let K=Wd()?.WorktreeRemove,_=fR()?.WorktreeRemove,z=K&&K.length>0,Y=_&&_.length>0;
  if(!z&&!Y)return!1;
  let $={
    ...n$(void 0),hook_event_name:"WorktreeRemove",worktree_path:q
  },O=await Vx({
    hookInput:$,timeoutMs:SA
  });
  if(O.length===0)return!1;
  for(let A of O)if(!A.succeeded)N(`WorktreeRemove hook failed [${A.command}]: ${A.output.trim()}`,{
    level:"error"
  });
  return!0
} /* confidence: 65% */

