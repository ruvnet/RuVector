// Module: system

/* original: dwz */ var dwz=1000,zs6;

/* original: Eo */ function helper_fn(q){
  if(!g7())return;
  if(zs6.length>=dwz)zs6.shift();
  zs6.push(q)
} /* confidence: 35% */

/* original: sN */ function system_task_notification(q,K,_){
  Eo({
    type:"system",subtype:"task_notification",task_id:q,tool_use_id:_?.toolUseId,status:K,output_file:_?.outputFile??"",summary:_?.summary??"",usage:_?.usage
  } /* confidence: 65% */

/* original: mR8 */ function system_task_progress(q){
  Eo({
    type:"system",subtype:"task_progress",task_id:q.taskId,tool_use_id:q.toolUseId,description:q.description,usage:{
      total_tokens:q.totalTokens,tool_uses:q.toolUses,duration_ms:Date.now()-q.startTime
    },last_tool_name:q.lastToolName,summary:q.summary,workflow_progress:q.workflowProgress
  })
} /* confidence: 65% */

