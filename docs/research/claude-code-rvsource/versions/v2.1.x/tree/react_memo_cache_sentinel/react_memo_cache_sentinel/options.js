// Module: options

/* original: F */ let agent_repl_main_thread=k26(),U=V26(A.options.tools),c=[],o=A.getAppState().tasks[_],q6=o&&o.type==="in_process_teammate"?o.permissionMode:"default",t={
  ...G,permissionMode:q6
} /* confidence: 65% */

/* original: $6 */ let composed_value=k26(),h6=V26(H.options.tools); /* confidence: 30% */

/* original: k26 */ function utility_fn(){
  return{
    toolUseCount:0,latestInputTokens:0,cumulativeOutputTokens:0,recentActivities:[]
  }
} /* confidence: 40% */

