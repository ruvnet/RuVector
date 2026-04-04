// Module: mode

/* original: ja8 */ function utility_fn(){
  return G8.needsAutoModeExitAttachment
} /* confidence: 40% */

/* original: pOY */ function auto_auto_mode_exit(q){
  if(!ja8())return[];
  if(q.getAppState().toolPermissionContext.mode==="auto"||(NGK?.isAutoModeActive()??!1))return s0(!1),[];
  return s0(!1),[{
    type:"auto_mode_exit"
  }]
} /* confidence: 65% */

