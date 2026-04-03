// Module: permissionMode

/* original: z */ let composed_value=gW(); /* confidence: 30% */

/* original: j */ let composed_value=QT(q.options.mainLoopModel,gW()); /* confidence: 30% */

/* original: gW */ function utility_fn(){
  return G8.sdkBetas
} /* confidence: 40% */

/* original: I56 */ function helper_fn(q,K){
  let _=QT(q,gW());
   /* confidence: 35% */

/* original: vl8 */ function named_entity(q){
  let _=k7()?.outputStyle??Gk,z={
    type:"system",subtype:"init",cwd:Z8(),session_id:N8(),tools:q.tools.map((Y)=>I$7(Y.name)),mcp_servers:q.mcpClients.map((Y)=>({
      name:Y.name,status:Y.type
    })),model:q.model,permissionMode:q.permissionMode,slash_commands:q.commands.filter((Y)=>Y.userInvocable!==!1).map((Y)=>Y.name),apiKeySource:XA().source,betas:gW(),claude_code_version:{
      ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"
    }.VERSION,output_style:_,agents:q.agents.map((Y)=>Y.agentType),skills:q.skills.filter((Y)=>Y.userInvocable!==!1).map((Y)=>Y.name),plugins:q.plugins.map((Y)=>({
      name:Y.name,path:Y.path,source:Y.source
    })),uuid:IBY()
  };
   /* confidence: 70% */

