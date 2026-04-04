// Module: has

/* original: v */ let composed_value=K??(_x()?void 0:hU.agentType),k=composed_value===void 0,V; /* confidence: 30% */

/* original: _x */ function utility_fn(){
  return!1
} /* confidence: 40% */

/* original: MuY */ function helper_fn(){
  return _x()?`Calling ${H4} without a subagent_type creates a fork, which runs in the background and keeps its tool output out of your context â so you can keep chatting with the user while it works. Reach for it when research or multi-step implementation work would otherwise fill your context with raw output you won't need again. **If you ARE the fork** â execute directly; do not re-delegate.`:`Use the ${H4} tool with specialized agents when the task at hand matches the agent's description. Subagents are valuable for parallelizing independent queries or for protecting the main context window from excessive results, but they should not be used excessively when not needed. Importantly, avoid duplicating work that subagents are already doing - if you delegate research to a subagent, do not also perform the same searches yourself.`
} /* confidence: 35% */

/* original: PuY */ function helper_fn(q,K){
  let _=q.has(OO),z=K.length>0&&q.has(kM),Y=q.has(H4)||q.has(Rj),$=Sj()?`\`find\` or \`grep\` via the ${Yq} tool`:`the ${Z_} or ${$9}`,O=[_?`If you do not understand why the user has denied a tool call, use the ${OO} to ask them.`:null,g7()?null:"If you need the user to run a shell command themselves (e.g., an interactive login like `gcloud auth login`), suggest they type `! <command>` in the prompt â the `!` prefix runs the command in this session so its output lands directly in the conversation.",Y?MuY():null,...Y&&Wo6()&&!_x()?[`For simple, directed codebase searches (e.g. for a specific file/class/function) use ${$} directly.`,`For broader codebase exploration and deep research, use the ${H4} tool with subagent_type=${LU.agentType}. This is slower than using ${$} directly, so use this only when a simple, directed search proves to be insufficient or when your task will clearly require more than ${EJ4} queries.`]:[],z?`/<skill-name> (e.g., /commit) is shorthand for users to invoke a user-invocable skill. When executed, the skill gets expanded to a full prompt. Use the ${kM} tool to execute them. IMPORTANT: Only use ${kM} for skills listed in its user-invocable skills section - do not guess or use built-in CLI commands.`:null,xdK!==null&&z&&q.has(xdK)?XuY():null,null].filter((A)=>A!==null);
   /* confidence: 35% */

