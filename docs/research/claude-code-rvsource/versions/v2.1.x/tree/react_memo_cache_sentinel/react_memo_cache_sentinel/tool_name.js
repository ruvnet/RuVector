// Module: tool_name

/* original: Xb7 */ var Xb7=()=>{
  
};

/* original: xD6 */ var composed_value=L(()=>{
  G31();
  Xb7()
} /* confidence: 30% */

/* original: io */ var composed_value=L(()=>{
  xD6();
  _8();
  _w8()
} /* confidence: 30% */

/* original: qu4 */ var composed_value=L(()=>{
  xD6();
  _8();
  io()
} /* confidence: 30% */

/* original: m9K */ var composed_value=L(()=>{
  xD6();
  _8();
  io()
} /* confidence: 30% */

/* original: RC6 */ var composed_value=L(()=>{
  T8();
  pA();
  i1();
  io()
} /* confidence: 30% */

/* original: imK */ var permission_handler=L(()=>{
  c4();
  T8();
  RC6();
  Qd8=$1(function(q){
    return{
      PreToolUse:{
        summary:"Before tool execution",description:`Input to command is JSON of tool call arguments.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and block tool call
Other exit codes - show stderr to user only but continue with tool call`,matcherMetadata:{
          fieldToMatch:"tool_name",values:q
        }
      },PostToolUse:{
        summary:"After tool execution",description:`Input to command is JSON with fields "inputs" (tool call arguments) and "response" (tool call response).
Exit code 0 - stdout shown in transcript mode (ctrl+o)
Exit code 2 - show stderr to model immediately
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"tool_name",values:q
        }
      },PostToolUseFailure:{
        summary:"After tool execution fails",description:`Input to command is JSON with tool_name, tool_input, tool_use_id, error, error_type, is_interrupt, and is_timeout.
Exit code 0 - stdout shown in transcript mode (ctrl+o)
Exit code 2 - show stderr to model immediately
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"tool_name",values:q
        }
      },PermissionDenied:{
        summary:"After auto mode classifier denies a tool call",description:`Input to command is JSON with tool_name, tool_input, tool_use_id, and reason.
Return {"hookSpecificOutput":{"hookEventName":"PermissionDenied","retry":true}} to tell the model it may retry.
Exit code 0 - stdout shown in transcript mode (ctrl+o)
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"tool_name",values:q
        }
      },Notification:{
        summary:"When notifications are sent",description:`Input to command is JSON with notification message and type.
Exit code 0 - stdout/stderr not shown
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"notification_type",values:["permission_prompt","idle_prompt","auth_success","elicitation_dialog","elicitation_complete","elicitation_response"]
        }
      },UserPromptSubmit:{
        summary:"When the user submits a prompt",description:`Input to command is JSON with original user prompt text.
Exit code 0 - stdout shown to Claude
Exit code 2 - block processing, erase original prompt, and show stderr to user only
Other exit codes - show stderr to user only`
      },SessionStart:{
        summary:"When a new session is started",description:`Input to command is JSON with session start source.
Exit code 0 - stdout shown to Claude
Blocking errors are ignored
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"source",values:["startup","resume","clear","compact"]
        }
      },Stop:{
        summary:"Right before Claude concludes its response",description:`Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and continue conversation
Other exit codes - show stderr to user only`
      },StopFailure:{
        summary:"When the turn ends due to an API error",description:"Fires instead of Stop when an API error (rate limit, auth failure, etc.) ended the turn. Fire-and-forget â hook output and exit codes are ignored.",matcherMetadata:{
          fieldToMatch:"error",values:["rate_limit","authentication_failed","billing_error","invalid_request","server_error","max_output_tokens","unknown"]
        }
      },SubagentStart:{
        summary:"When a subagent (Agent tool call) is started",description:`Input to command is JSON with agent_id and agent_type.
Exit code 0 - stdout shown to subagent
Blocking errors are ignored
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"agent_type",values:[]
        }
      },SubagentStop:{
        summary:"Right before a subagent (Agent tool call) concludes its response",description:`Input to command is JSON with agent_id, agent_type, and agent_transcript_path.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to subagent and continue having it run
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"agent_type",values:[]
        }
      },PreCompact:{
        summary:"Before conversation compaction",description:`Input to command is JSON with compaction details.
Exit code 0 - stdout appended as custom compact instructions
Exit code 2 - block compaction
Other exit codes - show stderr to user only but continue with compaction`,matcherMetadata:{
          fieldToMatch:"trigger",values:["manual","auto"]
        }
      },PostCompact:{
        summary:"After conversation compaction",description:`Input to command is JSON with compaction details and the summary.
Exit code 0 - stdout shown to user
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"trigger",values:["manual","auto"]
        }
      },SessionEnd:{
        summary:"When a session is ending",description:`Input to command is JSON with session end reason.
Exit code 0 - command completes successfully
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"reason",values:["clear","logout","prompt_input_exit","other"]
        }
      },PermissionRequest:{
        summary:"When a permission dialog is displayed",description:`Input to command is JSON with tool_name, tool_input, and tool_use_id.
Output JSON with hookSpecificOutput containing decision to allow or deny.
Exit code 0 - use hook decision if provided
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"tool_name",values:q
        }
      },Setup:{
        summary:"Repo setup hooks for init and maintenance",description:`Input to command is JSON with trigger (init or maintenance).
Exit code 0 - stdout shown to Claude
Blocking errors are ignored
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"trigger",values:["init","maintenance"]
        }
      },TeammateIdle:{
        summary:"When a teammate is about to go idle",description:`Input to command is JSON with teammate_name and team_name.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to teammate and prevent idle (teammate continues working)
Other exit codes - show stderr to user only`
      },TaskCreated:{
        summary:"When a task is being created",description:`Input to command is JSON with task_id, task_subject, task_description, teammate_name, and team_name.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and prevent task creation
Other exit codes - show stderr to user only`
      },TaskCompleted:{
        summary:"When a task is being marked as completed",description:`Input to command is JSON with task_id, task_subject, task_description, teammate_name, and team_name.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and prevent task completion
Other exit codes - show stderr to user only`
      },Elicitation:{
        summary:"When an MCP server requests user input (elicitation)",description:`Input to command is JSON with mcp_server_name, message, and requested_schema.
Output JSON with hookSpecificOutput containing action (accept/decline/cancel) and optional content.
Exit code 0 - use hook response if provided
Exit code 2 - deny the elicitation
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"mcp_server_name",values:[]
        }
      },ElicitationResult:{
        summary:"After a user responds to an MCP elicitation",description:`Input to command is JSON with mcp_server_name, action, content, mode, and elicitation_id.
Output JSON with hookSpecificOutput containing optional action and content to override the response.
Exit code 0 - use hook response if provided
Exit code 2 - block the response (action becomes decline)
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"mcp_server_name",values:[]
        }
      },ConfigChange:{
        summary:"When configuration files change during a session",description:`Input to command is JSON with source (user_settings, project_settings, local_settings, policy_settings, skills) and file_path.
Exit code 0 - allow the change
Exit code 2 - block the change from being applied to the session
Other exit codes - show stderr to user only`,matcherMetadata:{
          fieldToMatch:"source",values:["user_settings","project_settings","local_settings","policy_settings","skills"]
        }
      },InstructionsLoaded:{
        summary:"When an instruction file (CLAUDE.md or rule) is loaded",description:`Input to command is JSON with file_path, memory_type (User, Project, Local, Managed), load_reason (session_start, nested_traversal, path_glob_match, include, compact), globs (optional â the paths: frontmatter patterns that matched), trigger_file_path (optional â the file Claude touched that caused the load), and parent_file_path (optional â the file that @-included this one).
Exit code 0 - command completes successfully
Other exit codes - show stderr to user only
This hook is observability-only and does not support blocking.`,matcherMetadata:{
          fieldToMatch:"load_reason",values:["session_start","nested_traversal","path_glob_match","include","compact"]
        }
      },WorktreeCreate:{
        summary:"Create an isolated worktree for VCS-agnostic isolation",description:`Input to command is JSON with name (suggested worktree slug).
Stdout should contain the absolute path to the created worktree directory.
Exit code 0 - worktree created successfully
Other exit codes - worktree creation failed`
      },WorktreeRemove:{
        summary:"Remove a previously created worktree",description:`Input to command is JSON with worktree_path (absolute path to worktree).
Exit code 0 - worktree removed successfully
Other exit codes - show stderr to user only`
      },CwdChanged:{
        summary:"After the working directory changes",description:`Input to command is JSON with old_cwd and new_cwd.
CLAUDE_ENV_FILE is set â write bash exports there to apply env to subsequent BashTool commands.
Hook output can include hookSpecificOutput.watchPaths (array of absolute paths) to register with the FileChanged watcher.
Exit code 0 - command completes successfully
Other exit codes - show stderr to user only`
      },FileChanged:{
        summary:"When a watched file changes",description:`Input to command is JSON with file_path and event (change, add, unlink).
CLAUDE_ENV_FILE is set â write bash exports there to apply env to subsequent BashTool commands.
The matcher field specifies filenames to watch in the current directory (e.g. ".envrc|.env").
Hook output can include hookSpecificOutput.watchPaths (array of absolute paths) to dynamically update the watch list.
Exit code 0 - command completes successfully
Other exit codes - show stderr to user only`
      }
    }
  },(q)=>q.slice().sort().join(","))
} /* confidence: 95% */

/* original: smK */ var composed_value=L(()=>{
  t6();
  i6();
  RC6();
  j3();
  x4();
  R0=w6(D6(),1)
} /* confidence: 30% */

/* original: emK */ var composed_value=L(()=>{
  t6();
  i6();
  RC6();
  j3();
  x4();
  S0=w6(D6(),1)
} /* confidence: 30% */

/* original: KpK */ var composed_value=L(()=>{
  t6();
  i6();
  RC6();
  x4();
  n9=w6(D6(),1)
} /* confidence: 30% */

/* original: zeK */ var composed_value=L(()=>{
  _8();
  io();
  CH();
  r8();
  fY();
  uJ();
  eD()
} /* confidence: 30% */

/* original: K */ let composed_value=XeY(),_=WeY; /* confidence: 30% */

/* original: XeY */ function helper_fn(){
  let q=Bl(rW(),{
    io:"input"
  });
  return g6(q,null,2)
} /* confidence: 35% */

/* original: q45 */ function config(){
  ww({
    name:"update-config",description:'Use this skill to configure the Claude Code harness via settings.json. Automated behaviors ("from now on when X", "each time X", "whenever X", "before/after X") require hooks configured in settings.json - the harness executes these, not Claude, so memory/preferences cannot fulfill them. Also use for: permissions ("allow X", "add permission", "move permission to"), env vars ("set X=Y"), hook troubleshooting, or any changes to settings.json/settings.local.json files. Examples: "allow npm commands", "add bq permission to global settings", "move permission to user settings", "set DEBUG=true", "when claude stops show X". For simple settings like theme/model, use Config tool.',allowedTools:["Read"],userInvocable:!0,async getPromptForCommand(q){
      if(q.startsWith("[hooks-only]")){
        let z=q.slice(12).trim(),Y=tq5+`

`+eq5;
        if(z)Y+=`

## Task

${z}`;
        return[{
          type:"text",text:Y
        }]
      }let K=XeY(),_=WeY;
      if(_+=`

## Full Settings JSON Schema

\`\`\`json
${K}
\`\`\``,q)_+=`

## User Request

${q}`;
      return[{
        type:"text",text:_
      }]
    }
  } /* confidence: 95% */

