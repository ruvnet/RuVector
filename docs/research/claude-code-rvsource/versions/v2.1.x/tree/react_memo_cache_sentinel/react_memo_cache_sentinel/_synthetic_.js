// Module: _synthetic_

/* original: iwY */ var action_creator=`

Note: The user's next message may contain a correction or preference. Pay close attention â if they explain what went wrong or how they'd prefer you to work, consider saving that to memory for future sessions.`,IvK="Tool loaded.",K46="[Request interrupted by user]",gZ="[Request interrupted by user for tool use]",sj6="The user doesn't want to take this action right now. STOP what you are doing and wait for the user to tell you how to proceed.",tj6="The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.",ju8=`The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). To tell you how to proceed, the user said:
`,jp="Permission for this tool use was denied. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). Try a different approach or report the limitation to complete your task.",o68=`Permission for this tool use was denied. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). The user said:
`,va1=`The agent proposed a plan that was rejected by the user. The user chose to stay in plan mode rather than proceed with implementation.

Rejected plan:
`,A47="IMPORTANT: You *may* attempt to accomplish this action using other tools that might naturally be used to accomplish this goal, e.g. using head instead of cat. But you *should not* attempt to work around this denial in malicious ways, e.g. do not use your ability to run tests to execute non-test actions. You should only try to work around this restriction in reasonable ways that do not attempt to bypass the intent behind this denial. If you believe this capability is essential to complete the user's request, STOP and explain to the user what you were trying to do and why you need this permission. Let the user decide how to proceed.",Fj6="No response requested.",rwY="[Tool result missing due to internal error]",FvK="Permission for this action has been denied. Reason: ",Dw6="<synthetic>",Sq6,da,BF8,O2Y,mvK=`### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Begin with a **Context** section: explain why this change is being made â the problem or need it addresses, what prompted it, and the intended outcome
- Include only your recommended approach, not all alternatives
- Ensure that the plan file is concise enough to scan quickly, but detailed enough to execute effectively
- Include the paths of critical files to be modified
- Reference existing functions and utilities you found that should be reused, with their file paths
- Include a verification section describing how to test the changes end-to-end (run the code, use MCP tools, run tests)`,w2Y=`### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- One-line **Context**: what is being changed and why
- Include only your recommended approach, not all alternatives
- List the paths of files to be modified
- Reference existing functions and utilities to reuse, with their file paths
- End with **Verification**: the single command to run to confirm the change works (no numbered test procedures)`,j2Y=`### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Do NOT write a Context or Background section. The user just told you what they want.
- List the paths of files to be modified and what changes in each (one line per file)
- Reference existing functions and utilities to reuse, with their file paths
- End with **Verification**: the single command that confirms the change works
- Most good plans are under 40 lines. Prose is a sign you are padding.`,H2Y=`### Phase 4: Final Plan
Goal: Write your final plan to the plan file (the only file you can edit).
- Do NOT write a Context, Background, or Overview section. The user just told you what they want.
- Do NOT restate the user's request. Do NOT write prose paragraphs.
- List the paths of files to be modified and what changes in each (one bullet per file)
- Reference existing functions to reuse, with file:line
- End with the single verification command
- **Hard limit: 40 lines.** If the plan is longer, delete prose â not file paths.`; /* confidence: 95% */

/* original: rR6 */ function helper_fn(q){
  if(i3()&&L8("tengu_amber_prism",!1))return q+iwY;
   /* confidence: 35% */

