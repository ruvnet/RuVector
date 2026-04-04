// Module: includes

/* original: Z */ let response=f9Y({
  team_name:O
} /* confidence: 70% */

/* original: Z6 */ let options="",l6=`

If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full transcript at: ${kY()}`,K8=oq()?`

If this plan can be broken down into multiple independent tasks, consider using the ${ym} tool to create a team and parallelize the work.`:"",s6=i?`

User feedback on this plan: ${i}`:""; /* confidence: 70% */

/* original: jX_ */ function utility_fn(){
  return process.argv.includes("--agent-teams")
} /* confidence: 40% */

/* original: oq */ function helper_fn(){
  if(!c6(process.env.CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS)&&!jX_())return!1;
   /* confidence: 35% */

/* original: bPK */ function andpotentiallyassignedtoteamma(){
  let q=oq()?" and potentially assigned to teammates":"",K=oq()?"- Include enough detail in the description for another agent to understand and complete the task\n- New tasks are created with status 'pending' and no owner - use TaskUpdate with the `owner` parameter to assign them\n":"";
  return`Use this tool to create a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool

Use this tool proactively in these scenarios:

- Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
- Non-trivial and complex tasks - Tasks that require careful planning or multiple operations${q}
- Plan mode - When using plan mode, create a task list to track the work
- User explicitly requests todo list - When the user directly asks you to use the todo list
- User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
- After receiving new instructions - Immediately capture user requirements as tasks
- When you start working on a task - Mark it as in_progress BEFORE beginning work
- After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
- There is only a single, straightforward task
- The task is trivial and tracking it provides no organizational benefit
- The task can be completed in less than 3 trivial steps
- The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Task Fields

- **subject**: A brief, actionable title in imperative form (e.g., "Fix authentication bug in login flow")
- **description**: What needs to be done
- **activeForm** (optional): Present continuous form shown in the spinner when the task is in_progress (e.g., "Fixing authentication bug"). If omitted, the spinner shows the subject instead.

All tasks are created with status \`pending\`.

## Tips

- Create tasks with clear, specific subjects that describe the outcome
- After creating tasks, use TaskUpdate to set up dependencies (blocks/blockedBy) if needed
${K}- Check TaskList first to avoid creating duplicate tasks
`
} /* confidence: 65% */

/* original: lPK */ function idTaskidentifierusewithTaskGet(){
  let q=oq()?`- Before assigning tasks to teammates, to see what's available
`:"",K=oq()?"- **id**: Task identifier (use with TaskGet, TaskUpdate)":"- **id**: Task identifier (use with TaskGet, TaskUpdate)",_=oq()?`
## Teammate Workflow

When working as a teammate:
1. After completing your current task, call TaskList to find available work
2. Look for tasks with status 'pending', no owner, and empty blockedBy
3. **Prefer tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones
4. Claim an available task using TaskUpdate (set \`owner\` to your name), or wait for leader assignment
5. If blocked, focus on unblocking tasks or notify the team lead
`:"";
  return`Use this tool to list all tasks in the task list.

## When to Use This Tool

- To see what tasks are available to work on (status: 'pending', no owner, not blocked)
- To check overall progress on the project
- To find tasks that are blocked and need dependencies resolved
${q}- After completing a task, to check for newly unblocked work or claim the next available task
- **Prefer working on tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones

## Output

Returns a summary of each task:
${K}
- **subject**: Brief description of the task
- **status**: 'pending', 'in_progress', or 'completed'
- **owner**: Agent ID if assigned, empty if available
- **blockedBy**: List of open task IDs that must be resolved first (tasks with blockedBy cannot be claimed until dependencies resolve)

Use TaskGet with a specific task ID to view full details including description and comments.
${_}`
} /* confidence: 65% */

/* original: f9Y */ function helper_fn(q,K){
  if(!oq())return;
   /* confidence: 35% */

/* original: DAY */ function helper_fn(q){
  if(!oq())return[];
  return[]
} /* confidence: 35% */

/* original: clY */ function helper_fn(){
  if(!oq())return;
  let q=Lj();
  if(!q)return;
  if(hJ.includes(q))return FX[q];
  return
} /* confidence: 35% */

