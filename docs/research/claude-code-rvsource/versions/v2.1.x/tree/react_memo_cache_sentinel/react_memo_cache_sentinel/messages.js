// Module: messages

/* original: wKK */ var wKK={
  
};

/* original: nwY */ function helper_fn(){
  return uJ(),hq(wKK)
} /* confidence: 35% */

/* original: Yq7 */ function typed_entity(q){
  if(oq()){
    if(q.type==="teammate_mailbox")return[n8({
      content:nwY().formatTeammateMessages(q.messages),isMeta:!0
    })];
    if(q.type==="team_context")return[n8({
      content:`<system-reminder>
# Team Coordination

You are a teammate in team "${q.teamName}".

**Your Identity:**
- Name: ${q.agentName}

**Team Resources:**
- Team config: ${q.teamConfigPath}
- Task list: ${q.taskListPath}

**Team Leader:** The team lead's name is "team-lead". Send updates and completion notifications to them.

Read the team config to discover your teammates' names. Check the task list periodically. Create new tasks when work should be divided. Mark tasks resolved when complete.

**IMPORTANT:** Always refer to teammates by their NAME (e.g., "team-lead", "analyzer", "researcher"), never by UUID. When messaging, use the name directly:

\`\`\`json
{
  "to": "team-lead",
  "message": "Your message here",
  "summary": "Brief 5-10 word preview"
}
\`\`\`
</system-reminder>`,isMeta:!0
    })]
  } /* confidence: 70% */

