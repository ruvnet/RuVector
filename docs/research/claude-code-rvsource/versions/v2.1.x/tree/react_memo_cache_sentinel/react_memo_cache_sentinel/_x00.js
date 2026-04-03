// Module: _x00

/* original: K */ let composed_value=ME1(q),_=xu(); /* confidence: 30% */

/* original: K */ let composed_value=xu(),_=v08(composed_value,q),z=ME1(_); /* confidence: 30% */

/* original: xu */ function team_NFC(){
  return(v08(hj(),"team")+XE1).normalize("NFC")
} /* confidence: 65% */

/* original: Wsq */ function helper_fn(q){
  let K=ME1(q),_=xu();
   /* confidence: 35% */

/* original: AP_ */ function helper_fn(q){
  if(q.includes("\x00"))throw new PD(`Null byte in path: "${q}"`);
  let K=ME1(q),_=xu();
  if(!K.startsWith(_))throw new PD(`Path escapes team memory directory: "${q}"`);
  let z=await Xsq(K);
  if(!await Psq(z))throw new PD(`Path escapes team memory directory via symlink: "${q}"`);
  return K
} /* confidence: 35% */

/* original: il6 */ function helper_fn(q){
  return T08()&&Wsq(q)
} /* confidence: 35% */

/* original: ixY */ function MemoComponent(q,K=!1){
  let _=hj(),z=xu(),Y=K?["## How to save memories","","Write each memory to its own file in the chosen directory (private or team, per the type's scope guidance) using this frontmatter format:","",...aR6,"","- Keep the name, description, and type fields in memory files up-to-date with the content","- Organize memory semantically by topic, not chronologically","- Update or remove memories that turn out to be wrong or outdated","- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one."]:["## How to save memories","","Saving a memory is a two-step process:","","**Step 1** â write the memory to its own file in the chosen directory (private or team, per the type's scope guidance) using this frontmatter format:","",...aR6,"",`**Step 2** â add a pointer to that file in the same directory's \`${GW}\`. Each directory (private and team) has its own \`${GW}\` index â each entry should be one line, under ~150 characters: \`- [Title](file.md) â one-line hook\`. They have no frontmatter. Never write memory content directly into a \`${GW}\`.`,"",`- Both \`${GW}\` indexes are loaded into your conversation context â lines after ${b56} will be truncated, so keep them concise`,"- Keep the name, description, and type fields in memory files up-to-date with the content","- Organize memory semantically by topic, not chronologically","- Update or remove memories that turn out to be wrong or outdated","- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one."];
  return["# Memory","",`You have a persistent, file-based memory system with two directories: a private directory at \`${_}\` and a shared team directory at \`${z}\`. ${EQK}`,"","You should build up this memory system over time so that future conversations can have a complete picture of who the user is, how they'd like to collaborate with you, what behaviors to avoid or repeat, and the context behind the work the user gives you.","","If the user explicitly asks you to remember something, save it immediately as whichever type fits best. If they ask you to forget something, find and remove the relevant entry.","","## Memory scope","","There are two scope levels:","",`- private: memories that are private between you and the current user. They persist across conversations with only this specific user and are stored at the root \`${_}\`.`,`- team: memories that are shared with and contributed by all of the users who work within this project directory. Team memories are synced at the beginning of every session and they are stored at \`${z}\`.`,"",...KZK,...hg8,"- You MUST avoid saving sensitive data within shared team memories. For example, never save API keys or user credentials.","",...Y,"","## When to access memories","- When memories (personal or team) seem relevant, or the user references prior work with them or others in their organization.","- You MUST access memory when the user explicitly asks you to check, recall, or remember.","- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.",J77,"",...Rg8,"","## Memory and other forms of persistence","Memory is one of several persistence mechanisms available to you as you assist the user in a given conversation. The distinction is often that memory can be recalled in future conversations and should not be used for persisting information that is only useful within the scope of the current conversation.","- When to use or update a plan instead of memory: If you are about to start a non-trivial implementation task and would like to reach alignment with the user on your approach you should use a Plan rather than saving this information to memory. Similarly, if you already have a plan within the conversation and you have changed your approach persist that change by updating the plan rather than saving a memory.","- When to use or update tasks instead of memory: When you need to break your work in current conversation into discrete steps or keep track of your progress use tasks instead of saving to memory. Tasks are great for persisting information about the work that needs to be done in the current conversation, but memory should be reserved for information that will be useful in future conversations.",...q??[],"",...WY7(_)].join(`
`)
} /* confidence: 87% */

