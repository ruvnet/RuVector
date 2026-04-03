// Module: deny

/* original: N4 */ var Edit_claude_claude_Filehasnotb="Edit",B08="/.claude/**",g08="~/.claude/**",F08="File has not been read yet. Read it first before writing to it.",U08="File content has changed since it was last read. This commonly happens when a linter or formatter run via Bash rewrites the file. Call Read on this file to refresh, then retry the edit."; /* confidence: 65% */

/* original: _ */ let composed_value=KfK(q,K); /* confidence: 30% */

/* original: O */ let composed_value=KfK(q.tool_name,q.tool_input); /* confidence: 30% */

/* original: $ */ let composed_value=pQK(K,_,z); /* confidence: 30% */

/* original: rh6 */ function read_deny(q){
  let K=pQK(q,"read","deny"),_=new Map;
   /* confidence: 65% */

/* original: s6$ */ function named_entity(q){
  return async(K,_)=>{
    if(K.name===N4&&typeof _==="object"&&_!==null&&"file_path"in _){
      let z=_.file_path;
      if(typeof z==="string"&&z===q)return{
        behavior:"allow",updatedInput:_
      }
    }return{
      behavior:"deny",message:`only ${N4} on ${q} is allowed`,decisionReason:{
        type:"other",reason:`only ${N4} on ${q} is allowed`
      }
    }
  } /* confidence: 70% */

