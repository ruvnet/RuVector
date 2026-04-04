// Module: string

/* original: aP */ var aP="SendMessage";

/* original: M */ let composed_value=Q68($,{
  idleReason:"available",summary:n68(H)
} /* confidence: 30% */

/* original: n68 */ function typed_entity(q){
  for(let K=q.length-1;
  K>=0;
  K--){
    let _=q[K];
    if(!_)continue;
    if(_.type==="user"&&typeof _.message.content==="string")break;
    if(_.type!=="assistant")continue;
    for(let z of _.message.content)if(z.type==="tool_use"&&z.name===aP&&typeof z.input==="object"&&z.input!==null&&"to"in z.input&&typeof z.input.to==="string"&&z.input.to!=="*"&&z.input.to.toLowerCase()!==Hz.toLowerCase()&&"message"in z.input&&typeof z.input.message==="string"){
      let Y=z.input.to,$="summary"in z.input&&typeof z.input.summary==="string"?z.input.summary:z.input.message.slice(0,80);
      return`[to ${Y}] ${$}`
    }
  } /* confidence: 70% */

