// Module: tool_use

/* original: Jb */ var WriteTool="TodoWrite"; /* confidence: 31% */

/* original: z */ let composed_value=_.message.content.find((O)=>O.type==="tool_use"&&O.name===Jb); /* confidence: 30% */

/* original: biY */ function typed_entity(q){
  for(let K=q.length-1;
  K>=0;
  K--){
    let _=q[K];
    if(_?.type!=="assistant")continue;
    let z=_.message.content.find((O)=>O.type==="tool_use"&&O.name===Jb);
    if(!z||z.type!=="tool_use")continue;
    let Y=z.input;
    if(Y===null||typeof Y!=="object")return[];
    let $=nL6().safeParse(Y.todos);
    return $.success?$.data:[]
  }return[]
} /* confidence: 70% */

