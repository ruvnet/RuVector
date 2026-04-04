// Module: enterprise

/* original: I6 */ let UseRefHook=AhK(N6); /* confidence: 46% */

/* original: Ch8 */ function utility_fn(){
  return Sh8(bP(),"managed-mcp.json")
} /* confidence: 40% */

/* original: _v */ function config(q){
  switch(q){
    case"user":return xP();
    case"project":return BJz(Z8(),".mcp.json");
    case"local":return`${xP()} [project: ${Z8()}]`;
    case"dynamic":return"Dynamically configured";
    case"enterprise":return Ch8();
    case"claudeai":return"claude.ai";
    default:return q
  } /* confidence: 95% */

/* original: AhK */ function project_ProjectMCPs_user_UserM(q){
  switch(q){
    case"project":return{
      label:"Project MCPs",path:_v(q)
    };
    case"user":return{
      label:"User MCPs",path:_v(q)
    };
    case"local":return{
      label:"Local MCPs",path:_v(q)
    };
    case"enterprise":return{
      label:"Enterprise MCPs"
    };
    case"dynamic":return{
      label:"Built-in MCPs",path:"always available"
    };
    default:return{
      label:q
    }
  }
} /* confidence: 65% */

