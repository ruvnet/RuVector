// Module: argv

/* original: E */ let composed_value=tY7({
  execPath:process.execPath,scriptArgs:RcK(),env:process.env,verbose:!1,sandbox:q.sandbox,permissionMode:q.permissionMode,onDebug:z
} /* confidence: 30% */

/* original: Pj */ function utility_fn(){
  return typeof Bun<"u"&&Array.isArray(Bun.embeddedFiles)&&Bun.embeddedFiles.length>0
} /* confidence: 40% */

/* original: jBz */ function claude_localbinclaude_localbin(){
  if(Pj()){
    try{
      return await O1K(process.execPath)
    }catch{
      
    }try{
      let q=await uA("claude");
      if(q)return q
    }catch{
      
    }try{
      return await M8().stat(zp(Aj6(),".local/bin/claude")),zp(Aj6(),".local/bin/claude")
    }catch{
      
    }return"native"
  }try{
    return process.argv[0]||"unknown"
  }catch{
    return"unknown"
  }
} /* confidence: 65% */

/* original: HBz */ function unknown_unknown(){
  try{
    if(Pj())return process.execPath||"unknown";
    return process.argv[1]||"unknown"
  } /* confidence: 65% */

/* original: pKK */ function helper_fn(){
  if(process.env[jL6])return process.env[jL6];
  return Pj()?process.execPath:process.argv[1]
} /* confidence: 35% */

/* original: jz7 */ function helper_fn(){
  let q=Pj(),K=Rl.map((Y)=>`mcp__claude-in-chrome__${Y.name}`),_={
    
  };
   /* confidence: 35% */

/* original: RcK */ function helper_fn(){
  if(Pj()||!process.argv[1])return[];
   /* confidence: 35% */

