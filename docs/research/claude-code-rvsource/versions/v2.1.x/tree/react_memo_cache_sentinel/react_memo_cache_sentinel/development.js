// Module: development

/* original: OaY */ var OaY="Claude Code has switched from npm to native installer. Run `claude install` or see https://docs.anthropic.com/en/docs/claude-code/getting-started for more options.";

/* original: a85 */ function helper_fn(){
  Cx(AaY)
} /* confidence: 35% */

/* original: AaY */ function development_npmdeprecationwarn(){
  if(Pj()||c6(process.env.DISABLE_INSTALLATION_CHECKS))return null;
  if(await Ga()==="development")return null;
  return{
    timeoutMs:15000,key:"npm-deprecation-warning",text:OaY,color:"warning",priority:"high"
  }
} /* confidence: 65% */

