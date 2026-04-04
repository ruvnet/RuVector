// Module: _anthropic_ai_claude_code

/* original: Y */ let composed_value=yv6(); /* confidence: 30% */

/* original: z */ let composed_value=yv6(); /* confidence: 30% */

/* original: _ */ let composed_value=yv6(); /* confidence: 30% */

/* original: _ */ let composed_value=yv6(); /* confidence: 30% */

/* original: _ */ let composed_value=yv6(); /* confidence: 30% */

/* original: q */ let composed_value=await jC("tengu_sm_compact_config",{
  
} /* confidence: 30% */

/* original: _ */ let composed_value=await aK7(); /* confidence: 30% */

/* original: v */ let composed_value=await hcK(); /* confidence: 30% */

/* original: yv6 */ function utility_fn(){
  return
} /* confidence: 40% */

/* original: cX_ */ function helper_fn(){
  return yv6()??{
    
  } /* confidence: 35% */

/* original: e8K */ function config(){
  try{
    let q=await jC("tengu_version_config",{
      minVersion:"0.0.0"
    });
    if(q.minVersion&&Er({
      ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"
    }.VERSION,q.minVersion))console.error(`
It looks like your version of Claude Code (${{ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"}.VERSION}) needs an update.
A newer version (${q.minVersion} or higher) is required to continue.

To update, please run:
    claude update

This will ensure you have access to the latest features and improvements.
`),eK(1)
  }catch(q){
    j6(q)
  }
} /* confidence: 95% */

/* original: K1K */ function config(){
  try{
    return await jC("tengu_max_version_config",{
      
    })
  }catch(q){
    return j6(q),{
      
    }
  }
} /* confidence: 95% */

/* original: OK8 */ function reporttheissueathttpsgithubcom(){
  let q=await $K8();
  if(q.min_version&&Er({
    ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"
  }.VERSION,q.min_version))return`Your version of Claude Code (${{ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"}.VERSION}) is too old for Remote Control.
Version ${q.min_version} or higher is required. Run \`claude update\` to update.`;
  return null
} /* confidence: 65% */

/* original: JCY */ function helper_fn(){
  if(!DJ6())return!1;
  return(await $K8()).should_show_app_upgrade_message
} /* confidence: 35% */

/* original: _i8 */ function helper_fn(q,K){
  let[_,z]=Jw7.default.useState(K);
  return Jw7.default.useEffect(()=>{
    jC(q,K).then(z)
  },[q,K]),_
} /* confidence: 35% */

