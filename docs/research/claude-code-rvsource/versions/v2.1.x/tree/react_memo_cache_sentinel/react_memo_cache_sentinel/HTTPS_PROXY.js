// Module: HTTPS_PROXY

/* original: tcK */ var tcK=!1;

/* original: ecK */ function BashTool(){
  if(tcK)return;
  if(tcK=!0,c6(process.env.CLAUDE_CODE_USE_BEDROCK)||c6(process.env.CLAUDE_CODE_USE_VERTEX)||c6(process.env.CLAUDE_CODE_USE_FOUNDRY)||c6(process.env.CLAUDE_CODE_USE_ANTHROPIC_AWS))return;
  if(process.env.HTTPS_PROXY||process.env.https_proxy||process.env.HTTP_PROXY||process.env.http_proxy||process.env.ANTHROPIC_UNIX_SOCKET||process.env.CLAUDE_CODE_CLIENT_CERT||process.env.CLAUDE_CODE_CLIENT_KEY)return;
  let q=process.env.ANTHROPIC_BASE_URL||m7().BASE_API_URL;
  fetch(q,{
    method:"HEAD",signal:AbortSignal.timeout(1e4)
  }).catch(()=>{
    
  })
} /* confidence: 31% */

