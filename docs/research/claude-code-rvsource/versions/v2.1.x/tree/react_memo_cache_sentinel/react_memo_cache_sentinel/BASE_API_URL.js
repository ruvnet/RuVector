// Module: BASE_API_URL

/* original: A */ let headers=await O1.post(m7().TOKEN_URL,O,{
  headers:{
    "Content-Type":"application/json"
  },timeout:15000
} /* confidence: 70% */

/* original: C */ let Backgroundtasksdialogdismissed=m7(),g=`${Backgroundtasksdialogdismissed.MCP_PROXY_URL}${Backgroundtasksdialogdismissed.MCP_PROXY_PATH.replace("{server_id}",K.id)}`; /* confidence: 65% */

/* original: z */ let composed_value=m7().BASE_API_URL,Y; /* confidence: 30% */

/* original: _ */ let composed_value=VS(q)?m7().CLAUDEAI_SUCCESS_URL:m7().CONSOLE_SUCCESS_URL; /* confidence: 30% */

/* original: q */ let composed_value=m7().CLAUDEAI_SUCCESS_URL; /* confidence: 30% */

/* original: C */ let Backgroundtasksdialogdismissed=await Js1({
  oauthToken:z,sessionId:N8(),baseUrl:m7().BASE_API_URL
} /* confidence: 65% */

/* original: z */ let composed_value=process.env.VOICE_STREAM_BASE_URL||m7().BASE_API_URL.replace("https://","wss://").replace("http://","ws://"); /* confidence: 30% */

/* original: q */ let composed_value=process.env.ANTHROPIC_BASE_URL||m7().BASE_API_URL; /* confidence: 30% */

/* original: WL7 */ function utility_fn(){
  return"prod"
} /* confidence: 40% */

/* original: m7 */ function local_staging_prod(){
  let q=(()=>{
    switch(WL7()){
      case"local":return yh5();
      case"staging":return Nh5??PL7;
      case"prod":return PL7
    }
  } /* confidence: 65% */

