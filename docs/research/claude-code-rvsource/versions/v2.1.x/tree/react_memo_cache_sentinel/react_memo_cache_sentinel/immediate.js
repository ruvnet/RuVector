// Module: immediate

/* original: NrY */ function chrome_nochrome(){
  if(process.argv.includes("--chrome"))return!0;
  if(process.argv.includes("--no-chrome"))return!1;
  return
} /* confidence: 65% */

/* original: yrY */ function chromerequiressubscription_err(){
  let q=NrY();
  if(!Ac8(q))return null;
  if(!i7())return{
    key:"chrome-requires-subscription",jsx:p58.createElement(T,{
      color:"error"
    },"Claude in Chrome requires a claude.ai subscription"),priority:"immediate",timeoutMs:5000
  };
  if(!await zt()&&!iv())return{
    key:"chrome-extension-not-detected",jsx:p58.createElement(T,{
      color:"warning"
    },"Chrome extension not detected Â· https://claude.ai/chrome to install"),priority:"immediate",timeoutMs:3000
  };
  if(q===void 0)return{
    key:"claude-in-chrome-default-enabled",text:"Claude in Chrome enabled Â· /chrome",priority:"low"
  };
  return null
} /* confidence: 65% */

