// Module: number

/* original: gz4 */ var x1B_x1B_x07__Apple_Terminal_nu=B((im_,gb1)=>{
  var GY=im_;
  im_.default=GY;
  var DA="\x1B[",Dr6="\x1B]",Tk6="\x07",sk8=";",Bz4=process.env.TERM_PROGRAM==="Apple_Terminal";
  GY.cursorTo=(q,K)=>{
    if(typeof q!=="number")throw TypeError("The `x` argument is required");
    if(typeof K!=="number")return DA+(q+1)+"G";
    return DA+(K+1)+";"+(q+1)+"H"
  };
  GY.cursorMove=(q,K)=>{
    if(typeof q!=="number")throw TypeError("The `x` argument is required");
    let _="";
    if(q<0)_+=DA+-q+"D";
    else if(q>0)_+=DA+q+"C";
    if(K<0)_+=DA+-K+"A";
    else if(K>0)_+=DA+K+"B";
    return _
  };
  GY.cursorUp=(q=1)=>DA+q+"A";
  GY.cursorDown=(q=1)=>DA+q+"B";
  GY.cursorForward=(q=1)=>DA+q+"C";
  GY.cursorBackward=(q=1)=>DA+q+"D";
  GY.cursorLeft=DA+"G";
  GY.cursorSavePosition=Bz4?"\x1B7":DA+"s";
  GY.cursorRestorePosition=Bz4?"\x1B8":DA+"u";
  GY.cursorGetPosition=DA+"6n";
  GY.cursorNextLine=DA+"E";
  GY.cursorPrevLine=DA+"F";
  GY.cursorHide=DA+"?25l";
  GY.cursorShow=DA+"?25h";
  GY.eraseLines=(q)=>{
    let K="";
    for(let _=0;
    _<q;
    _++)K+=GY.eraseLine+(_<q-1?GY.cursorUp():"");
    if(q)K+=GY.cursorLeft;
    return K
  };
  GY.eraseEndLine=DA+"K";
  GY.eraseStartLine=DA+"1K";
  GY.eraseLine=DA+"2K";
  GY.eraseDown=DA+"J";
  GY.eraseUp=DA+"1J";
  GY.eraseScreen=DA+"2J";
  GY.scrollUp=DA+"S";
  GY.scrollDown=DA+"T";
  GY.clearScreen="\x1Bc";
  GY.clearTerminal=process.platform==="win32"?`${GY.eraseScreen}${DA}0f`:`${GY.eraseScreen}${DA}3J${DA}H`;
  GY.beep=Tk6;
  GY.link=(q,K)=>{
    return[Dr6,"8",sk8,sk8,K,Tk6,q,Dr6,"8",sk8,sk8,Tk6].join("")
  };
  GY.image=(q,K={
    
  })=>{
    let _=`${Dr6}1337;File=inline=1`;
    if(K.width)_+=`;width=${K.width}`;
    if(K.height)_+=`;height=${K.height}`;
    if(K.preserveAspectRatio===!1)_+=";preserveAspectRatio=0";
    return _+":"+q.toString("base64")+Tk6
  };
  GY.iTerm={
    setCwd:(q=process.cwd())=>`${Dr6}50;CurrentDir=${q}${Tk6}`,annotation:(q,K={
      
    })=>{
      let _=`${Dr6}1337;`,z=typeof K.x<"u",Y=typeof K.y<"u";
      if((z||Y)&&!(z&&Y&&typeof K.length<"u"))throw Error("`x`, `y` and `length` must be defined when `x` or `y` is defined");
      if(q=q.replace(/\|/g,""),_+=K.isHidden?"AddHiddenAnnotation=":"AddAnnotation=",K.length>0)_+=(z?[q,K.length,K.x,K.y]:[K.length,q]).join("|");
      else _+=q;
      return _+Tk6
    }
  }
} /* confidence: 65% */

/* original: DA */ var x1B_x1B_x07__Apple_Terminal="\x1B[",Dr6="\x1B]",Tk6="\x07",sk8=";",Bz4=process.env.TERM_PROGRAM==="Apple_Terminal"; /* confidence: 65% */

