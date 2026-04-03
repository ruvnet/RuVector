// Module: Ghostty

/* original: $A8 */ var $A8=()=>{
  
};

/* original: AT */ var composed_value=L(()=>{
  SA8();
  $A8()
} /* confidence: 30% */

/* original: Q51 */ var composed_value=L(()=>{
  F7();
  AT();
  r8()
} /* confidence: 30% */

/* original: PK */ var composed_value=L(()=>{
  SA8();
  F7();
  h8();
  $A8();
  Q51()
} /* confidence: 30% */

/* original: cU6 */ var composed_value=L(()=>{
  GY6();
  AT()
} /* confidence: 30% */

/* original: TO6 */ var composed_value=L(()=>{
  PK()
} /* confidence: 30% */

/* original: n16 */ var env_config=L(()=>{
  c4();
  R_();
  d8();
  PK();
  TO6();
  _X_=$1(async()=>{
    if(process.platform!=="linux")return!1;
    let{
      code:q
    }=await K1("test",["-f","/.dockerenv"]);
    return q===0
  });
  if(process.platform==="linux"){
    let q=process.arch==="x64"?"x86_64":"aarch64";
    KX_(`/lib/libc.musl-${q}.so.1`).then(()=>{
      gy1=!0
    },()=>{
      gy1=!1
    })
  }WN={
    ...Y7,terminal:OX_(),getIsDocker:_X_,getIsBubblewrapSandbox:zX_,isMuslEnvironment:YX_,getTerminalWithJetBrainsDetectionAsync:$X_,initJetBrainsDetection:Fy1
  }
} /* confidence: 95% */

/* original: xH */ var composed_value=L(()=>{
  PK()
} /* confidence: 30% */

/* original: Ey6 */ var composed_value=L(()=>{
  l1();
  _p1();
  _8();
  PK();
  e7();
  zb();
  h8();
  AT();
  OR8=/\.(png|jpe?g|gif|webp)$/i
} /* confidence: 30% */

/* original: nKK */ var composed_value=L(()=>{
  T8();
  R9();
  _8();
  r8();
  uJ();
  Vo1();
  e68();
  bb()
} /* confidence: 30% */

/* original: tKK */ var composed_value=L(()=>{
  _8();
  PK();
  h8();
  bb();
  pb();
  rKK=Promise.resolve();
  ho1(Lo1)
} /* confidence: 30% */

/* original: K5K */ var composed_value=L(()=>{
  _8();
  PK();
  bb();
  pb();
  wd=[],eKK=Promise.resolve();
  So1(Ro1)
} /* confidence: 30% */

/* original: pb */ var composed_value=L(()=>{
  T8();
  _8();
  NK();
  bb();
  SKK();
  Go1();
  nKK();
  bj6()
} /* confidence: 30% */

/* original: Tp8 */ var composed_value=L(()=>{
  F7();
  jG();
  PK();
  yK();
  P5()
} /* confidence: 30% */

/* original: NB8 */ var composed_value=L(()=>{
  T8();
  Tw();
  _8();
  TO6();
  AMK();
  t4()
} /* confidence: 30% */

/* original: I47 */ var composed_value=L(()=>{
  Z$();
  Lm();
  Ey6();
  NS6=w6(D6(),1)
} /* confidence: 30% */

/* original: lVK */ var composed_value=L(()=>{
  T8();
  _8();
  PK();
  yK();
  pq8=w6(OT6(),1)
} /* confidence: 30% */

/* original: UK7 */ var composed_value=L(()=>{
  t6();
  i6();
  xH();
  lVK();
  E8();
  AO();
  t4();
  WJ6();
  Fy=w6(D6(),1)
} /* confidence: 30% */

/* original: gyK */ var composed_value=L(()=>{
  Tp8();
  _36=w6(D6(),1)
} /* confidence: 30% */

/* original: nLK */ var composed_value=L(()=>{
  k8();
  k1();
  xH();
  PK();
  h8()
} /* confidence: 30% */

/* original: KhK */ var composed_value=L(()=>{
  k8();
  xH();
  k1()
} /* confidence: 30% */

/* original: E48 */ var composed_value=L(()=>{
  k8();
  PK();
  P5()
} /* confidence: 30% */

/* original: JIK */ var composed_value=L(()=>{
  l1();
  k8();
  OIK();
  II8();
  q56();
  T7();
  jG();
  PK();
  P5();
  lb()
} /* confidence: 30% */

/* original: SgK */ var composed_value=L(()=>{
  xH()
} /* confidence: 30% */

/* original: Cz7 */ var composed_value=L(()=>{
  AT();
  VV()
} /* confidence: 30% */

/* original: JUK */ var composed_value=L(()=>{
  b_();
  x4();
  WJ6();
  i6();
  k8();
  xH();
  Cz7();
  AT();
  jUK();
  x0=w6(D6(),1),Wc8=w6(D6(),1)
} /* confidence: 30% */

/* original: ZsK */ var composed_value=L(()=>{
  PK();
  P5();
  r8()
} /* confidence: 30% */

/* original: W55 */ var iTerm2_comgooglecodeiterm2_iTe=L(()=>{
  k1();
  _8();
  PK();
  VV();
  Fi8=[{
    name:"iTerm2",bundleId:"com.googlecode.iterm2",app:"iTerm"
  },{
    name:"Ghostty",bundleId:"com.mitchellh.ghostty",app:"Ghostty"
  },{
    name:"Kitty",bundleId:"net.kovidgoyal.kitty",app:"kitty"
  },{
    name:"Alacritty",bundleId:"org.alacritty",app:"Alacritty"
  },{
    name:"WezTerm",bundleId:"com.github.wez.wezterm",app:"WezTerm"
  },{
    name:"Terminal.app",bundleId:"com.apple.Terminal",app:"Terminal"
  }],y6$=["ghostty","kitty","alacritty","wezterm","gnome-terminal","konsole","xfce4-terminal","mate-terminal","tilix","xterm"]
} /* confidence: 65% */

