// Module: split

/* original: Q05 */ var composed_value=/^(?:([^#/:?]+):)?(?:\/\/((?:([^#/?@]*)@)?(\[[^#/?\]]+\]|[^#/:?]*)(?::(\d*))?))?([^#?]*)(?:\?([^#]*))?(?:#((?:.|[\n\r])*))?/u; /* confidence: 30% */

/* original: O */ let composed_value=q.match(Q05); /* confidence: 30% */

/* original: _ */ let composed_value=q.split(/\r?\n/); /* confidence: 30% */

/* original: z */ let composed_value=_.stdout.trim().split(/\r?\n/).filter(Boolean); /* confidence: 30% */

/* original: J6 */ let composed_value=function(s){
  s=s.replace(/\r\n?/g,`
`);
   /* confidence: 30% */

/* original: _ */ let composed_value=K.split(/\n|\r\n/); /* confidence: 30% */

/* original: dP_ */ var _=void 0,hE1="",Csq=" ",LE1="\\",cP_=/^\s+$/,lP_=/(?:[^\\]|^)\\$/,nP_=/^\\!/,iP_=/^\\#/,rP_=/\r?\n/g,oP_=/^\.{
  0,2
} /* confidence: 65% */

/* original: Q8 */ var composed_value=r&134217727; /* confidence: 30% */

/* original: f6 */ var composed_value=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,"name"); /* confidence: 30% */

/* original: f6 */ var composed_value=32-ME(r)-1; /* confidence: 30% */

/* original: E6 */ var composed_value=A4.bind(null,s9,!1,r.queue); /* confidence: 30% */

/* original: Q8 */ var composed_value=r.stack; /* confidence: 30% */

/* original: E6 */ var composed_value=Error(z(520),{
  cause:r
} /* confidence: 30% */

/* original: Q8 */ var composed_value=r.children; /* confidence: 30% */

/* original: Q */ var composed_value=S.containerInfo,r=h27(); /* confidence: 30% */

/* original: E6 */ var composed_value=r.stateNode; /* confidence: 30% */

/* original: Q */ var composed_value=I(Ol),r=32>composed_value?32:composed_value; /* confidence: 30% */

/* original: Q */ var composed_value=X5.T,r=j8(); /* confidence: 30% */

/* original: Q */ var composed_value=0,r=[]; /* confidence: 30% */

/* original: Y */ let composed_value=W74(q).split(/\r?\n/); /* confidence: 30% */

/* original: $ */ let composed_value=Y-1,O=q.substring(K.lineStarts[z-1],K.lineStarts[z]).replace(/[\n\r]+composed_value/,""); /* confidence: 30% */

/* original: _ */ let composed_value=[],z=q.split(/(\n|\r\n)/); /* confidence: 30% */

/* original: H */ var composed_value=_.normalizeLineEndings||N7K; /* confidence: 30% */

/* original: _ */ var composed_value=K.replace(/(^[ \t\r\n\f]+)|([ \t\r\n\f]+$)/g,""); /* confidence: 30% */

/* original: K */ var composed_value=W0.pattern.exec(this.url); /* confidence: 30% */

/* original: _ */ var composed_value=N4Y(q.textContent); /* confidence: 30% */

/* original: iNK */ var composed_value=L(()=>{
  cNK={
    30:{
      r:0,g:0,b:0
    },31:{
      r:205,g:49,b:49
    },32:{
      r:13,g:188,b:121
    },33:{
      r:229,g:229,b:16
    },34:{
      r:36,g:114,b:200
    },35:{
      r:188,g:63,b:188
    },36:{
      r:17,g:168,b:205
    },37:{
      r:229,g:229,b:229
    },90:{
      r:102,g:102,b:102
    },91:{
      r:241,g:76,b:76
    },92:{
      r:35,g:209,b:139
    },93:{
      r:245,g:245,b:67
    },94:{
      r:59,g:142,b:234
    },95:{
      r:214,g:112,b:214
    },96:{
      r:41,g:184,b:219
    },97:{
      r:255,g:255,b:255
    }
  },TJ6={
    r:229,g:229,b:229
  },lNK={
    r:30,g:30,b:30
  }
} /* confidence: 30% */

/* original: z */ let composed_value=_.split(/\r?\n/).map((J)=>J.trim()).filter((J)=>J.length>0&&!J.startsWith("#")); /* confidence: 30% */

/* original: C1 */ let composed_value=YA(D8).replace(/\r/g,`
`).replaceAll("\t","    "); /* confidence: 30% */

/* original: eb7 */ function helper_fn(q,K="Settings"){
  let _=q.split(/\r?\n/),z=K.replace(/[.*+?^${
    
  } /* confidence: 35% */

/* original: r */ function composed_value(u1){
  for(var W1=new Map;
  u1!==null;
  )u1.key!==null?W1.set(u1.key,u1):W1.set(u1.index,u1),u1=u1.sibling;
   /* confidence: 30% */

/* original: Ex1 */ function mcpbignore_utf8_(q){
  let K=yx1(q,".mcpbignore");
  if(!Rp_(K))return[];
  try{
    return Nx1(K,"utf-8").split(/\r?\n/).map((z)=>z.trim()).filter((z)=>z.length>0&&!z.startsWith("#"))
  }catch(_){
    return console.warn(`Warning: Could not read .mcpbignore file: ${_ instanceof Error?_.message:"Unknown error"}`),[]
  }
} /* confidence: 65% */

/* original: _i_ */ function helper_fn(q,K){
  if(K.stripTrailingCr)q=q.replace(/\r\n/g,`
`);
   /* confidence: 35% */

/* original: Ogz */ function helper_fn(q){
  return q?q.split(/[\t\n\f\r ]+/).filter($gz):[]
} /* confidence: 35% */

/* original: N7K */ function helper_fn(q){
  return q.replace(/\r[\n\u0085]/g,`
`).replace(/[\r\u0085\u2028]/g,`
`)
} /* confidence: 35% */

/* original: Op */ function helper_fn(q,K,_){
  return{
    r:Math.round(q.r+(K.r-q.r)*_),g:Math.round(q.g+(K.g-q.g)*_),b:Math.round(q.b+(K.b-q.b)*_)
  } /* confidence: 35% */

/* original: ve1 */ function helper_fn(q){
  let K=q.split(/(\r\n|\n|\r)/),_="";
   /* confidence: 35% */

/* original: W0 */ function url_handler(q){
  if(!q)return Object.create(url_handler.prototype);
  this.url=q.replace(/^[ \t\n\r\f]+|[ \t\n\r\f]+$/g,"");
  var K=url_handler.pattern.exec(this.url);
  if(K){
    if(K[2])this.scheme=K[2];
    if(K[4]){
      var _=K[4].match(url_handler.userinfoPattern);
      if(_)this.username=_[1],this.password=_[3],K[4]=K[4].substring(_[0].length);
      if(K[4].match(url_handler.portPattern)){
        var z=K[4].lastIndexOf(":");
        this.host=K[4].substring(0,z),this.port=K[4].substring(z+1)
      }else this.host=K[4]
    }if(K[5])this.path=K[5];
    if(K[6])this.query=K[7];
    if(K[8])this.fragment=K[9]
  }
} /* confidence: 70% */

/* original: hMK */ function _g(q){
  return q.replace(/[\r\n]/g,"").replace(/\\/g,"\\\\").replace(/"/g,"\\\"")
} /* confidence: 65% */

/* original: RS6 */ function helper_fn(q){
  return(q.match(/\r\n|\r|\n/g)||[]).length
} /* confidence: 35% */

/* original: nNK */ function x1B__m__x1B(q){
  let K=[],_=q.split(`
`);
  for(let z of _){
    let Y=[],$=TJ6,O=!1,A=0;
    while(A<z.length){
      if(z[A]==="\x1B"&&z[A+1]==="["){
        let H=A+2;
        while(H<z.length&&!/[A-Za-z]/.test(z[H]))H++;
        if(z[H]==="m"){
          let J=z.slice(A+2,H).split(";").map(Number),M=0;
          while(M<J.length){
            let X=J[M];
            if(X===0)$=TJ6,O=!1;
            else if(X===1)O=!0;
            else if(X>=30&&X<=37)$=cNK[X]||TJ6;
            else if(X>=90&&X<=97)$=cNK[X]||TJ6;
            else if(X===39)$=TJ6;
            else if(X===38){
              if(J[M+1]===5&&J[M+2]!==void 0){
                let P=J[M+2];
                $=CMY(P),M+=2
              }else if(J[M+1]===2&&J[M+2]!==void 0&&J[M+3]!==void 0&&J[M+4]!==void 0)$={
                r:J[M+2],g:J[M+3],b:J[M+4]
              },M+=4
            }M++
          }
        }A=H+1;
        continue
      }let w=A;
      while(A<z.length&&z[A]!=="\x1B")A++;
      let j=z.slice(w,A);
      if(j)Y.push({
        text:j,color:$,bold:O
      })
    }if(Y.length===0)Y.push({
      text:"",color:TJ6,bold:!1
    });
    K.push(Y)
  }return K
} /* confidence: 65% */

/* original: CMY */ function helper_fn(q){
  if(q<16)return[{
    r:0,g:0,b:0
  } /* confidence: 35% */

/* original: xIY */ function helper_fn(q){
  return q.replace(/[\r\n\x00]/g,"")
} /* confidence: 35% */

