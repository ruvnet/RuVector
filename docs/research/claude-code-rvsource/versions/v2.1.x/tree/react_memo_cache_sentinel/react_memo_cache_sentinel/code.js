// Module: code

/* original: H */ let composed_value={
  jsonrpc:"2.0",id:q.id,error:{
    code:Number.isSafeInteger(j.code)?j.code:z5.InternalError,message:j.message??"Internal error",...j.data!==void 0&&{
      data:j.data
    }
  }
}; /* confidence: 30% */

/* original: eG7 */ var schema_def=B((tG7)=>{
  Object.defineProperty(tG7,"__esModule",{
    value:!0
  });
  var XG5={
    keyword:"id",code(){
      throw Error('NOT SUPPORTED: keyword "id", use "$id" for schema ID')
    }
  };
  tG7.default=XG5
} /* confidence: 95% */

/* original: XG5 */ var schema_def={
  keyword:"id",code(){
    throw Error('NOT SUPPORTED: keyword "id", use "$id" for schema ID')
  }
}; /* confidence: 95% */

/* original: Av7 */ var schema_def=B((Ov7)=>{
  Object.defineProperty(Ov7,"__esModule",{
    value:!0
  });
  var GG5=eG7(),vG5=$v7(),TG5=["$schema","$id","$defs","$vocabulary",{
    keyword:"$comment"
  },"definitions",GG5.default,vG5.default];
  Ov7.default=TG5
} /* confidence: 95% */

/* original: GG5 */ var schema_def=eG7(),vG5=$v7(),TG5=["$schema","$id","$defs","$vocabulary",{
  keyword:"$comment"
} /* confidence: 95% */

/* original: jv7 */ var __esModule_____number_number=B((wv7)=>{
  Object.defineProperty(wv7,"__esModule",{
    value:!0
  });
  var f$8=h_(),ae=f$8.operators,Z$8={
    maximum:{
      okStr:"<=",ok:ae.LTE,fail:ae.GT
    },minimum:{
      okStr:">=",ok:ae.GTE,fail:ae.LT
    },exclusiveMaximum:{
      okStr:"<",ok:ae.LT,fail:ae.GTE
    },exclusiveMinimum:{
      okStr:">",ok:ae.GT,fail:ae.LTE
    }
  },VG5={
    message:({
      keyword:q,schemaCode:K
    })=>f$8.str`must be ${Z$8[q].okStr} ${K}`,params:({
      keyword:q,schemaCode:K
    })=>f$8._`{comparison: ${Z$8[q].okStr}, limit: ${K}}`
  },NG5={
    keyword:Object.keys(Z$8),type:"number",schemaType:"number",$data:!0,error:VG5,code(q){
      let{
        keyword:K,data:_,schemaCode:z
      }=q;
      q.fail$data(f$8._`${_} ${Z$8[K].fail} ${z} || isNaN(${_})`)
    }
  };
  wv7.default=NG5
} /* confidence: 65% */

/* original: Bv7 */ var items_array_object_array_boole=h_(),v$8=pY(),Lv5=hR(),hv5={
  keyword:"items",type:"array",schemaType:["object","array","boolean"],before:"uniqueItems",code(q){
    let{
      schema:K,it:_
    }=q;
    if(Array.isArray(K))return gv7(q,"additionalItems",K);
    if(_.items=!0,(0,v$8.alwaysValidSchema)(_,K))return;
    q.ok((0,Lv5.validateArray)(q))
  }
}; /* confidence: 65% */

/* original: dv7 */ var __esModule_prefixItems_array_a=B((Qv7)=>{
  Object.defineProperty(Qv7,"__esModule",{
    value:!0
  });
  var Sv5=bq1(),Cv5={
    keyword:"prefixItems",type:"array",schemaType:["array"],before:"uniqueItems",code:(q)=>(0,Sv5.validateTuple)(q,"items")
  };
  Qv7.default=Cv5
} /* confidence: 65% */

/* original: Sv5 */ var prefixItems_array_array_unique=bq1(),Cv5={
  keyword:"prefixItems",type:"array",schemaType:["array"],before:"uniqueItems",code:(q)=>(0,prefixItems_array_array_unique.validateTuple)(q,"items")
}; /* confidence: 65% */

/* original: nv7 */ var __esModule_items_array_object_=B((lv7)=>{
  Object.defineProperty(lv7,"__esModule",{
    value:!0
  });
  var cv7=h_(),xv5=pY(),Iv5=hR(),uv5=Cq1(),mv5={
    message:({
      params:{
        len:q
      }
    })=>cv7.str`must NOT have more than ${q} items`,params:({
      params:{
        len:q
      }
    })=>cv7._`{limit: ${q}}`
  },pv5={
    keyword:"items",type:"array",schemaType:["object","boolean"],before:"uniqueItems",error:mv5,code(q){
      let{
        schema:K,parentSchema:_,it:z
      }=q,{
        prefixItems:Y
      }=_;
      if(z.items=!0,(0,xv5.alwaysValidSchema)(z,K))return;
      if(Y)(0,uv5.validateAdditionalItems)(q,Y);
      else q.ok((0,Iv5.validateArray)(q))
    }
  };
  lv7.default=pv5
} /* confidence: 65% */

/* original: OT5 */ var named_entity=pY(),AT5={
  keyword:"not",schemaType:["object","boolean"],trackErrors:!0,code(q){
    let{
      gen:K,schema:_,it:z
    }=q;
    if((0,named_entity.alwaysValidSchema)(z,_)){
      q.fail();
      return
    }let Y=K.name("valid");
    q.subschema({
      keyword:"not",compositeRule:!0,createErrors:!1,allErrors:!1
    },Y),q.failResult(Y,()=>q.reset(),()=>q.error())
  },error:{
    message:"must NOT be valid"
  }
}; /* confidence: 70% */

/* original: fT7 */ var schema_def=B((DT7)=>{
  Object.defineProperty(DT7,"__esModule",{
    value:!0
  });
  var jT5=hR(),HT5={
    keyword:"anyOf",schemaType:"array",trackErrors:!0,code:jT5.validateUnion,error:{
      message:"must match a schema in anyOf"
    }
  };
  DT7.default=HT5
} /* confidence: 95% */

/* original: jT5 */ var schema_def=hR(),HT5={
  keyword:"anyOf",schemaType:"array",trackErrors:!0,code:schema_def.validateUnion,error:{
    message:"must match a schema in anyOf"
  }
}; /* confidence: 95% */

/* original: LT7 */ var __esModule_then_else_object_bo=B((ET7)=>{
  Object.defineProperty(ET7,"__esModule",{
    value:!0
  });
  var kT5=pY(),VT5={
    keyword:["then","else"],schemaType:["object","boolean"],code({
      keyword:q,parentSchema:K,it:_
    }){
      if(K.if===void 0)(0,kT5.checkStrictMode)(_,`"${q}" without "if" is ignored`)
    }
  };
  ET7.default=VT5
} /* confidence: 65% */

/* original: kT5 */ var then_else_object_boolean=pY(),VT5={
  keyword:["then","else"],schemaType:["object","boolean"],code({
    keyword:q,parentSchema:K,it:_
  }){
    if(K.if===void 0)(0,then_else_object_boolean.checkStrictMode)(_,`"${q}" without "if" is ignored`)
  }
}; /* confidence: 65% */

/* original: mR7 */ var win32_ENOENT_ENOENT_exit_error=B((nm$,uR7)=>{
  var M51=process.platform==="win32";
  function X51(q,K){
    return Object.assign(Error(`${K} ${q.command} ENOENT`),{
      code:"ENOENT",errno:"ENOENT",syscall:`${K} ${q.command}`,path:q.command,spawnargs:q.args
    })
  }function uS5(q,K){
    if(!M51)return;
    let _=q.emit;
    q.emit=function(z,Y){
      if(z==="exit"){
        let $=IR7(Y,K);
        if($)return _.call(q,"error",$)
      }return _.apply(q,arguments)
    }
  }function IR7(q,K){
    if(M51&&q===1&&!K.file)return X51(K.original,"spawn");
    return null
  }function mS5(q,K){
    if(M51&&q===1&&!K.file)return X51(K.original,"spawnSync");
    return null
  }uR7.exports={
    hookChildProcess:uS5,verifyENOENT:IR7,verifyENOENTSync:mS5,notFoundError:X51
  }
} /* confidence: 65% */

/* original: K */ let composed_value=process.env.LC_TERMINAL==="iTerm2"?["load-buffer","-"]:["load-buffer","-w","-"],{
  code:_
} /* confidence: 30% */

/* original: e */ let composed_value={
  code:G6,language:H6
}; /* confidence: 30% */

/* original: z */ let composed_value=new lg1("error",{
  code:K,message:q
} /* confidence: 30% */

/* original: z */ let composed_value=new lg1("error",{
  code:K,message:q
} /* confidence: 30% */

/* original: _ */ let composed_value=K==="darwin"?"open":"xdg-open",{
  code:z
} /* confidence: 30% */

/* original: HU1 */ var sourcegraph_cody_aider_tabby_t=L(()=>{
  jU1={
    src:"sourcegraph",cody:"cody",aider:"aider",tabby:"tabby",tabnine:"tabnine",augment:"augment",pieces:"pieces",qodo:"qodo",aide:"aide",hound:"hound",seagoat:"seagoat",bloop:"bloop",gitloop:"gitloop",q:"amazon-q",gemini:"gemini"
  },FMz=[{
    pattern:/^sourcegraph$/i,tool:"sourcegraph"
  },{
    pattern:/^cody$/i,tool:"cody"
  },{
    pattern:/^openctx$/i,tool:"openctx"
  },{
    pattern:/^aider$/i,tool:"aider"
  },{
    pattern:/^continue$/i,tool:"continue"
  },{
    pattern:/^github[-_]?copilot$/i,tool:"github-copilot"
  },{
    pattern:/^copilot$/i,tool:"github-copilot"
  },{
    pattern:/^cursor$/i,tool:"cursor"
  },{
    pattern:/^tabby$/i,tool:"tabby"
  },{
    pattern:/^codeium$/i,tool:"codeium"
  },{
    pattern:/^tabnine$/i,tool:"tabnine"
  },{
    pattern:/^augment[-_]?code$/i,tool:"augment"
  },{
    pattern:/^augment$/i,tool:"augment"
  },{
    pattern:/^windsurf$/i,tool:"windsurf"
  },{
    pattern:/^aide$/i,tool:"aide"
  },{
    pattern:/^codestory$/i,tool:"aide"
  },{
    pattern:/^pieces$/i,tool:"pieces"
  },{
    pattern:/^qodo$/i,tool:"qodo"
  },{
    pattern:/^amazon[-_]?q$/i,tool:"amazon-q"
  },{
    pattern:/^gemini[-_]?code[-_]?assist$/i,tool:"gemini"
  },{
    pattern:/^gemini$/i,tool:"gemini"
  },{
    pattern:/^hound$/i,tool:"hound"
  },{
    pattern:/^seagoat$/i,tool:"seagoat"
  },{
    pattern:/^bloop$/i,tool:"bloop"
  },{
    pattern:/^gitloop$/i,tool:"gitloop"
  },{
    pattern:/^claude[-_]?context$/i,tool:"claude-context"
  },{
    pattern:/^code[-_]?index[-_]?mcp$/i,tool:"code-index-mcp"
  },{
    pattern:/^code[-_]?index$/i,tool:"code-index-mcp"
  },{
    pattern:/^local[-_]?code[-_]?search$/i,tool:"local-code-search"
  },{
    pattern:/^codebase$/i,tool:"autodev-codebase"
  },{
    pattern:/^autodev[-_]?codebase$/i,tool:"autodev-codebase"
  },{
    pattern:/^code[-_]?context$/i,tool:"claude-context"
  }]
} /* confidence: 65% */

/* original: gS8 */ var error_handler=B((gQ4)=>{
  Object.defineProperty(gQ4,"__esModule",{
    value:!0
  });
  gQ4.OTLPExporterError=void 0;
  class BQ4 extends Error{
    code;
    name="OTLPExporterError";
    data;
    constructor(q,K,_){
      super(q);
      this.data=_,this.code=K
    }
  }gQ4.OTLPExporterError=BQ4
} /* confidence: 95% */

/* original: Ivz */ var composed_value=gS8(); /* confidence: 30% */

/* original: pkz */ var zlib_stream=U6("zlib"),Bkz=U6("stream"),El4=kl4(),gkz=gS8(),Fkz=yl4(),Ll4=`OTel-OTLP-Exporter-JavaScript/${Fkz.VERSION}`; /* confidence: 65% */

/* original: Ee6 */ var composed_value=B((ha4)=>{
  Object.defineProperty(ha4,"__esModule",{
    value:!0
  });
  ha4.restrictControlPlaneStatusCode=TSz;
  var lQ=e_(),vSz=[lQ.Status.OK,lQ.Status.INVALID_ARGUMENT,lQ.Status.NOT_FOUND,lQ.Status.ALREADY_EXISTS,lQ.Status.FAILED_PRECONDITION,lQ.Status.ABORTED,lQ.Status.OUT_OF_RANGE,lQ.Status.DATA_LOSS];
  function TSz(q,K){
    if(vSz.includes(q))return{
      code:lQ.Status.INTERNAL,details:`Invalid status from control plane: ${q} ${lQ.Status[q]} ${K}`
    };
    else return{
      code:q,details:K
    }
  }
} /* confidence: 30% */

/* original: eCz */ var composed_value=$C8(),t26=e_(),e26=LE6(),Ts4=nD(),qbz=Cw(),Kbz=Ee6(),_bz="resolving_call"; /* confidence: 30% */

/* original: Hbz */ var composed_value=ME6(),Jbz=Ji4(),Mbz=Ps4(),Zn1=Oa(),Xbz=nD(),MK6=e_(),Pbz=rC8(),Wbz=qn1(),us4=om(),Yb8=Cw(),Dbz=jn1(),$b8=Nk(),Sb=yk(),Ie6=HK6(),fbz=vs4(),Zbz=LE6(),Gbz=ys4(),Dn1=nC8(),vbz=Ee6(),fn1=Cs4(),Tbz=xe6(),kbz=2147483647,Vbz=1000,Nbz=1800000,Ob8=new Map,ybz=16777216,Ebz=1048576; /* confidence: 30% */

/* original: z */ let composed_value={
  code:Tn1.Status.UNKNOWN,details:"message"in q?q.message:"Unknown Error",metadata:(_=K!==null&&K!==void 0?K:q.metadata)!==null&&_!==void 0?_:null
}; /* confidence: 30% */

/* original: gn1 */ var composed_value=Oa(); /* confidence: 30% */

/* original: S3 */ let helper_fn=Q4,N5; /* confidence: 35% */

/* original: l */ let composed_value=wK8(v7().language),i=await Lz7(); /* confidence: 30% */

/* original: z6 */ let skip_immediate_prompt=v7().language,M6=wK8(skip_immediate_prompt); /* confidence: 65% */

const composed_value = btn.previousElementSibling; /* confidence: 30% */

/* original: btn */ const composed_value = document.querySelector('.copy-all-composed_value'); /* confidence: 30% */

/* original: K */ let composed_value=await this.taskOutput.getStdout(),_={
  code:q,stdout:composed_value,stderr:this.taskOutput.getStderr(),interrupted:q===kY7,backgroundTaskId:this.#composed_value
}; /* confidence: 30% */

/* original: Q4 */ let composed_value=Wq.map((N5)=>({
  ...N5,session_id:a
} /* confidence: 30% */

/* original: z */ function composed_value(Y){
  for(let O of Y)if(O.result.status==="valid")return O.result;
  for(let O of Y)if(O.result.status==="dirty")return K.common.issues.push(...O.ctx.common.issues),O.result;
  let $=Y.map((O)=>new MV(O.ctx.common.issues));
  return D4(K,{
    code:_q.invalid_union,unionErrors:$
  }),r5
} /* confidence: 30% */

/* original: WD7 */ function value_holder(q,K){
  if(!q.issues.length&&q.value===void 0)q.issues.push({
    code:"invalid_type",expected:"nonoptional",input:q.value,inst:K
  } /* confidence: 70% */

/* original: C07 */ function helper_fn(q){
  return q.scopeValue("func",{
    ref:Object.prototype.hasOwnProperty,code:k2._`Object.prototype.hasOwnProperty`
  })
} /* confidence: 35% */

/* original: X */ function composed_value(P){
  let W=K.scopeValue("schema",A.code.source===!0?{
    ref:P,code:(0,fV.stringify)(P)
  } /* confidence: 30% */

/* original: X51 */ function ENOENT_ENOENT(q,K){
  return Object.assign(Error(`${K} ${q.command} ENOENT`),{
    code:"ENOENT",errno:"ENOENT",syscall:`${K} ${q.command}`,path:q.command,spawnargs:q.args
  })
} /* confidence: 65% */

/* original: IR7 */ function helper_fn(q,K){
  if(M51&&q===1&&!K.file)return X51(K.original,"spawn");
   /* confidence: 35% */

/* original: mS5 */ function helper_fn(q,K){
  if(M51&&q===1&&!K.file)return X51(K.original,"spawnSync");
   /* confidence: 35% */

/* original: z31 */ function git_checkignore(q,K){
  let{
    code:_
  }=await x7("git",["check-ignore",q],{
    preserveOutputOnError:!1,cwd:K
  });
  return _===0
} /* confidence: 65% */

/* original: F31 */ function helper_fn(q,K){
  return new Promise((_)=>{
    fI5(q,K,{
      encoding:"utf-8",timeout:rb7
    },(z,Y)=>{
      _({
        stdout:Y??"",code:z?1:0
      })
    })
  })
} /* confidence: 35% */

/* original: fw8 */ function darwin_win32_query_v_query_v(){
  return(async()=>{
    if(process.platform==="darwin"){
      let q=ob7(),_=(await Promise.all(q.map(async({
        path:z,label:Y
      })=>{
        if(!ZI5(z))return{
          stdout:"",label:Y,ok:!1
        };
        let{
          stdout:$,code:O
        }=await F31(nb7,[...ib7,z]);
        return{
          stdout:$,label:Y,ok:O===0&&!!$
        }
      }))).find((z)=>z.ok);
      return{
        plistStdouts:_?[{
          stdout:_.stdout,label:_.label
        }]:[],hklmStdout:null,hkcuStdout:null
      }
    }if(process.platform==="win32"){
      let K=`${process.env.SYSTEMROOT||"C:\\Windows"}\\System32\\reg.exe`,[_,z]=await Promise.all([F31(K,["query",Ww8,"/v",cD6]),F31(K,["query",Dw8,"/v",cD6])]);
      return{
        plistStdouts:null,hklmStdout:_.code===0?_.stdout:null,hkcuStdout:z.code===0?z.stdout:null
      }
    }return{
      plistStdouts:null,hklmStdout:null,hkcuStdout:null
    }
  } /* confidence: 65% */

/* original: gI9 */ function error_handler(q){
  return new Promise((K,_)=>{
    let z=[];
    q.on("data",(Y)=>{
      if(Buffer.isBuffer(Y))z.push(Y);
      else z.push(Buffer.from(Y))
    }),q.on("end",()=>{
      K(Buffer.concat(z).toString("utf8"))
    }),q.on("error",(Y)=>{
      if(Y&&(Y===null||Y===void 0?void 0:Y.name)==="AbortError")_(Y);
      else _(new YN(`Error reading response as text: ${Y.message}`,{
        code:YN.PARSE_ERROR
      }))
    })
  } /* confidence: 95% */

/* original: GR4 */ function typed_entity(q){
  return{
    type:q.type,message:q.message,code:q.code,defaultPrevented:q.defaultPrevented,cancelable:q.cancelable,timeStamp:q.timeStamp
  } /* confidence: 70% */

/* original: bR4 */ function auth_handler(q,K,_){
  return new URLSearchParams({
    grant_type:"authorization_code",code:q,code_verifier:K,redirect_uri:String(_)
  } /* confidence: 95% */

/* original: Rvz */ function BashTool(q,K){
  return new eQ4(q.transport,q.serializer,(0,Lvz.createLoggingPartialSuccessResponseHandler)(),q.promiseHandler,K.timeout)
} /* confidence: 31% */

/* original: TSz */ function helper_fn(q,K){
  if(vSz.includes(q))return{
    code:lQ.Status.INTERNAL,details:`Invalid status from control plane: ${q} ${lQ.Status[q]} ${K}`
  };
  else return{
    code:q,details:K
  }
} /* confidence: 35% */

/* original: kn1 */ function error_handler(q,K){
  var _;
  let z={
    code:Tn1.Status.UNKNOWN,details:"message"in q?q.message:"Unknown Error",metadata:(_=K!==null&&K!==void 0?K:q.metadata)!==null&&_!==void 0?_:null
  };
  if("code"in q&&typeof q.code==="number"&&Number.isInteger(q.code)){
    if(z.code=q.code,"details"in q&&typeof q.details==="string")z.details=q.details
  }return z
} /* confidence: 95% */

/* original: un1 */ function helper_fn(q){
  return{
    code:bM.Status.UNIMPLEMENTED,details:`The server does not implement the method ${q}`
  } /* confidence: 35% */

/* original: Cxz */ function helper_fn(q,K){
  let _=un1(K);
   /* confidence: 35% */

/* original: PzK */ function McpErrorHandler(q){
  return g6({
    message:q.message,severity:q.severity,range:q.range,source:q.source||null,code:q.code||null
  } /* confidence: 45% */

/* original: pez */ function nooptionallocks_mergebase_HEAD(q){
  let K=process.env.CLAUDE_CODE_BASE_REF||await jT(),{
    stdout:_,code:z
  }=await x7(h7(),["--no-optional-locks","merge-base","HEAD",K],{
    cwd:q,timeout:Ke1
  });
  if(z===0&&_.trim())return _.trim();
  return"HEAD"
} /* confidence: 65% */

/* original: kwK */ function helper_fn(){
  return h18()===null?He1:null
} /* confidence: 35% */

/* original: X6Y */ function _add__remove_nochange(q){
  return q.map((K)=>{
    if(K.startsWith("+"))return{
      code:K.slice(1),i:0,type:"add",originalCode:K.slice(1)
    };
    if(K.startsWith("-"))return{
      code:K.slice(1),i:0,type:"remove",originalCode:K.slice(1)
    };
    return{
      code:K.slice(1),i:0,type:"nochange",originalCode:K.slice(1)
    }
  } /* confidence: 65% */

/* original: O37 */ function path_handler(q){
  return q.issues.map((K)=>({
    path:K.path.join(".")||"root",message:K.message,code:K.code
  } /* confidence: 70% */

/* original: wK8 */ function StringTrimmer(q){
  if(!q)return{
    code:hz7
  };
  let K=q.toLowerCase().trim();
  if(!K)return{
    code:hz7
  };
  if(eFK.has(K))return{
    code:K
  };
  let _=bCY[K];
  if(_)return{
    code:_
  };
  let z=K.split("-")[0];
  if(z&&eFK.has(z))return{
    code:z
  };
  return{
    code:hz7,fellBackFrom:q
  }
} /* confidence: 41% */

function Copied_Copy(btn) {
  
      const code = btn.previousElementSibling;
  
      navigator.clipboard.writeText(code.textContent).then(() => {
    
        btn.textContent = 'Copied!';
    
        setTimeout(() => {
       btn.textContent = 'Copy';
       
    }, 2000);
    
      
  });
  
    
} /* confidence: 65% */

/* original: W17 */ function status_porcelain_revlist_count(q,K){
  let{
    code:_,stdout:z
  }=await x7(h7(),["status","--porcelain"],{
    cwd:q
  });
  if(_!==0)return!0;
  if(z.trim().length>0)return!0;
  let{
    code:Y,stdout:$
  }=await x7(h7(),["rev-list","--count",`${K}..HEAD`],{
    cwd:q
  });
  if(Y!==0)return!0;
  if(parseInt($.trim(),10)>0)return!0;
  return!1
} /* confidence: 65% */

/* original: aD6 */ class value_holder{
  value=null;
  left=null;
  middle=null;
  right=null;
  code;
  constructor(q,K,_){
    if(_===void 0||_>=q.length)throw TypeError("Unreachable");
    if((this.code=q.charCodeAt(_))>127)throw TypeError("key must be ascii string");
    if(q.length!==++_)this.middle=new value_holder(q,K,_);
    else this.value=K
  }add(q,K){
    let _=q.length;
    if(_===0)throw TypeError("Unreachable");
    let z=0,Y=this;
    while(!0){
      let $=q.charCodeAt(z);
      if($>127)throw TypeError("key must be ascii string");
      if(Y.code===$)if(_===++z){
        Y.value=K;
        break
      }else if(Y.middle!==null)Y=Y.middle;
      else{
        Y.middle=new value_holder(q,K,z);
        break
      }else if(Y.code<$)if(Y.left!==null)Y=Y.left;
      else{
        Y.left=new value_holder(q,K,z);
        break
      }else if(Y.right!==null)Y=Y.right;
      else{
        Y.right=new value_holder(q,K,z);
        break
      }
    }
  }search(q){
    let K=q.length,_=0,z=this;
    while(z!==null&&_<K){
      let Y=q[_];
      if(Y<=90&&Y>=65)Y|=32;
      while(z!==null){
        if(Y===z.code){
          if(K===++_)return z;
          z=z.middle;
          break
        }z=z.code<Y?z.left:z.right
      }
    }return null
  }
} /* confidence: 70% */

/* original: noq */ class request_body{
  export(q,K){
    this._sendLogRecords(q,K)
  }shutdown(){
    return Promise.resolve()
  }_exportInfo(q){
    return{
      resource:{
        attributes:q.resource.attributes
      },instrumentationScope:q.instrumentationScope,timestamp:(0,loq.hrTimeToMicroseconds)(q.hrTime),traceId:q.spanContext?.traceId,spanId:q.spanContext?.spanId,traceFlags:q.spanContext?.traceFlags,severityText:q.severityText,severityNumber:q.severityNumber,body:q.body,attributes:q.attributes
    }
  }_sendLogRecords(q,K){
    for(let _ of q)console.dir(this._exportInfo(_),{
      depth:3
    });
    K?.({
      code:loq.ExportResultCode.SUCCESS
    })
  }
} /* confidence: 70% */

/* original: Kaq */ class entity_class{
  _finishedLogRecords=[];
  _stopped=!1;
  export(q,K){
    if(this._stopped)return K({
      code:qaq.ExportResultCode.FAILED,error:Error("Exporter has been stopped")
    });
    this._finishedLogRecords.push(...q),K({
      code:qaq.ExportResultCode.SUCCESS
    })
  }shutdown(){
    return this._stopped=!0,this.reset(),Promise.resolve()
  }getFinishedLogRecords(){
    return this._finishedLogRecords
  }reset(){
    this._finishedLogRecords=[]
  }
} /* confidence: 45% */

/* original: LB4 */ class entity_class{
  _shutdown=!1;
  _aggregationTemporality;
  _metrics=[];
  constructor(q){
    this._aggregationTemporality=q
  }export(q,K){
    if(this._shutdown){
      setTimeout(()=>K({
        code:EB4.ExportResultCode.FAILED
      }),0);
      return
    }this._metrics.push(q),setTimeout(()=>K({
      code:EB4.ExportResultCode.SUCCESS
    }),0)
  }getMetrics(){
    return this._metrics
  }forceFlush(){
    return Promise.resolve()
  }reset(){
    this._metrics=[]
  }selectAggregationTemporality(q){
    return this._aggregationTemporality
  }shutdown(){
    return this._shutdown=!0,Promise.resolve()
  }
} /* confidence: 45% */

/* original: Ad1 */ class entity_class{
  _shutdown=!1;
  _temporalitySelector;
  constructor(q){
    this._temporalitySelector=q?.temporalitySelector??gZz.DEFAULT_AGGREGATION_TEMPORALITY_SELECTOR
  }export(q,K){
    if(this._shutdown){
      setImmediate(K,{
        code:CB4.ExportResultCode.FAILED
      });
      return
    }return entity_class._sendMetrics(q,K)
  }forceFlush(){
    return Promise.resolve()
  }selectAggregationTemporality(q){
    return this._temporalitySelector(q)
  }shutdown(){
    return this._shutdown=!0,Promise.resolve()
  }static _sendMetrics(q,K){
    for(let _ of q.scopeMetrics)for(let z of _.metrics)console.dir({
      descriptor:z.descriptor,dataPointType:z.dataPointType,dataPoints:z.dataPoints
    },{
      depth:null
    });
    K({
      code:CB4.ExportResultCode.SUCCESS
    })
  }
} /* confidence: 45% */

/* original: gF4 */ class named_entity{
  _spanContext;
  kind;
  parentSpanContext;
  attributes={
    
  };
  links=[];
  events=[];
  startTime;
  resource;
  instrumentationScope;
  _droppedAttributesCount=0;
  _droppedEventsCount=0;
  _droppedLinksCount=0;
  name;
  status={
    code:yb.SpanStatusCode.UNSET
  };
  endTime=[0,0];
  _ended=!1;
  _duration=[-1,-1];
  _spanProcessor;
  _spanLimits;
  _attributeValueLengthLimit;
  _performanceStartTime;
  _performanceOffset;
  _startTimeProvided;
  constructor(q){
    let K=Date.now();
    if(this._spanContext=q.spanContext,this._performanceStartTime=jv.otperformance.now(),this._performanceOffset=K-(this._performanceStartTime+(0,jv.getTimeOrigin)()),this._startTimeProvided=q.startTime!=null,this._spanLimits=q.spanLimits,this._attributeValueLengthLimit=this._spanLimits.attributeValueLengthLimit||0,this._spanProcessor=q.spanProcessor,this.name=q.name,this.parentSpanContext=q.parentSpanContext,this.kind=q.kind,this.links=q.links||[],this.startTime=this._getTime(q.startTime??K),this.resource=q.resource,this.instrumentationScope=q.scope,q.attributes!=null)this.setAttributes(q.attributes);
    this._spanProcessor.onStart(this,q.context)
  }spanContext(){
    return this._spanContext
  }setAttribute(q,K){
    if(K==null||this._isSpanEnded())return this;
    if(q.length===0)return yb.diag.warn(`Invalid attribute key: ${q}`),this;
    if(!(0,jv.isAttributeValue)(K))return yb.diag.warn(`Invalid attribute value set for key: ${q}`),this;
    let{
      attributeCountLimit:_
    }=this._spanLimits;
    if(_!==void 0&&Object.keys(this.attributes).length>=_&&!Object.prototype.hasOwnProperty.call(this.attributes,q))return this._droppedAttributesCount++,this;
    return this.attributes[q]=this._truncateToSize(K),this
  }setAttributes(q){
    for(let[K,_]of Object.entries(q))this.setAttribute(K,_);
    return this
  }addEvent(q,K,_){
    if(this._isSpanEnded())return this;
    let{
      eventCountLimit:z
    }=this._spanLimits;
    if(z===0)return yb.diag.warn("No events allowed."),this._droppedEventsCount++,this;
    if(z!==void 0&&this.events.length>=z){
      if(this._droppedEventsCount===0)yb.diag.debug("Dropping extra events.");
      this.events.shift(),this._droppedEventsCount++
    }if((0,jv.isTimeInput)(K)){
      if(!(0,jv.isTimeInput)(_))_=K;
      K=void 0
    }let Y=(0,jv.sanitizeAttributes)(K);
    return this.events.push({
      name:q,attributes:Y,time:this._getTime(_),droppedAttributesCount:0
    }),this
  }addLink(q){
    return this.links.push(q),this
  }addLinks(q){
    return this.links.push(...q),this
  }setStatus(q){
    if(this._isSpanEnded())return this;
    if(this.status={
      ...q
    },this.status.message!=null&&typeof q.message!=="string")yb.diag.warn(`Dropping invalid status.message of type '${typeof q.message}', expected 'string'`),delete this.status.message;
    return this
  }updateName(q){
    if(this._isSpanEnded())return this;
    return this.name=q,this
  }end(q){
    if(this._isSpanEnded()){
      yb.diag.error(`${this.name} ${this._spanContext.traceId}-${this._spanContext.spanId} - You can only call end() on a span once.`);
      return
    }if(this._ended=!0,this.endTime=this._getTime(q),this._duration=(0,jv.hrTimeDuration)(this.startTime,this.endTime),this._duration[0]<0)yb.diag.warn("Inconsistent start and end time, startTime > endTime. Setting span duration to 0ms.",this.startTime,this.endTime),this.endTime=this.startTime.slice(),this._duration=[0,0];
    if(this._droppedEventsCount>0)yb.diag.warn(`Dropped ${this._droppedEventsCount} events because eventCountLimit reached`);
    this._spanProcessor.onEnd(this)
  }_getTime(q){
    if(typeof q==="number"&&q<=jv.otperformance.now())return(0,jv.hrTime)(q+this._performanceOffset);
    if(typeof q==="number")return(0,jv.millisToHrTime)(q);
    if(q instanceof Date)return(0,jv.millisToHrTime)(q.getTime());
    if((0,jv.isTimeInputHrTime)(q))return q;
    if(this._startTimeProvided)return(0,jv.millisToHrTime)(Date.now());
    let K=jv.otperformance.now()-this._performanceStartTime;
    return(0,jv.addHrTimes)(this.startTime,(0,jv.millisToHrTime)(K))
  }isRecording(){
    return this._ended===!1
  }recordException(q,K){
    let _={
      
    };
    if(typeof q==="string")_[p26.ATTR_EXCEPTION_MESSAGE]=q;
    else if(q){
      if(q.code)_[p26.ATTR_EXCEPTION_TYPE]=q.code.toString();
      else if(q.name)_[p26.ATTR_EXCEPTION_TYPE]=q.name;
      if(q.message)_[p26.ATTR_EXCEPTION_MESSAGE]=q.message;
      if(q.stack)_[p26.ATTR_EXCEPTION_STACKTRACE]=q.stack
    }if(_[p26.ATTR_EXCEPTION_TYPE]||_[p26.ATTR_EXCEPTION_MESSAGE])this.addEvent(ZGz.ExceptionEventName,_,K);
    else yb.diag.warn(`Failed to record an exception ${q}`)
  }get duration(){
    return this._duration
  }get ended(){
    return this._ended
  }get droppedAttributesCount(){
    return this._droppedAttributesCount
  }get droppedEventsCount(){
    return this._droppedEventsCount
  }get droppedLinksCount(){
    return this._droppedLinksCount
  }_isSpanEnded(){
    if(this._ended){
      let q=Error(`Operation attempted on ended Span {traceId: ${this._spanContext.traceId}, spanId: ${this._spanContext.spanId}}`);
      yb.diag.warn(`Cannot execute the operation on ended Span {traceId: ${this._spanContext.traceId}, spanId: ${this._spanContext.spanId}}`,q)
    }return this._ended
  }_truncateToLimitUtil(q,K){
    if(q.length<=K)return q;
    return q.substring(0,K)
  }_truncateToSize(q){
    let K=this._attributeValueLengthLimit;
    if(K<=0)return yb.diag.warn(`Attribute value limit must be positive, got ${K}`),q;
    if(typeof q==="string")return this._truncateToLimitUtil(q,K);
    if(Array.isArray(q))return q.map((_)=>typeof _==="string"?this._truncateToLimitUtil(_,K):_);
    return q
  }
} /* confidence: 70% */

/* original: cU4 */ class named_entity{
  export(q,K){
    return this._sendSpans(q,K)
  }shutdown(){
    return this._sendSpans([]),this.forceFlush()
  }forceFlush(){
    return Promise.resolve()
  }_exportInfo(q){
    return{
      resource:{
        attributes:q.resource.attributes
      },instrumentationScope:q.instrumentationScope,traceId:q.spanContext().traceId,parentSpanContext:q.parentSpanContext,traceState:q.spanContext().traceState?.serialize(),name:q.name,id:q.spanContext().spanId,kind:q.kind,timestamp:(0,yd1.hrTimeToMicroseconds)(q.startTime),duration:(0,yd1.hrTimeToMicroseconds)(q.duration),attributes:q.attributes,status:q.status,events:q.events,links:q.links
    }
  }_sendSpans(q,K){
    for(let _ of q)console.dir(this._exportInfo(_),{
      depth:3
    });
    if(K)return K({
      code:yd1.ExportResultCode.SUCCESS
    })
  }
} /* confidence: 70% */

/* original: oU4 */ class entity_class{
  _finishedSpans=[];
  _stopped=!1;
  export(q,K){
    if(this._stopped)return K({
      code:rU4.ExportResultCode.FAILED,error:Error("Exporter has been stopped")
    });
    this._finishedSpans.push(...q),setTimeout(()=>K({
      code:rU4.ExportResultCode.SUCCESS
    }),0)
  }shutdown(){
    return this._stopped=!0,this._finishedSpans=[],this.forceFlush()
  }forceFlush(){
    return Promise.resolve()
  }reset(){
    this._finishedSpans=[]
  }getFinishedSpans(){
    return this._finishedSpans
  }
} /* confidence: 45% */

/* original: eQ4 */ class status{
  _transport;
  _serializer;
  _responseHandler;
  _promiseQueue;
  _timeout;
  _diagLogger;
  constructor(q,K,_,z,Y){
    this._transport=q,this._serializer=K,this._responseHandler=_,this._promiseQueue=z,this._timeout=Y,this._diagLogger=hvz.diag.createComponentLogger({
      namespace:"OTLPExportDelegate"
    })
  }export(q,K){
    if(this._diagLogger.debug("items to be sent",q),this._promiseQueue.hasReachedLimit()){
      K({
        code:g26.ExportResultCode.FAILED,error:Error("Concurrent export limit reached")
      });
      return
    }let _=this._serializer.serializeRequest(q);
    if(_==null){
      K({
        code:g26.ExportResultCode.FAILED,error:Error("Nothing to send")
      });
      return
    }this._promiseQueue.pushPromise(this._transport.send(_,this._timeout).then((z)=>{
      if(z.status==="success"){
        if(z.data!=null)try{
          this._responseHandler.handleResponse(this._serializer.deserializeResponse(z.data))
        }catch(Y){
          this._diagLogger.warn("Export succeeded but could not deserialize response - is the response specification compliant?",Y,z.data)
        }K({
          code:g26.ExportResultCode.SUCCESS
        });
        return
      }else if(z.status==="failure"&&z.error){
        K({
          code:g26.ExportResultCode.FAILED,error:z.error
        });
        return
      }else if(z.status==="retryable")K({
        code:g26.ExportResultCode.FAILED,error:new tQ4.OTLPExporterError("Export failed with retryable status")
      });
      else K({
        code:g26.ExportResultCode.FAILED,error:new tQ4.OTLPExporterError("Export failed with unknown error")
      })
    },(z)=>K({
      code:g26.ExportResultCode.FAILED,error:z
    })))
  }forceFlush(){
    return this._promiseQueue.awaitAll()
  }async shutdown(){
    this._diagLogger.debug("shutdown started"),await this.forceFlush(),this._transport.shutdown()
  }
} /* confidence: 70% */

/* original: sn4 */ class status{
  constructor(q){
    this.status=Object.assign({
      code:hyz.Status.UNAVAILABLE,details:"No connection established",metadata:new Lyz.Metadata
    },q)
  }pick(q){
    return{
      pickResultType:JC8.TRANSIENT_FAILURE,subchannel:null,status:this.status,onCallStarted:null,onCallEnded:null
    }
  }
} /* confidence: 70% */

/* original: ps4 */ class entity_class{
  pick(q){
    return{
      pickResultType:Zn1.PickResultType.DROP,status:{
        code:MK6.Status.UNAVAILABLE,details:"Channel closed before call started",metadata:new Xbz.Metadata
      },subchannel:null,onCallStarted:null,onCallEnded:null
    }
  } /* confidence: 45% */

/* original: He1 */ class entity_class{
  code;
   /* confidence: 45% */

