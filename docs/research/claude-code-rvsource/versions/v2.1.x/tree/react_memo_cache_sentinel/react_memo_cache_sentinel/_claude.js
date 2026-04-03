// Module: _claude

/* original: z */ let composed_value=ru(K); /* confidence: 30% */

/* original: j */ let composed_value=sx(),H=z7(); /* confidence: 30% */

/* original: A */ let headers=_==="Project"?uo6(uo6(K)):z7(),w=$X4(q)?ur_(headers,q):q; /* confidence: 70% */

/* original: _ */ let composed_value=[],z=new Set,Y=kw(),$=q||Y.hasClaudeMdExternalIncludesApproved||!1,O=CO6("Managed"); /* confidence: 30% */

/* original: w */ let composed_value=[],j=z7(),H=j; /* confidence: 30% */

/* original: K */ let composed_value=z7(); /* confidence: 30% */

/* original: H */ let composed_value=aY(q),J=$?`
<${IR}>${$}</${IR}>`:"",M=`<${Pw}>
<${YG}>${q}</${YG}>${J}
<${uR}>${composed_value}</${uR}>
<${qM}>${_}</${qM}>
<${MH}>${UZ(j)}</${MH}>
</${Pw}>`; /* confidence: 30% */

/* original: z */ let composed_value=z7(),Y=_.find((A)=>A.scope==="local"&&A.projectPath===composed_value); /* confidence: 30% */

/* original: A */ let headers=O.filter(cq7); /* confidence: 70% */

/* original: $ */ let composed_value=await ns(z7()),O=await Gd8(composed_value); /* confidence: 30% */

/* original: K */ let composed_value=rj(z7()); /* confidence: 30% */

/* original: $ */ let composed_value=rj(z7()); /* confidence: 30% */

/* original: H */ let composed_value=rj(z7()); /* confidence: 30% */

/* original: K */ let composed_value=rj(z7()),_=Sj()||LL(),z=_?`grep -rn "<search term>" ${q} --include="*.md"`:`${$9} with pattern="<search term>" path="${q}" glob="*.md"`,Y=_?`grep -rn "<search term>" ${composed_value}/ --include="*.jsonl"`:`${$9} with pattern="<search term>" path="${composed_value}/" glob="*.jsonl"`; /* confidence: 30% */

/* original: K */ let composed_value=dk(z7(),".claude","commands"),_=dk(z7(),".claude","agents"),z=dk(z7(),".claude","skills"); /* confidence: 30% */

/* original: q */ let composed_value=M8(),K=I36(); /* confidence: 30% */

/* original: K */ let composed_value=I36(),_=iC6(q); /* confidence: 30% */

/* original: $ */ let composed_value=Uc8(); /* confidence: 30% */

/* original: K */ let composed_value=aY(q); /* confidence: 30% */

/* original: _ */ let composed_value=aY(q); /* confidence: 30% */

/* original: b */ let composed_value=Z8(),I=await u5(composed_value)?composed_value:z7(); /* confidence: 30% */

/* original: K */ let composed_value=Lt(q7(),"projects"),_=Lt(composed_value,XX(z7())),z=Lt(_,`${N8()}-${oh.timestamp}.cast`); /* confidence: 30% */

/* original: z */ let composed_value=Ew7(K,_,{
  projectRoot:z7()
} /* confidence: 30% */

/* original: K */ let composed_value=z7(),z=FY(composed_value)??composed_value,Y; /* confidence: 30% */

/* original: K6 */ let composed_value={
  ...U,...H8$($,Dc()?I36():void 0)
} /* confidence: 30% */

/* original: z7 */ function utility_fn(){
  return G8.originalCwd
} /* confidence: 40% */

/* original: CO6 */ function helper_fn(q){
  let K=z7();
   /* confidence: 35% */

/* original: HX4 */ function helper_fn(q){
  return yN(q,z7())
} /* confidence: 35% */

/* original: KL8 */ function helper_fn(){
  let q=Z8();
  return FY(q)??z7()
} /* confidence: 35% */

/* original: Bm8 */ function helper_fn(q){
  let K=Z8(),_=z7(),z=AM7();
   /* confidence: 35% */

/* original: MMK */ function helper_fn(q){
  let K=rj(z7());
  return(await OMK(K,!0)).filter((z)=>z.mtime>q).map((z)=>z.sessionId)
} /* confidence: 35% */

/* original: cq7 */ function user_managed(q){
  return q.scope==="user"||q.scope==="managed"||q.projectPath===z7()
} /* confidence: 65% */

/* original: IhK */ function project_local(q){
  return q==="project"||q==="local"?z7():void 0
} /* confidence: 65% */

/* original: fuK */ function WriteTool(q){
  let K=aY(q.id);
  try{
    let _=await BB(K,UyY);
    return{
      content:_.content,bytesTotal:_.bytesTotal
    }
  }catch{
    return{
      content:"",bytesTotal:0
    }
  }
} /* confidence: 31% */

/* original: Et6 */ function helper_fn(q){
  let K=rj(z7()),_=I0(K,`${q}.jsonl`),z=M8();
   /* confidence: 35% */

/* original: GQK */ function helper_fn(q){
  let K=rj(z7()),_=await UC6(K,q,z7());
  return await CxY(_),_
} /* confidence: 35% */

/* original: Uc8 */ function helper_fn(){
  return dk(RC(),XX(z7()))+vf
} /* confidence: 35% */

/* original: I36 */ function helper_fn(){
  return dk(Uc8(),N8(),"scratchpad")
} /* confidence: 35% */

/* original: IQK */ function helper_fn(){
  if(!Dc())throw Error("Scratchpad directory feature is not enabled");
  let q=M8(),K=I36();
  return await q.mkdir(K,{
    mode:448
  }),K
} /* confidence: 35% */

/* original: ru */ function helper_fn(q){
  return new Set([z7(),...q.additionalWorkingDirectories.keys()])
} /* confidence: 35% */

/* original: Gh6 */ function helper_fn(){
  if(GY7===void 0)GY7=QQK(Uc8(),N8(),"tasks");
  return GY7
} /* confidence: 35% */

/* original: TY7 */ function helper_fn(){
  await PIY(Gh6(),{
    recursive:!0
  })
} /* confidence: 35% */

/* original: aY */ function helper_fn(q){
  return QQK(Gh6(),`${q}.output`)
} /* confidence: 35% */

/* original: su8 */ function win32_wx(q){
  return cc8((async()=>{
    await TY7();
    let K=aY(q);
    return await(await UQK(K,process.platform==="win32"?"wx":$M6.O_WRONLY|$M6.O_CREAT|$M6.O_EXCL|dQK)).close(),K
  } /* confidence: 65% */

/* original: BH6 */ function helper_fn(q,K){
  return cc8((async()=>{
    try{
      await TY7();
      let _=aY(q);
      try{
        await gQK(K,_)
      }catch{
        await WIY(_),await gQK(K,_)
      }return _
    }catch(_){
      return j6(_),su8(q)
    }
  } /* confidence: 35% */

/* original: Y0 */ function helper_fn(q,K,_,z){
  return{
    id:q,type:K,status:"pending",description:_,toolUseId:z,startTime:Date.now(),outputFile:aY(q),outputOffset:0,notified:!1
  } /* confidence: 35% */

/* original: hiY */ function helper_fn(){
  let q=N8(),K=Lt(q7(),"projects"),_=Lt(K,XX(z7()));
   /* confidence: 35% */

/* original: g85 */ function path_handler(q,K){
  if((q.source==="directory"||q.source==="file")&&!eoY(q.path)){
    let _=K??z7(),z=PH(_);
    return{
      ...q,path:qaY(z??_,q.path)
    }
  } /* confidence: 70% */

