// Module: organizationUuid

/* original: z */ let composed_value=w8(),Y=composed_value.oauthAccount?.organizationRole,$=composed_value.oauthAccount?.workspaceRole; /* confidence: 30% */

/* original: _ */ let composed_value=w8().oauthAccount?.organizationRole; /* confidence: 30% */

/* original: Y */ let composed_value=w8(); /* confidence: 30% */

/* original: q */ let composed_value=!1,K=w8().penguinModeOrgEnabled===!0; /* confidence: 30% */

/* original: O */ let composed_value=eV.status!=="pending"?eV.status==="enabled":w8().penguinModeOrgEnabled; /* confidence: 30% */

/* original: $ */ let composed_value=o_()?.emailAddress; /* confidence: 30% */

/* original: q */ let composed_value=w8(); /* confidence: 30% */

/* original: q */ let composed_value=o_(); /* confidence: 30% */

/* original: zO6 */ var config=L(()=>{
  c4();
  T8();
  T7();
  k1();
  F7();
  R_();
  d8();
  AT();
  Av6=$1((q)=>{
    let K=qC(),_=w8(),z,Y,$;
    if(q){
      if(z=jK()??void 0,Y=xF()??void 0,z&&_.claudeCodeFirstTokenDate){
        let j=new Date(_.claudeCodeFirstTokenDate).getTime();
        if(!isNaN(j))$=j
      }
    }let O=o_(),A=O?.organizationUuid,w=O?.accountUuid;
    return{
      deviceId:K,sessionId:N8(),email:b7_(),appVersion:{
        ISSUES_EXPLAINER:"report the issue at https://github.com/anthropics/claude-code/issues",PACKAGE_URL:"@anthropic-ai/claude-code",README_URL:"https://code.claude.com/docs/en/overview",VERSION:"2.1.91",FEEDBACK_CHANNEL:"https://github.com/anthropics/claude-code/issues",BUILD_TIME:"2026-04-02T21:58:41Z"
      }.VERSION,platform:CD6(),organizationUuid:A,accountUuid:w,userType:"external",subscriptionType:z,rateLimitTier:Y,firstTokenTime:$,...c6(process.env.GITHUB_ACTIONS)&&{
        githubActionsMetadata:{
          actor:process.env.GITHUB_ACTOR,actorId:process.env.GITHUB_ACTOR_ID,repository:process.env.GITHUB_REPOSITORY,repositoryId:process.env.GITHUB_REPOSITORY_ID,repositoryOwner:process.env.GITHUB_REPOSITORY_OWNER,repositoryOwnerId:process.env.GITHUB_REPOSITORY_OWNER_ID
        }
      }
    }
  });
  kZ8=$1(async()=>{
    let q=await Xj("git config --get user.email",{
      reject:!1,cwd:Z8()
    });
    return q.exitCode===0&&q.stdout?q.stdout.trim():void 0
  })
} /* confidence: 95% */

/* original: O */ let composed_value=o_(),A=composed_value?.organizationUuid,w=composed_value?.accountUuid; /* confidence: 30% */

/* original: K */ let composed_value=qC(),{
  accountUuid:_,organizationUuid:z
} /* confidence: 30% */

/* original: z */ let composed_value=w8(),Y=composed_value.cachedGrowthBookFeatures?.[q]; /* confidence: 30% */

/* original: z */ let composed_value=w8(),Y=composed_value.cachedStatsigGates?.[q]; /* confidence: 30% */

/* original: q */ let composed_value=w8(),K=L08(); /* confidence: 30% */

/* original: K */ let composed_value=w8(); /* confidence: 30% */

/* original: z */ let composed_value=o_(); /* confidence: 30% */

/* original: Y */ let composed_value=w8().autoInstallIdeExtension??!0; /* confidence: 30% */

/* original: q */ let composed_value=o_()?.accountUuid; /* confidence: 30% */

/* original: _ */ let composed_value=w8().groveConfigCache?.[q],z=Date.now(); /* confidence: 30% */

/* original: _ */ let composed_value=K.data.grove_enabled,z=w8().groveConfigCache?.[q]; /* confidence: 30% */

/* original: K */ let composed_value=w8().metricsStatusCache; /* confidence: 30% */

/* original: q */ let composed_value=w8().metricsStatusCache; /* confidence: 30% */

/* original: _ */ let composed_value=w8(); /* confidence: 30% */

/* original: A */ let headers=w8().installMethod||"not set",w=null; /* confidence: 70% */

/* original: _ */ let composed_value=w8(); /* confidence: 30% */

/* original: K */ let composed_value=w8().overageCreditGrantCache?.[q]; /* confidence: 30% */

/* original: K */ let composed_value=w8().overageCreditGrantCache; /* confidence: 30% */

/* original: q */ let composed_value=o_()?.organizationUuid; /* confidence: 30% */

/* original: K */ let composed_value=w8().claudeCodeHints; /* confidence: 30% */

/* original: w */ let composed_value=FF(); /* confidence: 30% */

/* original: V6 */ let composed_value=FF(); /* confidence: 30% */

/* original: q */ let composed_value=w8(); /* confidence: 30% */

/* original: b */ let composed_value=new Set(R.clients.map((m)=>m.name)),I=Object.entries(V).filter(([m])=>!composed_value.has(m)).map(([m,p])=>({
  name:m,type:Kv(m)?"disabled":"pending",config:p
} /* confidence: 30% */

/* original: C */ let Backgroundtasksdialogdismissed=Object.fromEntries(Object.entries(b).filter(([g])=>!Kv(g))); /* confidence: 65% */

/* original: q */ let composed_value=o_()?.organizationUuid; /* confidence: 30% */

/* original: _ */ let composed_value=w8().passesEligibilityCache?.[q]; /* confidence: 30% */

/* original: q */ let composed_value=o_()?.organizationUuid; /* confidence: 30% */

/* original: q */ let composed_value=o_()?.organizationUuid; /* confidence: 30% */

/* original: _ */ let composed_value=w8().passesEligibilityCache?.[q],z=Date.now(); /* confidence: 30% */

/* original: _ */ let composed_value=w8().passesLastSeenRemaining??0; /* confidence: 30% */

/* original: _ */ let composed_value=w8(); /* confidence: 30% */

/* original: $ */ let composed_value=(w8().opus1mMergeNoticeSeenCount??0)+1; /* confidence: 30% */

/* original: $ */ let composed_value=(w8().voiceNoticeSeenCount??0)+1; /* confidence: 30% */

/* original: _ */ let composed_value=QF(w8().theme); /* confidence: 30% */

/* original: K */ let composed_value=w8(); /* confidence: 30% */

/* original: O */ let composed_value=w8().oauthAccount?.organizationUuid; /* confidence: 30% */

/* original: uUK */ var config=L(()=>{
  Ji6();
  uz7();
  yUK();
  mz7();
  k1();
  AO();
  Bz7();
  bUK();
  Fz7=w6(D6(),1);
  TbY={
    type:"local-jsx",name:"buddy",description:"Hatch a coding companion Â· pet, off",argumentHint:"[pet|off]",get isHidden(){
      return!Zc8()
    },immediate:!0,load:()=>Promise.resolve({
      async call(q,K,_){
        let z=w8(),Y=_?.trim();
        if(Y==="off"){
          if(z.companionMuted!==!0)S8((A)=>({
            ...A,companionMuted:!0
          }));
          return q("companion muted",{
            display:"system"
          }),null
        }if(Y==="on"){
          if(z.companionMuted===!0)S8((A)=>({
            ...A,companionMuted:!1
          }));
          return q("companion unmuted",{
            display:"system"
          }),null
        }if(!Zc8())return q("buddy is unavailable on this configuration",{
          display:"system"
        }),null;
        if(Y==="pet"){
          let A=TC();
          if(!A)return q("no companion yet Â· run /buddy first",{
            display:"system"
          }),null;
          if(z.companionMuted===!0)S8((w)=>({
            ...w,companionMuted:!1
          }));
          return K.setAppState((w)=>({
            ...w,companionPetAt:Date.now()
          })),vUK(xUK(K.setAppState)),q(`petted ${A.name}`,{
            display:"system"
          }),null
        }if(z.companionMuted===!0)S8((A)=>({
          ...A,companionMuted:!1
        }));
        let $=TC();
        if($)return Fz7.default.createElement(vc8,{
          companion:$,lastReaction:fUK(),onDone:q
        });
        let O=vbY(YS1($S1()));
        return O.then((A)=>GUK(A,xUK(K.setAppState))).catch(()=>{
          
        }),Fz7.default.createElement(CUK,{
          hatching:O,onDone:q
        })
      }
    })
  },kbY=TbY
} /* confidence: 95% */

/* original: $ */ let composed_value=TC(); /* confidence: 30% */

/* original: O */ let composed_value=vbY(YS1($S1())); /* confidence: 30% */

/* original: J */ let composed_value=TC(); /* confidence: 30% */

/* original: Z */ let response=w8(); /* confidence: 70% */

/* original: l6 */ let composed_value=(w8().voiceFooterHintSeenCount??0)+1; /* confidence: 30% */

/* original: t36 */ let typed_entity=gs()?Pq.createElement(kA7,{
  ...YX6,initialMode:E,onModeChange:R
} /* confidence: 70% */

/* original: _ */ let composed_value=w8(); /* confidence: 30% */

/* original: F */ let agent_repl_main_thread=w8().feedbackSurveyState; /* confidence: 65% */

/* original: q */ let composed_value=w8(); /* confidence: 30% */

/* original: q */ let composed_value=w8(); /* confidence: 30% */

/* original: K */ let composed_value=q.map((_)=>({
  tip:_,sessions:Ai8(_.id)
} /* confidence: 30% */

/* original: A */ let headers=z.filter(boY),w=z.filter(CoY),j=z.filter(SoY),H=z.filter(RoY); /* confidence: 70% */

/* original: _ */ let composed_value=await FoY(),Y=w8().lspRecommendationNeverPlugins??[],$=[]; /* confidence: 30% */

/* original: K */ let composed_value=q.client_data??null,_=q.additional_model_options??[],z=w8(); /* confidence: 30% */

/* original: O */ let composed_value=w8(),A=!1; /* confidence: 30% */

/* original: M */ let composed_value=kw(); /* confidence: 30% */

/* original: Y */ let composed_value=kw(),$=w8(),{
  servers:O
} /* confidence: 30% */

/* original: _ */ let composed_value=w8(); /* confidence: 30% */

/* original: o_ */ function helper_fn(){
  return yJ()?w8().oauthAccount:void 0
} /* confidence: 35% */

/* original: zv6 */ function helper_fn(){
  let K=o_()?.billingType;
   /* confidence: 35% */

/* original: x7_ */ function helper_fn(){
  let q=o_();
  if(q?.emailAddress)return q.emailAddress;
  return
} /* confidence: 35% */

/* original: oaq */ function helper_fn(){
  let q=Object.fromEntries(wC),K=w8();
   /* confidence: 35% */

/* original: WP_ */ function helper_fn(q){
  let K=w8(),_=r_6(V08(q));
   /* confidence: 35% */

/* original: w8 */ function utility_fn(){
  if(gF.config)return rl6++,gF.config;
   /* confidence: 40% */

/* original: FF */ function helper_fn(){
  let q=w8().remoteControlAtStartup;
   /* confidence: 35% */

/* original: al6 */ function helper_fn(q){
  let K=w8();
   /* confidence: 35% */

/* original: kw */ function helper_fn(){
  let q=L08(),K=w8();
   /* confidence: 35% */

/* original: qC */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: RD_ */ function helper_fn(){
  return w8().theme
} /* confidence: 35% */

/* original: $S1 */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: TC */ function helper_fn(){
  let q=w8().companion;
   /* confidence: 35% */

/* original: TK4 */ function helper_fn(q){
  let K=TC();
   /* confidence: 35% */

/* original: Dm1 */ function helper_fn(){
  let q=kw();
  if(q.hasClaudeMdExternalIncludesApproved||q.hasClaudeMdExternalIncludesWarningShown)return!1;
  return tN8(await RH(!0))
} /* confidence: 35% */

/* original: nF1 */ function helper_fn(q){
  return(w8().claudeAiMcpEverConnected??[]).includes(q)
} /* confidence: 35% */

/* original: Kv */ function helper_fn(q){
  let K=kw();
   /* confidence: 35% */

/* original: _x4 */ function helper_fn(){
  let q=w8(),K=WN.terminal||"unknown";
   /* confidence: 35% */

/* original: VXz */ function helper_fn(){
  if(_x4())return;
   /* confidence: 35% */

/* original: Pm4 */ function config(){
  let q=w8().cachedExtraUsageDisabledReason;
  if(q===void 0)return!1;
  if(q===null)return!0;
  switch(q){
    case"out_of_credits":return!0;
    case"overage_not_provisioned":case"org_level_disabled":case"org_level_disabled_until":case"seat_tier_level_disabled":case"member_level_disabled":case"seat_tier_zero_credit_limit":case"group_zero_credit_limit":case"member_zero_credit_limit":case"org_service_level_disabled":case"org_service_zero_credit_limit":case"no_limits_configured":case"unknown":return!1;
    default:return!1
  }
} /* confidence: 95% */

/* original: Uy6 */ function http_client(){
  if(!Zl6())return!1;
  let q=o_()?.accountUuid;
  if(!q)return!1;
  let _=w8().groveConfigCache?.[q],z=Date.now();
  if(!_)return N("Grove: No cache, fetching config in background (dialog skipped this session)"),Lm4(q),!1;
  if(z-_.timestamp>hm4)return N("Grove: Cache stale, returning cached data and refreshing in background"),Lm4(q),_.grove_enabled;
  return N("Grove: Using fresh cached config"),_.grove_enabled
} /* confidence: 95% */

/* original: ZQ4 */ function helper_fn(){
  if(i7()&&!OD())return{
    enabled:!1,hasError:!1
  };
  let q=w8().metricsStatusCache;
  if(q){
    if(Date.now()-q.timestamp>fQ4)DQ4().catch(j6);
    return{
      enabled:q.enabled,hasError:!1
    }
  }return DQ4()
} /* confidence: 35% */

/* original: IKK */ function helper_fn(q){
  if(w8().preferTmuxOverIterm2!==q)S8((_)=>({
    ..._,preferTmuxOverIterm2:q
  } /* confidence: 35% */

/* original: uKK */ function helper_fn(){
  return w8().preferTmuxOverIterm2===!0
} /* confidence: 35% */

/* original: gj6 */ function helper_fn(){
  let q=o_()?.organizationUuid;
   /* confidence: 35% */

/* original: h5K */ function helper_fn(){
  let q=o_()?.organizationUuid;
   /* confidence: 35% */

/* original: I5K */ function helper_fn(){
  if(c6(process.env.DISABLE_EXTRA_USAGE_COMMAND))return!1;
  return zv6()
} /* confidence: 35% */

/* original: La1 */ function helper_fn(q){
  let _=w8().skillUsage?.[q];
   /* confidence: 35% */

/* original: yS6 */ function helper_fn(){
  if(kw().hasCompletedProjectOnboarding)return;
   /* confidence: 35% */

/* original: jjY */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: i47 */ function helper_fn(){
  return w8().shiftEnterKeyBindingInstalled===!0
} /* confidence: 35% */

/* original: r47 */ function helper_fn(){
  return w8().hasUsedBackslashReturn===!0
} /* confidence: 35% */

/* original: n */ function named_entity(l){
  if(H&&R.offset>0&&R.text[R.offset-1]==="\\")return o47(),R.backspace().insert(`
`);
  if(l.meta||l.shift)return R.insert(`
`);
  if(Y7.terminal==="Apple_Terminal"&&ZkK("shift"))return R.insert(`
`);
  _?.(q)
} /* confidence: 70% */

/* original: oJY */ function helper_fn(){
  try{
    return o_()
  } /* confidence: 35% */

/* original: $57 */ function helper_fn(){
  let q=gj6();
  if(!q||!q.available||q.granted)return!1;
  return bL6(q)!==null
} /* confidence: 35% */

/* original: O57 */ function helper_fn(){
  if(!$57())return!1;
   /* confidence: 35% */

/* original: ZMY */ function helper_fn(){
  if(gj6()!==null)return;
   /* confidence: 35% */

/* original: gs */ function helper_fn(){
  return w8().editorMode==="vim"
} /* confidence: 35% */

/* original: oEK */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: sEK */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: OkY */ function helper_fn(){
  let q=w8().powerupsUnlocked??[];
   /* confidence: 35% */

/* original: G36 */ function helper_fn(){
  let q=o_()?.organizationUuid;
   /* confidence: 35% */

/* original: $d8 */ function helper_fn(){
  let q=o_()?.organizationUuid;
   /* confidence: 35% */

/* original: V97 */ function http_client(){
  if(!xbK())return null;
  let q=o_()?.organizationUuid;
  if(!q)return null;
  let _=w8().passesEligibilityCache?.[q],z=Date.now();
  if(!_)return N("Passes: No cache, fetching eligibility in background (command unavailable this session)"),SbK(),null;
  if(z-_.timestamp>CbK){
    N("Passes: Cache stale, returning cached data and refreshing in background"),SbK();
    let{
      timestamp:O,...A
    }=_;
    return A
  }N("Passes: Using fresh cached eligibility data");
  let{
    timestamp:Y,...$
  }=_;
  return $
} /* confidence: 95% */

/* original: pbK */ function helper_fn(){
  let q=G36(),K=q?`Share Claude Code and earn ${Z36(q)} of extra usage`:"Share Claude Code with friends";
   /* confidence: 35% */

/* original: lkY */ function helper_fn(){
  let q=$d8();
  if(q==null||q<=0)return;
  let _=w8().passesLastSeenRemaining??0;
  if(q>_)S8((z)=>({
    ...z,passesUpsellSeenCount:0,hasVisitedPasses:!1,passesLastSeenRemaining:q
  }))
} /* confidence: 35% */

/* original: Md8 */ function MemoComponent(){
  let q=Y6(1),K;
  if(q[0]===Symbol.for("react.memo_cache_sentinel"))K=as.createElement(YVY,null),q[0]=K;
  else K=q[0];
  return K
} /* confidence: 87% */

/* original: dFK */ function helper_fn(){
  if(w8().remoteDialogSeen)return!1;
   /* confidence: 35% */

/* original: ZUK */ function helper_fn(q,K){
  let _=TC();
   /* confidence: 35% */

/* original: GUK */ function helper_fn(q,K){
  if(w8().companionMuted)return;
   /* confidence: 35% */

/* original: vUK */ function helper_fn(q){
  let K=TC();
   /* confidence: 35% */

/* original: MO7 */ function helper_fn(){
  return w8().permissionExplainerEnabled!==!1
} /* confidence: 35% */

/* original: QrK */ function helper_fn(q,K){
  let _=TC();
   /* confidence: 35% */

/* original: VsK */ function helper_fn(){
  return w8().prStatusFooterEnabled??!0
} /* confidence: 35% */

/* original: eiY */ function helper_fn(){
  return!w8().hasSeenUltraplanTerms
} /* confidence: 35% */

/* original: Ai8 */ function helper_fn(q){
  let K=w8(),_=K.tipsHistory?.[q];
   /* confidence: 35% */

/* original: urY */ function helper_fn(){
  let q=(w8().desktopUpsellSeenCount??0)+1;
   /* confidence: 35% */

/* original: FrY */ function helper_fn(q){
  if(q.length===0)return;
  if(q.length===1)return q[0];
  let K=q.map((_)=>({
    tip:_,sessions:Ai8(_.id)
  }));
  return K.sort((_,z)=>z.sessions-_.sessions),K[0]?.tip
} /* confidence: 35% */

/* original: RoY */ function typed_entity(q){
  return q.type==="needs-auth"&&q.config.type==="claudeai-proxy"&&nF1(q.name)
} /* confidence: 70% */

/* original: CoY */ function typed_entity(q){
  return q.type==="failed"&&q.config.type==="claudeai-proxy"&&nF1(q.name)
} /* confidence: 70% */

/* original: UoY */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: K15 */ function helper_fn(){
  Cx(MaY)
} /* confidence: 35% */

/* original: MaY */ function helper_fn(){
  let q=w8(),K=[];
   /* confidence: 35% */

/* original: Si8 */ function helper_fn(q){
  let K=w8(),_=q.toLowerCase();
   /* confidence: 35% */

/* original: FK5 */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

/* original: z8$ */ function helper_fn(){
  let q=w8();
   /* confidence: 35% */

