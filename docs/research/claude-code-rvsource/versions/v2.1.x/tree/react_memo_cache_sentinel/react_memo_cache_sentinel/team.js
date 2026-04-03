// Module: team

/* original: q */ let composed_value=jK(); /* confidence: 30% */

/* original: z */ let composed_value=jK(),Y=composed_value==="team"||composed_value==="enterprise",$=o_()?.hasExtraUsageEnabled===!0; /* confidence: 30% */

/* original: _ */ let composed_value=q.utilization?Math.floor(q.utilization*100):void 0,z=q.resetsAt?uP6(q.resetsAt,!0):void 0,Y=HAz(q.rateLimitType); /* confidence: 30% */

/* original: Y */ let composed_value=jK(); /* confidence: 30% */

/* original: q */ let composed_value=jK(),K=composed_value==="team"||composed_value==="enterprise"; /* confidence: 30% */

/* original: H */ let composed_value=jK(),J=composed_value==="max"||composed_value==="team"||composed_value===null,M=[{
  title:"Current session",limit:q.five_hour
} /* confidence: 30% */

/* original: z */ let composed_value=jK(),Y=composed_value==="team"||composed_value==="enterprise",$=Y?V1("policySettings"):void 0; /* confidence: 30% */

/* original: GOq */ function utility_fn(){
  return iM8&&WOq!==null&&!1
} /* confidence: 40% */

/* original: D06 */ function helper_fn(q=!1){
  if(bS()||p86()){
    if(vJ())return`Opus 4.6 with 1M context Â· Most capable for complex work${q?Oi(!0):""}`;
    return`Opus 4.6 Â· Most capable for complex work${q?Oi(!0):""}`
  } /* confidence: 35% */

/* original: vJ */ function helper_fn(){
  if(m86()||xS()||Dq()!=="firstParty")return!1;
   /* confidence: 35% */

/* original: h7_ */ function helper_fn(){
  let q=jK();
   /* confidence: 35% */

/* original: jK */ function helper_fn(){
  if(GOq())return ZOq();
   /* confidence: 35% */

/* original: bS */ function helper_fn(){
  return jK()==="max"
} /* confidence: 35% */

/* original: KO6 */ function helper_fn(){
  return jK()==="team"
} /* confidence: 35% */

/* original: p86 */ function team_default_claude_max_5x(){
  return jK()==="team"&&xF()==="default_claude_max_5x"
} /* confidence: 65% */

/* original: Yv6 */ function helper_fn(){
  return jK()==="enterprise"
} /* confidence: 35% */

/* original: xS */ function helper_fn(){
  return jK()==="pro"
} /* confidence: 35% */

/* original: ZZ8 */ function enterprise_ClaudeEnterprise_te(){
  switch(jK()){
    case"enterprise":return"Claude Enterprise";
    case"team":return"Claude Team";
    case"max":return"Claude Max";
    case"pro":return"Claude Pro";
    default:return"Claude API"
  } /* confidence: 65% */

/* original: Zl6 */ function helper_fn(){
  let q=jK();
   /* confidence: 35% */

/* original: Hr6 */ function opus46_medium_medium(q){
  if(q.toLowerCase().includes("opus-4-6")){
    if(xS())return"medium";
    if(jr6().enabled&&(bS()||KO6()))return"medium"
  } /* confidence: 65% */

/* original: HAz */ function helper_fn(q){
  let K=jK(),_=o_()?.hasExtraUsageEnabled===!0;
   /* confidence: 35% */

/* original: IPz */ function helper_fn(){
  let q=jK();
   /* confidence: 35% */

/* original: U87 */ function pro_tengu_gypsum_kite_2usagevs(){
  if(jK()==="pro"&&L8("tengu_gypsum_kite",!1))return" Â· ~2Ã usage vs Sonnet";
   /* confidence: 65% */

/* original: xbK */ function helper_fn(){
  return!!(o_()?.organizationUuid&&i7()&&jK()==="max")
} /* confidence: 35% */

/* original: ZC6 */ function helper_fn(){
  if(!xbK())return{
    eligible:!1,needsRefresh:!1,hasCache:!1
  };
   /* confidence: 35% */

/* original: jIK */ function proceed_proceed_proceed_procee(){
  if(KO6()||Yv6())return{
    kind:"proceed",billingNote:""
  };
  let[q,K]=await Promise.all([$IK(),xL6().catch(()=>null)]);
  if(!q)return{
    kind:"proceed",billingNote:""
  };
  if(q.reviews_remaining>0)return{
    kind:"proceed",billingNote:` This is free ultrareview ${q.reviews_used+1} of ${q.reviews_limit}.`
  };
  if(!K)return{
    kind:"proceed",billingNote:""
  };
  let _=K.extra_usage;
  if(!_?.is_enabled)return d("tengu_review_overage_not_enabled",{
    
  }),{
    kind:"not-enabled"
  };
  let z=_.monthly_limit,Y=_.used_credits??0,$=z===null||z===void 0?1/0:z-Y;
  if($<10)return d("tengu_review_overage_low_balance",{
    available:$
  }),{
    kind:"low-balance",available:$
  };
  if(!AIK)return d("tengu_review_overage_dialog_shown",{
    
  }),{
    kind:"needs-confirm"
  };
  return{
    kind:"proceed",billingNote:" This review bills as Extra Usage."
  }
} /* confidence: 65% */

