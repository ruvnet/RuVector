// Module: essential_traffic

/* original: cL7 */ function essentialtraffic_notelemetry_d(){
  if(process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC)return"essential-traffic";
  if(process.env.DISABLE_TELEMETRY)return"no-telemetry";
  return"default"
} /* confidence: 65% */

/* original: XY */ function SseTransport(){
  return cL7()==="essential-traffic"
} /* confidence: 30% */

/* original: dK1 */ function helper_fn(){
  return cL7()!=="default"
} /* confidence: 35% */

/* original: c16 */ function helper_fn(){
  return c6(process.env.CLAUDE_CODE_USE_BEDROCK)||c6(process.env.CLAUDE_CODE_USE_VERTEX)||c6(process.env.CLAUDE_CODE_USE_FOUNDRY)||dK1()
} /* confidence: 35% */

/* original: Zv6 */ function helper_fn(){
  return dK1()
} /* confidence: 35% */

/* original: NO6 */ function helper_fn(){
  return!c16()
} /* confidence: 35% */

/* original: KE1 */ function helper_fn(q){
  if(!NO6())return;
   /* confidence: 35% */

/* original: Or */ function helper_fn(){
  return NO6()
} /* confidence: 35% */

/* original: OE1 */ function helper_fn(q){
  let K=Nv6();
  if(K&&q in K)return Boolean(K[q]);
  let _=yv6();
  if(_&&q in _)return Boolean(_[q]);
  if(!Or())return!1;
  if(dl6)await dl6;
  let z=w8(),Y=z.cachedStatsigGates?.[q];
  if(Y!==void 0)return Boolean(Y);
  let $=z.cachedGrowthBookFeatures?.[q];
  if($!==void 0)return Boolean($);
  return!1
} /* confidence: 35% */

/* original: LO6 */ function helper_fn(){
  if(!Or())return;
   /* confidence: 35% */

/* original: qsq */ function helper_fn(){
  if(!Or())return;
   /* confidence: 35% */

/* original: Hvz */ function helper_fn(){
  if(XY())return{
    enabled:!1,hasError:!1
  };
  try{
    let q=await oi(jvz,{
      also403Revoked:!0
    });
    return N(`Metrics opt-out API response: enabled=${q.metrics_logging_enabled}`),{
      enabled:q.metrics_logging_enabled,hasError:!1
    }
  }catch(q){
    return N(`Failed to check metrics opt-out status: ${F6(q)}`),j6(q),{
      enabled:!1,hasError:!0
    }
  }
} /* confidence: 35% */

/* original: hI8 */ function permission_handler(){
  return OE1("tengu_disable_bypass_permissions_mode")
} /* confidence: 95% */

/* original: IbK */ function helper_fn(){
  if(XY())return;
  V97()
} /* confidence: 35% */

/* original: Zc8 */ function helper_fn(){
  if(Dq()!=="firstParty")return!1;
  if(XY())return!1;
  let q=new Date;
  return q.getFullYear()>2026||q.getFullYear()===2026&&q.getMonth()>=3
} /* confidence: 35% */

