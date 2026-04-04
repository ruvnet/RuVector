// Module: trackedFeatureUsage

/* original: H */ let composed_value={
  result:j,jsonrpc:"2.0",id:q.id
}; /* confidence: 30% */

/* original: u1z */ var composed_value={
  "after:highlightElement":({
    el:q,result:K,text:_
  } /* confidence: 30% */

/* original: z */ let composed_value=()=>{
  d("tengu_remote_setup_result",{
    result:"cancelled"
  }),q()
} /* confidence: 30% */

const WriteTool = block.content; /* confidence: 31% */

/* original: Mh5 */ function value_holder(q,K,_){
  if(q.user.trackedFeatureUsage){
    let z=JSON.stringify(_.value);
    if(q.user.trackedFeatureUsage[K]===z)return;
    if(q.user.trackedFeatureUsage[K]=z,q.user.enableDevMode&&q.user.devLogs)q.user.devLogs.push({
      featureKey:K,result:_,timestamp:Date.now().toString(),logType:"feature"
    })
  } /* confidence: 70% */

/* original: X2 */ function helper_fn(q){
  let K=Y6(25),{
    result:_,verbose:z
  } /* confidence: 35% */

/* original: C2K */ function helper_fn({
  bytes:q,code:K,codeText:_,result:z
} /* confidence: 35% */

/* original: KCY */ function helper_fn(q){
  let K=Y6(6),{
    result:_,onDone:z
  } /* confidence: 35% */

/* original: Ob6 */ function result_success(q){
  return{
    type:"result",subtype:"success",duration_ms:0,duration_api_ms:0,is_error:!1,num_turns:0,result:"",stop_reason:null,total_cost_usd:0,usage:{
      ...wf
    },modelUsage:{
      
    },permission_denials:[],session_id:q,uuid:UBY()
  } /* confidence: 65% */

