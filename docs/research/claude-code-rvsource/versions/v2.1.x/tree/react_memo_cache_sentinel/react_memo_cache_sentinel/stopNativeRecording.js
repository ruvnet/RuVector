// Module: stopNativeRecording

/* original: i87 */ var 20_3=null,TPK=null,s87=16000,t87=1,n5Y="2.0",kPK="3%",r87=null,o87=null,fs=null,LR6=!1; /* confidence: 65% */

/* original: s5Y */ function helper_fn(){
  if(!(await oB8()).isNativeAudioAvailable())return!0;
  if(await yPK((_)=>{
    
  },()=>{
    
  },{
    silenceDetection:!1
  }))return EPK(),!0;
  return!1
} /* confidence: 35% */

/* original: EPK */ function helper_fn(){
  if(LR6&&i87){
    i87.stopNativeRecording(),LR6=!1;
    return
  } /* confidence: 35% */

