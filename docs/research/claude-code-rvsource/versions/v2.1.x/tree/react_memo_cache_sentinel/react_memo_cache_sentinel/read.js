// Module: read

/* original: HmY */ function CliSpinner(){
  zj("Initializing...");
  let q=new $cK,K=new OcK;
  await q.start();
  while(!0){
    let _=await K.read();
    if(_===null)break;
    await q.handleMessage(_)
  }await q.stop()
} /* confidence: 36% */

/* original: OcK */ class entity_class{
  buffer=Buffer.alloc(0);
   /* confidence: 45% */

