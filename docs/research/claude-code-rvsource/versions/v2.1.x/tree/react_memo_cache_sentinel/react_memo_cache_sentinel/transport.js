// Module: transport

/* original: Fl4 */ var status=B((Bl4)=>{
  Object.defineProperty(Bl4,"__esModule",{
    value:!0
  });
  Bl4.createRetryingTransport=void 0;
  var ikz=5,rkz=1000,okz=5000,akz=1.5,ml4=0.2;
  function skz(){
    return Math.random()*(2*ml4)-ml4
  }class pl4{
    _transport;
    constructor(q){
      this._transport=q
    }retry(q,K,_){
      return new Promise((z,Y)=>{
        setTimeout(()=>{
          this._transport.send(q,K).then(z,Y)
        },_)
      })
    }async send(q,K){
      let _=Date.now()+K,z=await this._transport.send(q,K),Y=ikz,$=rkz;
      while(z.status==="retryable"&&Y>0){
        Y--;
        let O=Math.max(Math.min($,okz)+skz(),0);
        $=$*akz;
        let A=z.retryInMillis??O,w=_-Date.now();
        if(A>w)return z;
        z=await this.retry(q,w,A)
      }return z
    }shutdown(){
      return this._transport.shutdown()
    }
  }function tkz(q){
    return new pl4(q.transport)
  }Bl4.createRetryingTransport=tkz
} /* confidence: 70% */

/* original: dl4 */ var composed_value=B((Ul4)=>{
  Object.defineProperty(Ul4,"__esModule",{
    value:!0
  });
  Ul4.createOtlpHttpExportDelegate=void 0;
  var ekz=Fd1(),qVz=ul4(),KVz=gd1(),_Vz=Fl4();
  function zVz(q,K){
    return(0,ekz.createOtlpExportDelegate)({
      transport:(0,_Vz.createRetryingTransport)({
        transport:(0,qVz.createHttpExporterTransport)(q)
      }),serializer:K,promiseHandler:(0,KVz.createBoundedQueueExportPromiseHandler)(q)
    },{
      timeout:q.timeoutMillis
    })
  }Ul4.createOtlpHttpExportDelegate=zVz
} /* confidence: 30% */

/* original: ekz */ var composed_value=Fd1(),qVz=ul4(),KVz=gd1(),_Vz=Fl4(); /* confidence: 30% */

/* original: tkz */ function helper_fn(q){
  return new pl4(q.transport)
} /* confidence: 35% */

/* original: pl4 */ class entity_class{
  _transport;
   /* confidence: 45% */

