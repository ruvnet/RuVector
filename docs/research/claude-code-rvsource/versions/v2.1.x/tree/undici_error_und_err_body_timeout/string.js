// Module: string

/* original: hf6 */ var dispatcher=B((gd$,Zg7)=>{
  Zg7.exports={
    kAgent:Symbol("agent"),kOptions:Symbol("options"),kFactory:Symbol("factory"),kDispatches:Symbol("dispatches"),kDispatchKey:Symbol("dispatch key"),kDefaultHeaders:Symbol("default headers"),kDefaultTrailers:Symbol("default trailers"),kContentLength:Symbol("content length"),kMockAgent:Symbol("mock agent"),kMockAgentSet:Symbol("mock agent set"),kMockAgentGet:Symbol("mock agent get"),kMockDispatch:Symbol("mock dispatch"),kClose:Symbol("close"),kOriginalClose:Symbol("original agent close"),kOrigin:Symbol("origin"),kIsMockActive:Symbol("is mock active"),kNetConnect:Symbol("net connect"),kGetNetConnect:Symbol("get net connect"),kConnected:Symbol("connected")
  }
} /* confidence: 95% */

/* original: KF7 */ var function_Argumentoptsagentmust=B((ld$,qF7)=>{
  var{
    kClients:yz6
  }=Dj(),qd5=vf6(),{
    kAgent:F_1,kMockAgentSet:W28,kMockAgentGet:sg7,kDispatches:U_1,kIsMockActive:D28,kNetConnect:Ez6,kGetNetConnect:Kd5,kOptions:f28,kFactory:Z28
  }=hf6(),_d5=p_1(),zd5=g_1(),{
    matchValue:Yd5,buildMockOptions:$d5
  }=Ng6(),{
    InvalidArgumentError:tg7,UndiciError:Od5
  }=t$(),Ad5=rB6(),wd5=rg7(),jd5=ag7();
  class eg7 extends Ad5{
    constructor(q){
      super(q);
      if(this[Ez6]=!0,this[D28]=!0,q?.agent&&typeof q.agent.dispatch!=="function")throw new tg7("Argument opts.agent must implement Agent");
      let K=q?.agent?q.agent:new qd5(q);
      this[F_1]=K,this[yz6]=K[yz6],this[f28]=$d5(q)
    }get(q){
      let K=this[sg7](q);
      if(!K)K=this[Z28](q),this[W28](q,K);
      return K
    }dispatch(q,K){
      return this.get(q.origin),this[F_1].dispatch(q,K)
    }async close(){
      await this[F_1].close(),this[yz6].clear()
    }deactivate(){
      this[D28]=!1
    }activate(){
      this[D28]=!0
    }enableNetConnect(q){
      if(typeof q==="string"||typeof q==="function"||q instanceof RegExp)if(Array.isArray(this[Ez6]))this[Ez6].push(q);
      else this[Ez6]=[q];
      else if(typeof q>"u")this[Ez6]=!0;
      else throw new tg7("Unsupported matcher. Must be one of String|Function|RegExp.")
    }disableNetConnect(){
      this[Ez6]=!1
    }get isMockActive(){
      return this[D28]
    }[W28](q,K){
      this[yz6].set(q,K)
    }[Z28](q){
      let K=Object.assign({
        agent:this
      },this[f28]);
      return this[f28]&&this[f28].connections===1?new _d5(q,K):new zd5(q,K)
    }[sg7](q){
      let K=this[yz6].get(q);
      if(K)return K;
      if(typeof q!=="string"){
        let _=this[Z28]("http://localhost:9999");
        return this[W28](q,_),_
      }for(let[_,z]of Array.from(this[yz6]))if(z&&typeof _!=="string"&&Yd5(_,q)){
        let Y=this[Z28](q);
        return this[W28](q,Y),Y[U_1]=z[U_1],Y
      }
    }[Kd5](){
      return this[Ez6]
    }pendingInterceptors(){
      let q=this[yz6];
      return Array.from(q.entries()).flatMap(([K,_])=>_[U_1].map((z)=>({
        ...z,origin:K
      }))).filter(({
        pending:K
      })=>K)
    }assertNoPendingInterceptors({
      pendingInterceptorsFormatter:q=new jd5
    }={
      
    }){
      let K=this.pendingInterceptors();
      if(K.length===0)return;
      let _=new wd5("interceptor","interceptors").pluralize(K.length);
      throw new Od5(`
${_.count} ${_.noun} ${_.is} pending:

${q.format(K)}
`.trim())
    }
  }qF7.exports=eg7
} /* confidence: 65% */

