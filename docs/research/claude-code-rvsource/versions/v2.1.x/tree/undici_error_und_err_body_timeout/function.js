// Module: function

/* original: Dj */ var dispatcher=B((cQ$,hx7)=>{
  hx7.exports={
    kClose:Symbol("close"),kDestroy:Symbol("destroy"),kDispatch:Symbol("dispatch"),kUrl:Symbol("url"),kWriting:Symbol("writing"),kResuming:Symbol("resuming"),kQueue:Symbol("queue"),kConnect:Symbol("connect"),kConnecting:Symbol("connecting"),kKeepAliveDefaultTimeout:Symbol("default keep alive timeout"),kKeepAliveMaxTimeout:Symbol("max keep alive timeout"),kKeepAliveTimeoutThreshold:Symbol("keep alive timeout threshold"),kKeepAliveTimeoutValue:Symbol("keep alive timeout"),kKeepAlive:Symbol("keep alive"),kHeadersTimeout:Symbol("headers timeout"),kBodyTimeout:Symbol("body timeout"),kServerName:Symbol("server name"),kLocalAddress:Symbol("local address"),kHost:Symbol("host"),kNoRef:Symbol("no ref"),kBodyUsed:Symbol("used"),kBody:Symbol("abstracted request body"),kRunning:Symbol("running"),kBlocking:Symbol("blocking"),kPending:Symbol("pending"),kSize:Symbol("size"),kBusy:Symbol("busy"),kQueued:Symbol("queued"),kFree:Symbol("free"),kConnected:Symbol("connected"),kClosed:Symbol("closed"),kNeedDrain:Symbol("need drain"),kReset:Symbol("reset"),kDestroyed:Symbol.for("nodejs.stream.destroyed"),kResume:Symbol("resume"),kOnError:Symbol("on error"),kMaxHeadersSize:Symbol("max headers size"),kRunningIdx:Symbol("running index"),kPendingIdx:Symbol("pending index"),kError:Symbol("error"),kClients:Symbol("clients"),kClient:Symbol("client"),kParser:Symbol("parser"),kOnDestroyed:Symbol("destroy callbacks"),kPipelining:Symbol("pipelining"),kSocket:Symbol("socket"),kHostHeader:Symbol("host header"),kConnector:Symbol("connector"),kStrictContentLength:Symbol("strict content length"),kMaxRedirections:Symbol("maxRedirections"),kMaxRequests:Symbol("maxRequestsPerClient"),kProxy:Symbol("proxy agent options"),kCounter:Symbol("socket request counter"),kInterceptors:Symbol("dispatch interceptors"),kMaxResponseSize:Symbol("max response size"),kHTTP2Session:Symbol("http2Session"),kHTTP2SessionState:Symbol("http2Session state"),kRetryHandlerDefaultRetry:Symbol("retry agent default retry"),kConstruct:Symbol("constructable"),kListeners:Symbol("listeners"),kHTTPContext:Symbol("http context"),kMaxConcurrentStreams:Symbol("max concurrent streams"),kNoProxyAgent:Symbol("no proxy agent"),kHttpProxyAgent:Symbol("http proxy agent"),kHttpsProxyAgent:Symbol("https proxy agent")
  }
} /* confidence: 95% */

/* original: _f6 */ var error_handler=B((tQ$,tI7)=>{
  var ru5=rB6(),{
    ClientDestroyedError:A91,ClientClosedError:ou5,InvalidArgumentError:eD6
  }=t$(),{
    kDestroy:au5,kClose:su5,kClosed:oB6,kDestroyed:qf6,kDispatch:w91,kInterceptors:Xz6
  }=Dj(),Mn=Symbol("onDestroyed"),Kf6=Symbol("onClosed"),Rw8=Symbol("Intercepted Dispatch");
  class sI7 extends ru5{
    constructor(){
      super();
      this[qf6]=!1,this[Mn]=null,this[oB6]=!1,this[Kf6]=[]
    }get destroyed(){
      return this[qf6]
    }get closed(){
      return this[oB6]
    }get interceptors(){
      return this[Xz6]
    }set interceptors(q){
      if(q){
        for(let K=q.length-1;
        K>=0;
        K--)if(typeof this[Xz6][K]!=="function")throw new eD6("interceptor must be an function")
      }this[Xz6]=q
    }close(q){
      if(q===void 0)return new Promise((_,z)=>{
        this.close((Y,$)=>{
          return Y?z(Y):_($)
        })
      });
      if(typeof q!=="function")throw new eD6("invalid callback");
      if(this[qf6]){
        queueMicrotask(()=>q(new A91,null));
        return
      }if(this[oB6]){
        if(this[Kf6])this[Kf6].push(q);
        else queueMicrotask(()=>q(null,null));
        return
      }this[oB6]=!0,this[Kf6].push(q);
      let K=()=>{
        let _=this[Kf6];
        this[Kf6]=null;
        for(let z=0;
        z<_.length;
        z++)_[z](null,null)
      };
      this[su5]().then(()=>this.destroy()).then(()=>{
        queueMicrotask(K)
      })
    }destroy(q,K){
      if(typeof q==="function")K=q,q=null;
      if(K===void 0)return new Promise((z,Y)=>{
        this.destroy(q,($,O)=>{
          return $?Y($):z(O)
        })
      });
      if(typeof K!=="function")throw new eD6("invalid callback");
      if(this[qf6]){
        if(this[Mn])this[Mn].push(K);
        else queueMicrotask(()=>K(null,null));
        return
      }if(!q)q=new A91;
      this[qf6]=!0,this[Mn]=this[Mn]||[],this[Mn].push(K);
      let _=()=>{
        let z=this[Mn];
        this[Mn]=null;
        for(let Y=0;
        Y<z.length;
        Y++)z[Y](null,null)
      };
      this[au5](q).then(()=>{
        queueMicrotask(_)
      })
    }[Rw8](q,K){
      if(!this[Xz6]||this[Xz6].length===0)return this[Rw8]=this[w91],this[w91](q,K);
      let _=this[w91].bind(this);
      for(let z=this[Xz6].length-1;
      z>=0;
      z--)_=this[Xz6][z](_);
      return this[Rw8]=_,_(q,K)
    }dispatch(q,K){
      if(!K||typeof K!=="object")throw new eD6("handler must be an object");
      try{
        if(!q||typeof q!=="object")throw new eD6("opts must be an object.");
        if(this[qf6]||this[Mn])throw new A91;
        if(this[oB6])throw new ou5;
        return this[Rw8](q,K)
      }catch(_){
        if(typeof K.onError!=="function")throw new eD6("invalid onError method");
        return K.onError(_),!1
      }
    }
  }tI7.exports=sI7
} /* confidence: 95% */

/* original: Gp7 */ var composed_value=B((Td$,Zp7)=>{
  var{
    kFree:DF5,kConnected:fF5,kPending:ZF5,kQueued:GF5,kRunning:vF5,kSize:TF5
  }=Dj(),Gz6=Symbol("pool");
  class fp7{
    constructor(q){
      this[Gz6]=q
    }get connected(){
      return this[Gz6][fF5]
    }get free(){
      return this[Gz6][DF5]
    }get pending(){
      return this[Gz6][ZF5]
    }get queued(){
      return this[Gz6][GF5]
    }get running(){
      return this[Gz6][vF5]
    }get size(){
      return this[Gz6][TF5]
    }
  }Zp7.exports=fp7
} /* confidence: 30% */

/* original: Y_1 */ var error_handler=B((kd$,Sp7)=>{
  var kF5=_f6(),VF5=e91(),{
    kConnected:q_1,kSize:vp7,kRunning:Tp7,kPending:kp7,kQueued:Wg6,kBusy:NF5,kFree:yF5,kUrl:EF5,kClose:LF5,kDestroy:hF5,kDispatch:RF5
  }=Dj(),SF5=Gp7(),SV=Symbol("clients"),WT=Symbol("needDrain"),Dg6=Symbol("queue"),K_1=Symbol("closed resolve"),__1=Symbol("onDrain"),Vp7=Symbol("onConnect"),Np7=Symbol("onDisconnect"),yp7=Symbol("onConnectionError"),z_1=Symbol("get dispatcher"),Lp7=Symbol("add client"),hp7=Symbol("remove client"),Ep7=Symbol("stats");
  class Rp7 extends kF5{
    constructor(){
      super();
      this[Dg6]=new VF5,this[SV]=[],this[Wg6]=0;
      let q=this;
      this[__1]=function(_,z){
        let Y=q[Dg6],$=!1;
        while(!$){
          let O=Y.shift();
          if(!O)break;
          q[Wg6]--,$=!this.dispatch(O.opts,O.handler)
        }if(this[WT]=$,!this[WT]&&q[WT])q[WT]=!1,q.emit("drain",_,[q,...z]);
        if(q[K_1]&&Y.isEmpty())Promise.all(q[SV].map((O)=>O.close())).then(q[K_1])
      },this[Vp7]=(K,_)=>{
        q.emit("connect",K,[q,..._])
      },this[Np7]=(K,_,z)=>{
        q.emit("disconnect",K,[q,..._],z)
      },this[yp7]=(K,_,z)=>{
        q.emit("connectionError",K,[q,..._],z)
      },this[Ep7]=new SF5(this)
    }get[NF5](){
      return this[WT]
    }get[q_1](){
      return this[SV].filter((q)=>q[q_1]).length
    }get[yF5](){
      return this[SV].filter((q)=>q[q_1]&&!q[WT]).length
    }get[kp7](){
      let q=this[Wg6];
      for(let{
        [kp7]:K
      }of this[SV])q+=K;
      return q
    }get[Tp7](){
      let q=0;
      for(let{
        [Tp7]:K
      }of this[SV])q+=K;
      return q
    }get[vp7](){
      let q=this[Wg6];
      for(let{
        [vp7]:K
      }of this[SV])q+=K;
      return q
    }get stats(){
      return this[Ep7]
    }async[LF5](){
      if(this[Dg6].isEmpty())await Promise.all(this[SV].map((q)=>q.close()));
      else await new Promise((q)=>{
        this[K_1]=q
      })
    }async[hF5](q){
      while(!0){
        let K=this[Dg6].shift();
        if(!K)break;
        K.handler.onError(q)
      }await Promise.all(this[SV].map((K)=>K.destroy(q)))
    }[RF5](q,K){
      let _=this[z_1]();
      if(!_)this[WT]=!0,this[Dg6].push({
        opts:q,handler:K
      }),this[Wg6]++;
      else if(!_.dispatch(q,K))_[WT]=!0,this[WT]=!this[z_1]();
      return!this[WT]
    }[Lp7](q){
      if(q.on("drain",this[__1]).on("connect",this[Vp7]).on("disconnect",this[Np7]).on("connectionError",this[yp7]),this[SV].push(q),this[WT])queueMicrotask(()=>{
        if(this[WT])this[__1](q[EF5],[this,q])
      });
      return this
    }[hp7](q){
      q.close(()=>{
        let K=this[SV].indexOf(q);
        if(K!==-1)this[SV].splice(K,1)
      }),this[WT]=this[SV].some((K)=>!K[WT]&&K.closed!==!0&&K.destroyed!==!0)
    }
  }Sp7.exports={
    PoolBase:Rp7,kClients:SV,kNeedDrain:WT,kAddClient:Lp7,kRemoveClient:hp7,kGetDispatcher:z_1
  }
} /* confidence: 95% */

/* original: F28 */ var composed_value=B((wc$,pU7)=>{
  pU7.exports={
    kConstruct:Dj().kConstruct
  }
} /* confidence: 30% */

