// Module: default

/* original: z56 */ var action_creator=B(($YK)=>{
  Object.defineProperty($YK,"__esModule",{
    value:!0
  });
  var Es1;
  function Ls1(){
    if(Es1===void 0)throw Error("No runtime abstraction layer installed");
    return Es1
  }(function(q){
    function K(_){
      if(_===void 0)throw Error("No runtime abstraction layer provided");
      Es1=_
    }q.install=K
  })(Ls1||(Ls1={
    
  }));
  $YK.default=Ls1
} /* confidence: 95% */

/* original: Ah6 */ var __esModule_Whenaddingalistener=B((wYK)=>{
  Object.defineProperty(wYK,"__esModule",{
    value:!0
  });
  wYK.Emitter=wYK.Event=void 0;
  var Grz=z56(),OYK;
  (function(q){
    let K={
      dispose(){
        
      }
    };
    q.None=function(){
      return K
    }
  })(OYK||(wYK.Event=OYK={
    
  }));
  class AYK{
    add(q,K=null,_){
      if(!this._callbacks)this._callbacks=[],this._contexts=[];
      if(this._callbacks.push(q),this._contexts.push(K),Array.isArray(_))_.push({
        dispose:()=>this.remove(q,K)
      })
    }remove(q,K=null){
      if(!this._callbacks)return;
      let _=!1;
      for(let z=0,Y=this._callbacks.length;
      z<Y;
      z++)if(this._callbacks[z]===q)if(this._contexts[z]===K){
        this._callbacks.splice(z,1),this._contexts.splice(z,1);
        return
      }else _=!0;
      if(_)throw Error("When adding a listener with a context, you should remove it with the same context")
    }invoke(...q){
      if(!this._callbacks)return[];
      let K=[],_=this._callbacks.slice(0),z=this._contexts.slice(0);
      for(let Y=0,$=_.length;
      Y<$;
      Y++)try{
        K.push(_[Y].apply(z[Y],q))
      }catch(O){
        (0,Grz.default)().console.error(O)
      }return K
    }isEmpty(){
      return!this._callbacks||this._callbacks.length===0
    }dispose(){
      this._callbacks=void 0,this._contexts=void 0
    }
  }class Ym8{
    constructor(q){
      this._options=q
    }get event(){
      if(!this._event)this._event=(q,K,_)=>{
        if(!this._callbacks)this._callbacks=new AYK;
        if(this._options&&this._options.onFirstListenerAdd&&this._callbacks.isEmpty())this._options.onFirstListenerAdd(this);
        this._callbacks.add(q,K);
        let z={
          dispose:()=>{
            if(!this._callbacks)return;
            if(this._callbacks.remove(q,K),z.dispose=Ym8._noop,this._options&&this._options.onLastListenerRemove&&this._callbacks.isEmpty())this._options.onLastListenerRemove(this)
          }
        };
        if(Array.isArray(_))_.push(z);
        return z
      };
      return this._event
    }fire(q){
      if(this._callbacks)this._callbacks.invoke.call(this._callbacks,q)
    }dispose(){
      if(this._callbacks)this._callbacks.dispose(),this._callbacks=void 0
    }
  }wYK.Emitter=Ym8;
  Ym8._noop=function(){
    
  }
} /* confidence: 65% */

/* original: Grz */ var composed_value=z56(),OYK; /* confidence: 30% */

/* original: Om8 */ var composed_value=B((JYK)=>{
  Object.defineProperty(JYK,"__esModule",{
    value:!0
  });
  JYK.CancellationTokenSource=JYK.CancellationToken=void 0;
  var Trz=z56(),krz=Oh6(),hs1=Ah6(),$m8;
  (function(q){
    q.None=Object.freeze({
      isCancellationRequested:!1,onCancellationRequested:hs1.Event.None
    }),q.Cancelled=Object.freeze({
      isCancellationRequested:!0,onCancellationRequested:hs1.Event.None
    });
    function K(_){
      let z=_;
      return z&&(z===q.None||z===q.Cancelled||krz.boolean(z.isCancellationRequested)&&!!z.onCancellationRequested)
    }q.is=K
  })($m8||(JYK.CancellationToken=$m8={
    
  }));
  var Vrz=Object.freeze(function(q,K){
    let _=(0,Trz.default)().timer.setTimeout(q.bind(K),0);
    return{
      dispose(){
        _.dispose()
      }
    }
  });
  class Rs1{
    constructor(){
      this._isCancelled=!1
    }cancel(){
      if(!this._isCancelled){
        if(this._isCancelled=!0,this._emitter)this._emitter.fire(void 0),this.dispose()
      }
    }get isCancellationRequested(){
      return this._isCancelled
    }get onCancellationRequested(){
      if(this._isCancelled)return Vrz;
      if(!this._emitter)this._emitter=new hs1.Emitter;
      return this._emitter.event
    }dispose(){
      if(this._emitter)this._emitter.dispose(),this._emitter=void 0
    }
  }class HYK{
    get token(){
      if(!this._token)this._token=new Rs1;
      return this._token
    }cancel(){
      if(!this._token)this._token=$m8.Cancelled;
      else this._token.cancel()
    }dispose(){
      if(!this._token)this._token=$m8.None;
      else if(this._token instanceof Rs1)this._token.dispose()
    }
  }JYK.CancellationTokenSource=HYK
} /* confidence: 30% */

/* original: Trz */ var composed_value=z56(),krz=Oh6(),hs1=Ah6(),$m8; /* confidence: 30% */

/* original: Vrz */ var composed_value=Object.freeze(function(q,K){
  let _=(0,Trz.default)().timer.setTimeout(q.bind(K),0);
  return{
    dispose(){
      _.dispose()
    }
  }
} /* confidence: 30% */

/* original: yrz */ var composed_value=Om8(),s88; /* confidence: 30% */

/* original: Ss1 */ var __esModule_Capacitymustbegreat=B((TYK)=>{
  Object.defineProperty(TYK,"__esModule",{
    value:!0
  });
  TYK.Semaphore=void 0;
  var Lrz=z56();
  class vYK{
    constructor(q=1){
      if(q<=0)throw Error("Capacity must be greater than 0");
      this._capacity=q,this._active=0,this._waiting=[]
    }lock(q){
      return new Promise((K,_)=>{
        this._waiting.push({
          thunk:q,resolve:K,reject:_
        }),this.runNext()
      })
    }get active(){
      return this._active
    }runNext(){
      if(this._waiting.length===0||this._active===this._capacity)return;
      (0,Lrz.default)().timer.setImmediate(()=>this.doRunNext())
    }doRunNext(){
      if(this._waiting.length===0||this._active===this._capacity)return;
      let q=this._waiting.shift();
      if(this._active++,this._active>this._capacity)throw Error("To many thunks active");
      try{
        let K=q.thunk();
        if(K instanceof Promise)K.then((_)=>{
          this._active--,q.resolve(_),this.runNext()
        },(_)=>{
          this._active--,q.reject(_),this.runNext()
        });
        else this._active--,q.resolve(K),this.runNext()
      }catch(K){
        this._active--,q.reject(K),this.runNext()
      }
    }
  }TYK.Semaphore=vYK
} /* confidence: 65% */

/* original: Lrz */ var composed_value=z56(); /* confidence: 30% */

/* original: LYK */ var named_entity=B((yYK)=>{
  Object.defineProperty(yYK,"__esModule",{
    value:!0
  });
  yYK.ReadableStreamMessageReader=yYK.AbstractMessageReader=yYK.MessageReader=void 0;
  var bs1=z56(),wh6=Oh6(),Cs1=Ah6(),hrz=Ss1(),VYK;
  (function(q){
    function K(_){
      let z=_;
      return z&&wh6.func(z.listen)&&wh6.func(z.dispose)&&wh6.func(z.onError)&&wh6.func(z.onClose)&&wh6.func(z.onPartialMessage)
    }q.is=K
  })(VYK||(yYK.MessageReader=VYK={
    
  }));
  class Is1{
    constructor(){
      this.errorEmitter=new Cs1.Emitter,this.closeEmitter=new Cs1.Emitter,this.partialMessageEmitter=new Cs1.Emitter
    }dispose(){
      this.errorEmitter.dispose(),this.closeEmitter.dispose()
    }get onError(){
      return this.errorEmitter.event
    }fireError(q){
      this.errorEmitter.fire(this.asError(q))
    }get onClose(){
      return this.closeEmitter.event
    }fireClose(){
      this.closeEmitter.fire(void 0)
    }get onPartialMessage(){
      return this.partialMessageEmitter.event
    }firePartialMessage(q){
      this.partialMessageEmitter.fire(q)
    }asError(q){
      if(q instanceof Error)return q;
      else return Error(`Reader received error. Reason: ${wh6.string(q.message)?q.message:"unknown"}`)
    }
  }yYK.AbstractMessageReader=Is1;
  var xs1;
  (function(q){
    function K(_){
      let z,Y,$,O=new Map,A,w=new Map;
      if(_===void 0||typeof _==="string")z=_??"utf-8";
      else{
        if(z=_.charset??"utf-8",_.contentDecoder!==void 0)$=_.contentDecoder,O.set($.name,$);
        if(_.contentDecoders!==void 0)for(let j of _.contentDecoders)O.set(j.name,j);
        if(_.contentTypeDecoder!==void 0)A=_.contentTypeDecoder,w.set(A.name,A);
        if(_.contentTypeDecoders!==void 0)for(let j of _.contentTypeDecoders)w.set(j.name,j)
      }if(A===void 0)A=(0,bs1.default)().applicationJson.decoder,w.set(A.name,A);
      return{
        charset:z,contentDecoder:$,contentDecoders:O,contentTypeDecoder:A,contentTypeDecoders:w
      }
    }q.fromOptions=K
  })(xs1||(xs1={
    
  }));
  class NYK extends Is1{
    constructor(q,K){
      super();
      this.readable=q,this.options=xs1.fromOptions(K),this.buffer=(0,bs1.default)().messageBuffer.create(this.options.charset),this._partialMessageTimeout=1e4,this.nextMessageLength=-1,this.messageToken=0,this.readSemaphore=new hrz.Semaphore(1)
    }set partialMessageTimeout(q){
      this._partialMessageTimeout=q
    }get partialMessageTimeout(){
      return this._partialMessageTimeout
    }listen(q){
      this.nextMessageLength=-1,this.messageToken=0,this.partialMessageTimer=void 0,this.callback=q;
      let K=this.readable.onData((_)=>{
        this.onData(_)
      });
      return this.readable.onError((_)=>this.fireError(_)),this.readable.onClose(()=>this.fireClose()),K
    }onData(q){
      try{
        this.buffer.append(q);
        while(!0){
          if(this.nextMessageLength===-1){
            let _=this.buffer.tryReadHeaders(!0);
            if(!_)return;
            let z=_.get("content-length");
            if(!z){
              this.fireError(Error(`Header must provide a Content-Length property.
${JSON.stringify(Object.fromEntries(_))}`));
              return
            }let Y=parseInt(z);
            if(isNaN(Y)){
              this.fireError(Error(`Content-Length value must be a number. Got ${z}`));
              return
            }this.nextMessageLength=Y
          }let K=this.buffer.tryReadBody(this.nextMessageLength);
          if(K===void 0){
            this.setPartialMessageTimer();
            return
          }this.clearPartialMessageTimer(),this.nextMessageLength=-1,this.readSemaphore.lock(async()=>{
            let _=this.options.contentDecoder!==void 0?await this.options.contentDecoder.decode(K):K,z=await this.options.contentTypeDecoder.decode(_,this.options);
            this.callback(z)
          }).catch((_)=>{
            this.fireError(_)
          })
        }
      }catch(K){
        this.fireError(K)
      }
    }clearPartialMessageTimer(){
      if(this.partialMessageTimer)this.partialMessageTimer.dispose(),this.partialMessageTimer=void 0
    }setPartialMessageTimer(){
      if(this.clearPartialMessageTimer(),this._partialMessageTimeout<=0)return;
      this.partialMessageTimer=(0,bs1.default)().timer.setTimeout((q,K)=>{
        if(this.partialMessageTimer=void 0,q===this.messageToken)this.firePartialMessage({
          messageToken:q,waitingTime:K
        }),this.setPartialMessageTimer()
      },this._partialMessageTimeout,this.messageToken,this._partialMessageTimeout)
    }
  }yYK.ReadableStreamMessageReader=NYK
} /* confidence: 70% */

/* original: bs1 */ var composed_value=z56(),wh6=Oh6(),Cs1=Ah6(),hrz=Ss1(),VYK; /* confidence: 30% */

/* original: uYK */ var options=B((xYK)=>{
  Object.defineProperty(xYK,"__esModule",{
    value:!0
  });
  xYK.WriteableStreamMessageWriter=xYK.AbstractMessageWriter=xYK.MessageWriter=void 0;
  var hYK=z56(),t88=Oh6(),Crz=Ss1(),RYK=Ah6(),brz="Content-Length: ",SYK=`\r
`,CYK;
  (function(q){
    function K(_){
      let z=_;
      return z&&t88.func(z.dispose)&&t88.func(z.onClose)&&t88.func(z.onError)&&t88.func(z.write)
    }q.is=K
  })(CYK||(xYK.MessageWriter=CYK={
    
  }));
  class ms1{
    constructor(){
      this.errorEmitter=new RYK.Emitter,this.closeEmitter=new RYK.Emitter
    }dispose(){
      this.errorEmitter.dispose(),this.closeEmitter.dispose()
    }get onError(){
      return this.errorEmitter.event
    }fireError(q,K,_){
      this.errorEmitter.fire([this.asError(q),K,_])
    }get onClose(){
      return this.closeEmitter.event
    }fireClose(){
      this.closeEmitter.fire(void 0)
    }asError(q){
      if(q instanceof Error)return q;
      else return Error(`Writer received error. Reason: ${t88.string(q.message)?q.message:"unknown"}`)
    }
  }xYK.AbstractMessageWriter=ms1;
  var us1;
  (function(q){
    function K(_){
      if(_===void 0||typeof _==="string")return{
        charset:_??"utf-8",contentTypeEncoder:(0,hYK.default)().applicationJson.encoder
      };
      else return{
        charset:_.charset??"utf-8",contentEncoder:_.contentEncoder,contentTypeEncoder:_.contentTypeEncoder??(0,hYK.default)().applicationJson.encoder
      }
    }q.fromOptions=K
  })(us1||(us1={
    
  }));
  class bYK extends ms1{
    constructor(q,K){
      super();
      this.writable=q,this.options=us1.fromOptions(K),this.errorCount=0,this.writeSemaphore=new Crz.Semaphore(1),this.writable.onError((_)=>this.fireError(_)),this.writable.onClose(()=>this.fireClose())
    }async write(q){
      return this.writeSemaphore.lock(async()=>{
        return this.options.contentTypeEncoder.encode(q,this.options).then((_)=>{
          if(this.options.contentEncoder!==void 0)return this.options.contentEncoder.encode(_);
          else return _
        }).then((_)=>{
          let z=[];
          return z.push(brz,_.byteLength.toString(),SYK),z.push(SYK),this.doWrite(q,z,_)
        },(_)=>{
          throw this.fireError(_),_
        })
      })
    }async doWrite(q,K,_){
      try{
        return await this.writable.write(K.join(""),"ascii"),this.writable.write(_)
      }catch(z){
        return this.handleError(z,q),Promise.reject(z)
      }
    }handleError(q,K){
      this.errorCount++,this.fireError(q,K,this.errorCount)
    }end(){
      this.writable.end()
    }
  }xYK.WriteableStreamMessageWriter=bYK
} /* confidence: 70% */

/* original: hYK */ var composed_value=z56(),t88=Oh6(),Crz=Ss1(),RYK=Ah6(),brz="Content-Length: ",SYK=`\r
`,CYK; /* confidence: 30% */

/* original: FYK */ var composed_value=z56(),nH=Oh6(),v3=Vs1(),UYK=ys1(),e88=Ah6(),ps1=Om8(),_18; /* confidence: 30% */

/* original: Xm8 */ var error_handler=B((vK)=>{
  Object.defineProperty(vK,"__esModule",{
    value:!0
  });
  vK.ProgressType=vK.ProgressToken=vK.createMessageConnection=vK.NullLogger=vK.ConnectionOptions=vK.ConnectionStrategy=vK.AbstractMessageBuffer=vK.WriteableStreamMessageWriter=vK.AbstractMessageWriter=vK.MessageWriter=vK.ReadableStreamMessageReader=vK.AbstractMessageReader=vK.MessageReader=vK.SharedArrayReceiverStrategy=vK.SharedArraySenderStrategy=vK.CancellationToken=vK.CancellationTokenSource=vK.Emitter=vK.Event=vK.Disposable=vK.LRUCache=vK.Touch=vK.LinkedMap=vK.ParameterStructures=vK.NotificationType9=vK.NotificationType8=vK.NotificationType7=vK.NotificationType6=vK.NotificationType5=vK.NotificationType4=vK.NotificationType3=vK.NotificationType2=vK.NotificationType1=vK.NotificationType0=vK.NotificationType=vK.ErrorCodes=vK.ResponseError=vK.RequestType9=vK.RequestType8=vK.RequestType7=vK.RequestType6=vK.RequestType5=vK.RequestType4=vK.RequestType3=vK.RequestType2=vK.RequestType1=vK.RequestType0=vK.RequestType=vK.Message=vK.RAL=void 0;
  vK.MessageStrategy=vK.CancellationStrategy=vK.CancellationSenderStrategy=vK.CancellationReceiverStrategy=vK.ConnectionError=vK.ConnectionErrors=vK.LogTraceNotification=vK.SetTraceNotification=vK.TraceFormat=vK.TraceValues=vK.Trace=void 0;
  var P2=Vs1();
  Object.defineProperty(vK,"Message",{
    enumerable:!0,get:function(){
      return P2.Message
    }
  });
  Object.defineProperty(vK,"RequestType",{
    enumerable:!0,get:function(){
      return P2.RequestType
    }
  });
  Object.defineProperty(vK,"RequestType0",{
    enumerable:!0,get:function(){
      return P2.RequestType0
    }
  });
  Object.defineProperty(vK,"RequestType1",{
    enumerable:!0,get:function(){
      return P2.RequestType1
    }
  });
  Object.defineProperty(vK,"RequestType2",{
    enumerable:!0,get:function(){
      return P2.RequestType2
    }
  });
  Object.defineProperty(vK,"RequestType3",{
    enumerable:!0,get:function(){
      return P2.RequestType3
    }
  });
  Object.defineProperty(vK,"RequestType4",{
    enumerable:!0,get:function(){
      return P2.RequestType4
    }
  });
  Object.defineProperty(vK,"RequestType5",{
    enumerable:!0,get:function(){
      return P2.RequestType5
    }
  });
  Object.defineProperty(vK,"RequestType6",{
    enumerable:!0,get:function(){
      return P2.RequestType6
    }
  });
  Object.defineProperty(vK,"RequestType7",{
    enumerable:!0,get:function(){
      return P2.RequestType7
    }
  });
  Object.defineProperty(vK,"RequestType8",{
    enumerable:!0,get:function(){
      return P2.RequestType8
    }
  });
  Object.defineProperty(vK,"RequestType9",{
    enumerable:!0,get:function(){
      return P2.RequestType9
    }
  });
  Object.defineProperty(vK,"ResponseError",{
    enumerable:!0,get:function(){
      return P2.ResponseError
    }
  });
  Object.defineProperty(vK,"ErrorCodes",{
    enumerable:!0,get:function(){
      return P2.ErrorCodes
    }
  });
  Object.defineProperty(vK,"NotificationType",{
    enumerable:!0,get:function(){
      return P2.NotificationType
    }
  });
  Object.defineProperty(vK,"NotificationType0",{
    enumerable:!0,get:function(){
      return P2.NotificationType0
    }
  });
  Object.defineProperty(vK,"NotificationType1",{
    enumerable:!0,get:function(){
      return P2.NotificationType1
    }
  });
  Object.defineProperty(vK,"NotificationType2",{
    enumerable:!0,get:function(){
      return P2.NotificationType2
    }
  });
  Object.defineProperty(vK,"NotificationType3",{
    enumerable:!0,get:function(){
      return P2.NotificationType3
    }
  });
  Object.defineProperty(vK,"NotificationType4",{
    enumerable:!0,get:function(){
      return P2.NotificationType4
    }
  });
  Object.defineProperty(vK,"NotificationType5",{
    enumerable:!0,get:function(){
      return P2.NotificationType5
    }
  });
  Object.defineProperty(vK,"NotificationType6",{
    enumerable:!0,get:function(){
      return P2.NotificationType6
    }
  });
  Object.defineProperty(vK,"NotificationType7",{
    enumerable:!0,get:function(){
      return P2.NotificationType7
    }
  });
  Object.defineProperty(vK,"NotificationType8",{
    enumerable:!0,get:function(){
      return P2.NotificationType8
    }
  });
  Object.defineProperty(vK,"NotificationType9",{
    enumerable:!0,get:function(){
      return P2.NotificationType9
    }
  });
  Object.defineProperty(vK,"ParameterStructures",{
    enumerable:!0,get:function(){
      return P2.ParameterStructures
    }
  });
  var ds1=ys1();
  Object.defineProperty(vK,"LinkedMap",{
    enumerable:!0,get:function(){
      return ds1.LinkedMap
    }
  });
  Object.defineProperty(vK,"LRUCache",{
    enumerable:!0,get:function(){
      return ds1.LRUCache
    }
  });
  Object.defineProperty(vK,"Touch",{
    enumerable:!0,get:function(){
      return ds1.Touch
    }
  });
  var qoz=YYK();
  Object.defineProperty(vK,"Disposable",{
    enumerable:!0,get:function(){
      return qoz.Disposable
    }
  });
  var sYK=Ah6();
  Object.defineProperty(vK,"Event",{
    enumerable:!0,get:function(){
      return sYK.Event
    }
  });
  Object.defineProperty(vK,"Emitter",{
    enumerable:!0,get:function(){
      return sYK.Emitter
    }
  });
  var tYK=Om8();
  Object.defineProperty(vK,"CancellationTokenSource",{
    enumerable:!0,get:function(){
      return tYK.CancellationTokenSource
    }
  });
  Object.defineProperty(vK,"CancellationToken",{
    enumerable:!0,get:function(){
      return tYK.CancellationToken
    }
  });
  var eYK=GYK();
  Object.defineProperty(vK,"SharedArraySenderStrategy",{
    enumerable:!0,get:function(){
      return eYK.SharedArraySenderStrategy
    }
  });
  Object.defineProperty(vK,"SharedArrayReceiverStrategy",{
    enumerable:!0,get:function(){
      return eYK.SharedArrayReceiverStrategy
    }
  });
  var cs1=LYK();
  Object.defineProperty(vK,"MessageReader",{
    enumerable:!0,get:function(){
      return cs1.MessageReader
    }
  });
  Object.defineProperty(vK,"AbstractMessageReader",{
    enumerable:!0,get:function(){
      return cs1.AbstractMessageReader
    }
  });
  Object.defineProperty(vK,"ReadableStreamMessageReader",{
    enumerable:!0,get:function(){
      return cs1.ReadableStreamMessageReader
    }
  });
  var ls1=uYK();
  Object.defineProperty(vK,"MessageWriter",{
    enumerable:!0,get:function(){
      return ls1.MessageWriter
    }
  });
  Object.defineProperty(vK,"AbstractMessageWriter",{
    enumerable:!0,get:function(){
      return ls1.AbstractMessageWriter
    }
  });
  Object.defineProperty(vK,"WriteableStreamMessageWriter",{
    enumerable:!0,get:function(){
      return ls1.WriteableStreamMessageWriter
    }
  });
  var Koz=gYK();
  Object.defineProperty(vK,"AbstractMessageBuffer",{
    enumerable:!0,get:function(){
      return Koz.AbstractMessageBuffer
    }
  });
  var j0=aYK();
  Object.defineProperty(vK,"ConnectionStrategy",{
    enumerable:!0,get:function(){
      return j0.ConnectionStrategy
    }
  });
  Object.defineProperty(vK,"ConnectionOptions",{
    enumerable:!0,get:function(){
      return j0.ConnectionOptions
    }
  });
  Object.defineProperty(vK,"NullLogger",{
    enumerable:!0,get:function(){
      return j0.NullLogger
    }
  });
  Object.defineProperty(vK,"createMessageConnection",{
    enumerable:!0,get:function(){
      return j0.createMessageConnection
    }
  });
  Object.defineProperty(vK,"ProgressToken",{
    enumerable:!0,get:function(){
      return j0.ProgressToken
    }
  });
  Object.defineProperty(vK,"ProgressType",{
    enumerable:!0,get:function(){
      return j0.ProgressType
    }
  });
  Object.defineProperty(vK,"Trace",{
    enumerable:!0,get:function(){
      return j0.Trace
    }
  });
  Object.defineProperty(vK,"TraceValues",{
    enumerable:!0,get:function(){
      return j0.TraceValues
    }
  });
  Object.defineProperty(vK,"TraceFormat",{
    enumerable:!0,get:function(){
      return j0.TraceFormat
    }
  });
  Object.defineProperty(vK,"SetTraceNotification",{
    enumerable:!0,get:function(){
      return j0.SetTraceNotification
    }
  });
  Object.defineProperty(vK,"LogTraceNotification",{
    enumerable:!0,get:function(){
      return j0.LogTraceNotification
    }
  });
  Object.defineProperty(vK,"ConnectionErrors",{
    enumerable:!0,get:function(){
      return j0.ConnectionErrors
    }
  });
  Object.defineProperty(vK,"ConnectionError",{
    enumerable:!0,get:function(){
      return j0.ConnectionError
    }
  });
  Object.defineProperty(vK,"CancellationReceiverStrategy",{
    enumerable:!0,get:function(){
      return j0.CancellationReceiverStrategy
    }
  });
  Object.defineProperty(vK,"CancellationSenderStrategy",{
    enumerable:!0,get:function(){
      return j0.CancellationSenderStrategy
    }
  });
  Object.defineProperty(vK,"CancellationStrategy",{
    enumerable:!0,get:function(){
      return j0.CancellationStrategy
    }
  });
  Object.defineProperty(vK,"MessageStrategy",{
    enumerable:!0,get:function(){
      return j0.MessageStrategy
    }
  });
  var _oz=z56();
  vK.RAL=_oz.default
} /* confidence: 95% */

/* original: tYK */ var composed_value=Om8(); /* confidence: 30% */

/* original: cs1 */ var composed_value=LYK(); /* confidence: 30% */

/* original: ls1 */ var composed_value=uYK(); /* confidence: 30% */

/* original: _oz */ var composed_value=z56(); /* confidence: 30% */

/* original: q$K */ var composed_value=U6("util"),sa=Xm8(); /* confidence: 30% */

/* original: O$K */ var path_os_crypto_net=U6("path"),joz=U6("os"),Hoz=U6("crypto"),Wm8=U6("net"),vh=Xm8(); /* confidence: 65% */

/* original: Ls1 */ function action_creator(){
  if(Es1===void 0)throw Error("No runtime abstraction layer installed");
  return Es1
} /* confidence: 95% */

/* original: z6 */ function skip_immediate_prompt(){
  if(P||W.size===0)return;
  P=(0,FYK.default)().timer.setImmediate(()=>{
    P=void 0,J6()
  })
} /* confidence: 65% */

/* original: Rs1 */ class entity_class{
  constructor(){
    this._isCancelled=!1
  }cancel(){
    if(!this._isCancelled){
      if(this._isCancelled=!0,this._emitter)this._emitter.fire(void 0),this.dispose()
    }
  }get isCancellationRequested(){
    return this._isCancelled
  }get onCancellationRequested(){
    if(this._isCancelled)return Vrz;
    if(!this._emitter)this._emitter=new hs1.Emitter;
    return this._emitter.event
  }dispose(){
    if(this._emitter)this._emitter.dispose(),this._emitter=void 0
  }
} /* confidence: 45% */

