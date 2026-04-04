// Module: assign

/* original: Zj */ var schema_def=B((J13)=>{
  var e83=OF6(),rn7=ZG(),yj8=(q)=>{
    if(typeof q==="function")return q();
    return q
  },mY1=(q,K,_,z,Y)=>({
    name:K,namespace:q,traits:_,input:z,output:Y
  }),q13=(q)=>(K,_)=>async(z)=>{
    let{
      response:Y
    }=await K(z),{
      operationSchema:$
    }=rn7.getSmithyContext(_),[,O,A,w,j,H]=$??[];
    try{
      let J=await q.protocol.deserializeResponse(mY1(O,A,w,j,H),{
        ...q,..._
      },Y);
      return{
        response:Y,output:J
      }
    }catch(J){
      if(Object.defineProperty(J,"$response",{
        value:Y,enumerable:!1,writable:!1,configurable:!1
      }),!("$metadata"in J)){
        try{
          J.message+=`
  Deserialization error: to see the raw response, inspect the hidden field {error}.$response on this object.`
        }catch(X){
          if(!_.logger||_.logger?.constructor?.name==="NoOpLogger")console.warn("Deserialization error: to see the raw response, inspect the hidden field {error}.$response on this object.");
          else _.logger?.warn?.("Deserialization error: to see the raw response, inspect the hidden field {error}.$response on this object.")
        }if(typeof J.$responseBodyText<"u"){
          if(J.$response)J.$response.body=J.$responseBodyText
        }try{
          if(e83.HttpResponse.isInstance(Y)){
            let{
              headers:X={
                
              }
            }=Y,P=Object.entries(X);
            J.$metadata={
              httpStatusCode:Y.statusCode,requestId:IY1(/^x-[\w-]+-request-?id$/,P),extendedRequestId:IY1(/^x-[\w-]+-id-2$/,P),cfId:IY1(/^x-[\w-]+-cf-id$/,P)
            }
          }
        }catch(X){
          
        }
      }throw J
    }
  },IY1=(q,K)=>{
    return(K.find(([_])=>{
      return _.match(q)
    })||[void 0,void 0])[1]
  },K13=(q)=>(K,_)=>async(z)=>{
    let{
      operationSchema:Y
    }=rn7.getSmithyContext(_),[,$,O,A,w,j]=Y??[],H=_.endpointV2?.url&&q.urlParser?async()=>q.urlParser(_.endpointV2.url):q.endpoint,J=await q.protocol.serializeRequest(mY1($,O,A,w,j),z.input,{
      ...q,..._,endpoint:H
    });
    return K({
      ...z,request:J
    })
  },on7={
    name:"deserializerMiddleware",step:"deserialize",tags:["DESERIALIZER"],override:!0
  },an7={
    name:"serializerMiddleware",step:"serialize",tags:["SERIALIZER"],override:!0
  };
  function _13(q){
    return{
      applyToStack:(K)=>{
        K.add(K13(q),an7),K.add(q13(q),on7),q.protocol.setSerdeContext(q)
      }
    }
  }class BV{
    name;
    namespace;
    traits;
    static assign(q,K){
      return Object.assign(q,K)
    }static[Symbol.hasInstance](q){
      let K=this.prototype.isPrototypeOf(q);
      if(!K&&typeof q==="object"&&q!==null)return q.symbol===this.symbol;
      return K
    }getName(){
      return this.namespace+"#"+this.name
    }
  }class Ej8 extends BV{
    static symbol=Symbol.for("@smithy/lis");
    name;
    traits;
    valueSchema;
    symbol=Ej8.symbol
  }var z13=(q,K,_,z)=>BV.assign(new Ej8,{
    name:K,namespace:q,traits:_,valueSchema:z
  });
  class Lj8 extends BV{
    static symbol=Symbol.for("@smithy/map");
    name;
    traits;
    keySchema;
    valueSchema;
    symbol=Lj8.symbol
  }var Y13=(q,K,_,z,Y)=>BV.assign(new Lj8,{
    name:K,namespace:q,traits:_,keySchema:z,valueSchema:Y
  });
  class hj8 extends BV{
    static symbol=Symbol.for("@smithy/ope");
    name;
    traits;
    input;
    output;
    symbol=hj8.symbol
  }var $13=(q,K,_,z,Y)=>BV.assign(new hj8,{
    name:K,namespace:q,traits:_,input:z,output:Y
  });
  class HF6 extends BV{
    static symbol=Symbol.for("@smithy/str");
    name;
    traits;
    memberNames;
    memberList;
    symbol=HF6.symbol
  }var O13=(q,K,_,z,Y)=>BV.assign(new HF6,{
    name:K,namespace:q,traits:_,memberNames:z,memberList:Y
  });
  class Rj8 extends HF6{
    static symbol=Symbol.for("@smithy/err");
    ctor;
    symbol=Rj8.symbol
  }var A13=(q,K,_,z,Y,$)=>BV.assign(new Rj8,{
    name:K,namespace:q,traits:_,memberNames:z,memberList:Y,ctor:null
  });
  function jF6(q){
    if(typeof q==="object")return q;
    q=q|0;
    let K={
      
    },_=0;
    for(let z of["httpLabel","idempotent","idempotencyToken","sensitive","httpPayload","httpResponseCode","httpQueryParams"])if((q>>_++&1)===1)K[z]=1;
    return K
  }class En{
    ref;
    memberName;
    static symbol=Symbol.for("@smithy/nor");
    symbol=En.symbol;
    name;
    schema;
    _isMemberSchema;
    traits;
    memberTraits;
    normalizedTraits;
    constructor(q,K){
      this.ref=q,this.memberName=K;
      let _=[],z=q,Y=q;
      this._isMemberSchema=!1;
      while(uY1(z))_.push(z[1]),z=z[0],Y=yj8(z),this._isMemberSchema=!0;
      if(_.length>0){
        this.memberTraits={
          
        };
        for(let $=_.length-1;
        $>=0;
        --$){
          let O=_[$];
          Object.assign(this.memberTraits,jF6(O))
        }
      }else this.memberTraits=0;
      if(Y instanceof En){
        let $=this.memberTraits;
        Object.assign(this,Y),this.memberTraits=Object.assign({
          
        },$,Y.getMemberTraits(),this.getMemberTraits()),this.normalizedTraits=void 0,this.memberName=K??Y.memberName;
        return
      }if(this.schema=yj8(Y),sn7(this.schema))this.name=`${this.schema[1]}#${this.schema[2]}`,this.traits=this.schema[3];
      else this.name=this.memberName??String(Y),this.traits=0;
      if(this._isMemberSchema&&!K)throw Error(`@smithy/core/schema - NormalizedSchema member init ${this.getName(!0)} missing member name.`)
    }static[Symbol.hasInstance](q){
      let K=this.prototype.isPrototypeOf(q);
      if(!K&&typeof q==="object"&&q!==null)return q.symbol===this.symbol;
      return K
    }static of(q){
      let K=yj8(q);
      if(K instanceof En)return K;
      if(uY1(K)){
        let[_,z]=K;
        if(_ instanceof En)return Object.assign(_.getMergedTraits(),jF6(z)),_;
        throw Error(`@smithy/core/schema - may not init unwrapped member schema=${JSON.stringify(q,null,2)}.`)
      }return new En(K)
    }getSchema(){
      let q=this.schema;
      if(q[0]===0)return q[4];
      return q
    }getName(q=!1){
      let{
        name:K
      }=this;
      return!q&&K&&K.includes("#")?K.split("#")[1]:K||void 0
    }getMemberName(){
      return this.memberName
    }isMemberSchema(){
      return this._isMemberSchema
    }isListSchema(){
      let q=this.getSchema();
      return typeof q==="number"?q>=64&&q<128:q[0]===1
    }isMapSchema(){
      let q=this.getSchema();
      return typeof q==="number"?q>=128&&q<=255:q[0]===2
    }isStructSchema(){
      let q=this.getSchema();
      return q[0]===3||q[0]===-3
    }isBlobSchema(){
      let q=this.getSchema();
      return q===21||q===42
    }isTimestampSchema(){
      let q=this.getSchema();
      return typeof q==="number"&&q>=4&&q<=7
    }isUnitSchema(){
      return this.getSchema()==="unit"
    }isDocumentSchema(){
      return this.getSchema()===15
    }isStringSchema(){
      return this.getSchema()===0
    }isBooleanSchema(){
      return this.getSchema()===2
    }isNumericSchema(){
      return this.getSchema()===1
    }isBigIntegerSchema(){
      return this.getSchema()===17
    }isBigDecimalSchema(){
      return this.getSchema()===19
    }isStreaming(){
      let{
        streaming:q
      }=this.getMergedTraits();
      return!!q||this.getSchema()===42
    }isIdempotencyToken(){
      let q=(Y)=>(Y&4)===4||!!Y?.idempotencyToken,{
        normalizedTraits:K,traits:_,memberTraits:z
      }=this;
      return q(K)||q(_)||q(z)
    }getMergedTraits(){
      return this.normalizedTraits??(this.normalizedTraits={
        ...this.getOwnTraits(),...this.getMemberTraits()
      })
    }getMemberTraits(){
      return jF6(this.memberTraits)
    }getOwnTraits(){
      return jF6(this.traits)
    }getKeySchema(){
      let[q,K]=[this.isDocumentSchema(),this.isMapSchema()];
      if(!q&&!K)throw Error(`@smithy/core/schema - cannot get key for non-map: ${this.getName(!0)}`);
      let _=this.getSchema(),z=q?15:_[4]??0;
      return wF6([z,0],"key")
    }getValueSchema(){
      let q=this.getSchema(),[K,_,z]=[this.isDocumentSchema(),this.isMapSchema(),this.isListSchema()],Y=typeof q==="number"?63&q:q&&typeof q==="object"&&(_||z)?q[3+q[0]]:K?15:void 0;
      if(Y!=null)return wF6([Y,0],_?"value":"member");
      throw Error(`@smithy/core/schema - ${this.getName(!0)} has no value member.`)
    }getMemberSchema(q){
      let K=this.getSchema();
      if(this.isStructSchema()&&K[4].includes(q)){
        let _=K[4].indexOf(q),z=K[5][_];
        return wF6(uY1(z)?z:[z,0],q)
      }if(this.isDocumentSchema())return wF6([15,0],q);
      throw Error(`@smithy/core/schema - ${this.getName(!0)} has no no member=${q}.`)
    }getMemberSchemas(){
      let q={
        
      };
      try{
        for(let[K,_]of this.structIterator())q[K]=_
      }catch(K){
        
      }return q
    }getEventStreamMember(){
      if(this.isStructSchema()){
        for(let[q,K]of this.structIterator())if(K.isStreaming()&&K.isStructSchema())return q
      }return""
    }*structIterator(){
      if(this.isUnitSchema())return;
      if(!this.isStructSchema())throw Error("@smithy/core/schema - cannot iterate non-struct schema.");
      let q=this.getSchema();
      for(let K=0;
      K<q[4].length;
      ++K)yield[q[4][K],wF6([q[5][K],0],q[4][K])]
    }
  }function wF6(q,K){
    if(q instanceof En)return Object.assign(q,{
      memberName:K,_isMemberSchema:!0
    });
    return new En(q,K)
  }var uY1=(q)=>Array.isArray(q)&&q.length===2,sn7=(q)=>Array.isArray(q)&&q.length>=5;
  class JF6 extends BV{
    static symbol=Symbol.for("@smithy/sim");
    name;
    schemaRef;
    traits;
    symbol=JF6.symbol
  }var w13=(q,K,_,z)=>BV.assign(new JF6,{
    name:K,namespace:q,traits:z,schemaRef:_
  }),j13=(q,K,_,z)=>BV.assign(new JF6,{
    name:K,namespace:q,traits:_,schemaRef:z
  }),H13={
    BLOB:21,STREAMING_BLOB:42,BOOLEAN:2,STRING:0,NUMERIC:1,BIG_INTEGER:17,BIG_DECIMAL:19,DOCUMENT:15,TIMESTAMP_DEFAULT:4,TIMESTAMP_DATE_TIME:5,TIMESTAMP_HTTP_DATE:6,TIMESTAMP_EPOCH_SECONDS:7,LIST_MODIFIER:64,MAP_MODIFIER:128
  };
  class yn{
    namespace;
    schemas;
    exceptions;
    static registries=new Map;
    constructor(q,K=new Map,_=new Map){
      this.namespace=q,this.schemas=K,this.exceptions=_
    }static for(q){
      if(!yn.registries.has(q))yn.registries.set(q,new yn(q));
      return yn.registries.get(q)
    }register(q,K){
      let _=this.normalizeShapeId(q);
      yn.for(_.split("#")[0]).schemas.set(_,K)
    }getSchema(q){
      let K=this.normalizeShapeId(q);
      if(!this.schemas.has(K))throw Error(`@smithy/core/schema - schema not found for ${K}`);
      return this.schemas.get(K)
    }registerError(q,K){
      let _=q,z=yn.for(_[1]);
      z.schemas.set(_[1]+"#"+_[2],_),z.exceptions.set(_,K)
    }getErrorCtor(q){
      let K=q;
      return yn.for(K[1]).exceptions.get(K)
    }getBaseException(){
      for(let q of this.exceptions.keys())if(Array.isArray(q)){
        let[,K,_]=q,z=K+"#"+_;
        if(z.startsWith("smithy.ts.sdk.synthetic.")&&z.endsWith("ServiceException"))return q
      }return
    }find(q){
      return[...this.schemas.values()].find(q)
    }clear(){
      this.schemas.clear(),this.exceptions.clear()
    }normalizeShapeId(q){
      if(q.includes("#"))return q;
      return this.namespace+"#"+q
    }
  }J13.ErrorSchema=Rj8;
  J13.ListSchema=Ej8;
  J13.MapSchema=Lj8;
  J13.NormalizedSchema=En;
  J13.OperationSchema=hj8;
  J13.SCHEMA=H13;
  J13.Schema=BV;
  J13.SimpleSchema=JF6;
  J13.StructureSchema=HF6;
  J13.TypeRegistry=yn;
  J13.deref=yj8;
  J13.deserializerMiddlewareOption=on7;
  J13.error=A13;
  J13.getSchemaSerdePlugin=_13;
  J13.isStaticSchema=sn7;
  J13.list=z13;
  J13.map=Y13;
  J13.op=$13;
  J13.operation=mY1;
  J13.serializerMiddlewareOption=an7;
  J13.sim=w13;
  J13.simAdapter=j13;
  J13.struct=O13;
  J13.translateTraits=jF6
} /* confidence: 95% */

/* original: z13 */ var composed_value=(q,K,_,z)=>BV.assign(new Ej8,{
  name:K,namespace:q,traits:_,valueSchema:z
} /* confidence: 30% */

/* original: Y13 */ var composed_value=(q,K,_,z,Y)=>BV.assign(new Lj8,{
  name:K,namespace:q,traits:_,keySchema:z,valueSchema:Y
} /* confidence: 30% */

/* original: $13 */ var composed_value=(q,K,_,z,Y)=>BV.assign(new hj8,{
  name:K,namespace:q,traits:_,input:z,output:Y
} /* confidence: 30% */

/* original: O13 */ var composed_value=(q,K,_,z,Y)=>BV.assign(new HF6,{
  name:K,namespace:q,traits:_,memberNames:z,memberList:Y
} /* confidence: 30% */

/* original: A13 */ var composed_value=(q,K,_,z,Y,$)=>BV.assign(new Rj8,{
  name:K,namespace:q,traits:_,memberNames:z,memberList:Y,ctor:null
} /* confidence: 30% */

/* original: w13 */ var composed_value=(q,K,_,z)=>BV.assign(new JF6,{
  name:K,namespace:q,traits:z,schemaRef:_
} /* confidence: 30% */

/* original: BV */ class entity_class{
  name;
   /* confidence: 45% */

