// Module: EventStreamMarshaller

/* original: Ezq */ var stream_error_data_end_function=B((n99)=>{
  var Q99=yzq(),d99=U6("stream");
  async function*c99(q){
    let K=!1,_=!1,z=[];
    q.on("error",(Y)=>{
      if(!K)K=!0;
      if(Y)throw Y
    }),q.on("data",(Y)=>{
      z.push(Y)
    }),q.on("end",()=>{
      K=!0
    });
    while(!_){
      let Y=await new Promise(($)=>setTimeout(()=>$(z.shift()),0));
      if(Y)yield Y;
      _=K&&z.length===0
    }
  }class CM1{
    universalMarshaller;
    constructor({
      utf8Encoder:q,utf8Decoder:K
    }){
      this.universalMarshaller=new Q99.EventStreamMarshaller({
        utf8Decoder:K,utf8Encoder:q
      })
    }deserialize(q,K){
      let _=typeof q[Symbol.asyncIterator]==="function"?q:c99(q);
      return this.universalMarshaller.deserialize(_,K)
    }serialize(q,K){
      return d99.Readable.from(this.universalMarshaller.serialize(q,K))
    }
  }var l99=(q)=>new CM1(q);
  n99.EventStreamMarshaller=CM1;
  n99.eventStreamSerdeProvider=l99
} /* confidence: 65% */

/* original: l99 */ var composed_value=(q)=>new CM1(q); /* confidence: 30% */

/* original: CM1 */ class entity_class{
  universalMarshaller;
   /* confidence: 45% */

