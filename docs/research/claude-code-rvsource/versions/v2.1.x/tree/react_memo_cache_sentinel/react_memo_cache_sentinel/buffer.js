// Module: buffer

/* original: uP */ var uP=null,Zg=null,lw8=0,Gg=null,zg5=0,$g6=1,Pf6=2|$g6,iw8=4|$g6,U91=8|zg5;

/* original: _g5 */ function helper_fn(){
  let q=process.env.JEST_WORKER_ID?k91():void 0,K;
  try{
    K=await WebAssembly.compile(Su7())
  }catch(_){
    K=await WebAssembly.compile(q||k91())
  }return await WebAssembly.instantiate(K,{
    env:{
      wasm_on_url:(_,z,Y)=>{
        return 0
      },wasm_on_status:(_,z,Y)=>{
        Y3(uP.ptr===_);
        let $=z-Gg+Zg.byteOffset;
        return uP.onStatus(new dw8(Zg.buffer,$,Y))||0
      },wasm_on_message_begin:(_)=>{
        return Y3(uP.ptr===_),uP.onMessageBegin()||0
      },wasm_on_header_field:(_,z,Y)=>{
        Y3(uP.ptr===_);
        let $=z-Gg+Zg.byteOffset;
        return uP.onHeaderField(new dw8(Zg.buffer,$,Y))||0
      },wasm_on_header_value:(_,z,Y)=>{
        Y3(uP.ptr===_);
        let $=z-Gg+Zg.byteOffset;
        return uP.onHeaderValue(new dw8(Zg.buffer,$,Y))||0
      },wasm_on_headers_complete:(_,z,Y,$)=>{
        return Y3(uP.ptr===_),uP.onHeadersComplete(z,Boolean(Y),Boolean($))||0
      },wasm_on_body:(_,z,Y)=>{
        Y3(uP.ptr===_);
        let $=z-Gg+Zg.byteOffset;
        return uP.onBody(new dw8(Zg.buffer,$,Y))||0
      },wasm_on_message_complete:(_)=>{
        return Y3(uP.ptr===_),uP.onMessageComplete()||0
      }
    }
  })
} /* confidence: 35% */

