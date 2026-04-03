// Module: function

/* original: $z1 */ var state_manager=B((zc$,vU7)=>{
  vU7.exports={
    kState:Symbol("FileReader state"),kResult:Symbol("FileReader result"),kError:Symbol("FileReader error"),kLastProgressEventFired:Symbol("FileReader last progress event fired timestamp"),kEvents:Symbol("FileReader events"),kAborted:Symbol("FileReader aborted")
  }
} /* confidence: 95% */

/* original: mU7 */ var empty_FileReaderreadAsArrayBuf=B((Ac$,uU7)=>{
  var{
    staticPropertyDescriptors:If6,readOperation:B28,fireAProgressEvent:xU7
  }=bU7(),{
    kState:Sz6,kError:IU7,kResult:g28,kEvents:e$,kAborted:vl5
  }=$z1(),{
    webidl:BA
  }=nf(),{
    kEnumerableProperty:CV
  }=Az();
  class wA extends EventTarget{
    constructor(){
      super();
      this[Sz6]="empty",this[g28]=null,this[IU7]=null,this[e$]={
        loadend:null,error:null,abort:null,load:null,progress:null,loadstart:null
      }
    }readAsArrayBuffer(q){
      BA.brandCheck(this,wA),BA.argumentLengthCheck(arguments,1,"FileReader.readAsArrayBuffer"),q=BA.converters.Blob(q,{
        strict:!1
      }),B28(this,q,"ArrayBuffer")
    }readAsBinaryString(q){
      BA.brandCheck(this,wA),BA.argumentLengthCheck(arguments,1,"FileReader.readAsBinaryString"),q=BA.converters.Blob(q,{
        strict:!1
      }),B28(this,q,"BinaryString")
    }readAsText(q,K=void 0){
      if(BA.brandCheck(this,wA),BA.argumentLengthCheck(arguments,1,"FileReader.readAsText"),q=BA.converters.Blob(q,{
        strict:!1
      }),K!==void 0)K=BA.converters.DOMString(K,"FileReader.readAsText","encoding");
      B28(this,q,"Text",K)
    }readAsDataURL(q){
      BA.brandCheck(this,wA),BA.argumentLengthCheck(arguments,1,"FileReader.readAsDataURL"),q=BA.converters.Blob(q,{
        strict:!1
      }),B28(this,q,"DataURL")
    }abort(){
      if(this[Sz6]==="empty"||this[Sz6]==="done"){
        this[g28]=null;
        return
      }if(this[Sz6]==="loading")this[Sz6]="done",this[g28]=null;
      if(this[vl5]=!0,xU7("abort",this),this[Sz6]!=="loading")xU7("loadend",this)
    }get readyState(){
      switch(BA.brandCheck(this,wA),this[Sz6]){
        case"empty":return this.EMPTY;
        case"loading":return this.LOADING;
        case"done":return this.DONE
      }
    }get result(){
      return BA.brandCheck(this,wA),this[g28]
    }get error(){
      return BA.brandCheck(this,wA),this[IU7]
    }get onloadend(){
      return BA.brandCheck(this,wA),this[e$].loadend
    }set onloadend(q){
      if(BA.brandCheck(this,wA),this[e$].loadend)this.removeEventListener("loadend",this[e$].loadend);
      if(typeof q==="function")this[e$].loadend=q,this.addEventListener("loadend",q);
      else this[e$].loadend=null
    }get onerror(){
      return BA.brandCheck(this,wA),this[e$].error
    }set onerror(q){
      if(BA.brandCheck(this,wA),this[e$].error)this.removeEventListener("error",this[e$].error);
      if(typeof q==="function")this[e$].error=q,this.addEventListener("error",q);
      else this[e$].error=null
    }get onloadstart(){
      return BA.brandCheck(this,wA),this[e$].loadstart
    }set onloadstart(q){
      if(BA.brandCheck(this,wA),this[e$].loadstart)this.removeEventListener("loadstart",this[e$].loadstart);
      if(typeof q==="function")this[e$].loadstart=q,this.addEventListener("loadstart",q);
      else this[e$].loadstart=null
    }get onprogress(){
      return BA.brandCheck(this,wA),this[e$].progress
    }set onprogress(q){
      if(BA.brandCheck(this,wA),this[e$].progress)this.removeEventListener("progress",this[e$].progress);
      if(typeof q==="function")this[e$].progress=q,this.addEventListener("progress",q);
      else this[e$].progress=null
    }get onload(){
      return BA.brandCheck(this,wA),this[e$].load
    }set onload(q){
      if(BA.brandCheck(this,wA),this[e$].load)this.removeEventListener("load",this[e$].load);
      if(typeof q==="function")this[e$].load=q,this.addEventListener("load",q);
      else this[e$].load=null
    }get onabort(){
      return BA.brandCheck(this,wA),this[e$].abort
    }set onabort(q){
      if(BA.brandCheck(this,wA),this[e$].abort)this.removeEventListener("abort",this[e$].abort);
      if(typeof q==="function")this[e$].abort=q,this.addEventListener("abort",q);
      else this[e$].abort=null
    }
  }wA.EMPTY=wA.prototype.EMPTY=0;
  wA.LOADING=wA.prototype.LOADING=1;
  wA.DONE=wA.prototype.DONE=2;
  Object.defineProperties(wA.prototype,{
    EMPTY:If6,LOADING:If6,DONE:If6,readAsArrayBuffer:CV,readAsBinaryString:CV,readAsText:CV,readAsDataURL:CV,abort:CV,readyState:CV,result:CV,error:CV,onloadstart:CV,onprogress:CV,onload:CV,onabort:CV,onerror:CV,onloadend:CV,[Symbol.toStringTag]:{
      value:"FileReader",writable:!1,enumerable:!1,configurable:!0
    }
  });
  Object.defineProperties(wA,{
    EMPTY:If6,LOADING:If6,DONE:If6
  });
  uU7.exports={
    FileReader:wA
  }
} /* confidence: 65% */

