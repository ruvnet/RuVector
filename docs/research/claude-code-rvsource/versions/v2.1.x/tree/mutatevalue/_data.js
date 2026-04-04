// Module: _data

/* original: G67 */ var ConfigManager=B((WZw,ajK)=>{
  ajK.exports=Z67;
  var I7Y=X0(),ojK=Q18();
  function Z67(q,K){
    ojK.call(this),this.nodeType=I7Y.COMMENT_NODE,this.ownerDocument=q,this._data=K
  }var c18={
    get:function(){
      return this._data
    },set:function(q){
      if(q===null||q===void 0)q="";
      else q=String(q);
      if(this._data=q,this.rooted)this.ownerDocument.mutateValue(this)
    }
  };
  Z67.prototype=Object.create(ojK.prototype,{
    nodeName:{
      value:"#comment"
    },nodeValue:c18,textContent:c18,innerText:c18,data:{
      get:c18.get,set:function(q){
        c18.set.call(this,q===null?"":String(q))
      }
    },clone:{
      value:function(){
        return new Z67(this.ownerDocument,this._data)
      }
    }
  })
} /* confidence: 34% */

/* original: c18 */ var c18={
  get:function(){
    return this._data
  }

