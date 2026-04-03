// Module: Unreachable

/* original: NI7 */ var value_holder=B((iQ$,VI7)=>{
  var{
    wellknownHeaderNames:TI7,headerNameLowerCasedRecord:nI5
  }=Vw8();
  class aD6{
    value=null;
    left=null;
    middle=null;
    right=null;
    code;
    constructor(q,K,_){
      if(_===void 0||_>=q.length)throw TypeError("Unreachable");
      if((this.code=q.charCodeAt(_))>127)throw TypeError("key must be ascii string");
      if(q.length!==++_)this.middle=new aD6(q,K,_);
      else this.value=K
    }add(q,K){
      let _=q.length;
      if(_===0)throw TypeError("Unreachable");
      let z=0,Y=this;
      while(!0){
        let $=q.charCodeAt(z);
        if($>127)throw TypeError("key must be ascii string");
        if(Y.code===$)if(_===++z){
          Y.value=K;
          break
        }else if(Y.middle!==null)Y=Y.middle;
        else{
          Y.middle=new aD6(q,K,z);
          break
        }else if(Y.code<$)if(Y.left!==null)Y=Y.left;
        else{
          Y.left=new aD6(q,K,z);
          break
        }else if(Y.right!==null)Y=Y.right;
        else{
          Y.right=new aD6(q,K,z);
          break
        }
      }
    }search(q){
      let K=q.length,_=0,z=this;
      while(z!==null&&_<K){
        let Y=q[_];
        if(Y<=90&&Y>=65)Y|=32;
        while(z!==null){
          if(Y===z.code){
            if(K===++_)return z;
            z=z.middle;
            break
          }z=z.code<Y?z.left:z.right
        }
      }return null
    }
  }class q91{
    node=null;
    insert(q,K){
      if(this.node===null)this.node=new aD6(q,K,0);
      else this.node.add(q,K)
    }lookup(q){
      return this.node?.search(q)?.value??null
    }
  }var kI7=new q91;
  for(let q=0;
  q<TI7.length;
  ++q){
    let K=nI5[TI7[q]];
    kI7.insert(K,K)
  }VI7.exports={
    TernarySearchTree:q91,tree:kI7
  }
} /* confidence: 70% */

/* original: kI7 */ var composed_value=new q91; /* confidence: 30% */

/* original: q91 */ class entity_class{
  node=null;
   /* confidence: 45% */

