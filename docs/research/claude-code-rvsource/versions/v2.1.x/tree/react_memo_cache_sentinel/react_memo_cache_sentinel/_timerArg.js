// Module: _timerArg

/* original: D91 */ var composed_value=B((eQ$,_u7)=>{
  var zf6=0,j91=1000,H91=(j91>>1)-1,Xn,J91=Symbol("kFastTimer"),Pn=[],M91=-2,X91=-1,qu7=0,eI7=1;
  function P91(){
    zf6+=H91;
    let q=0,K=Pn.length;
    while(q<K){
      let _=Pn[q];
      if(_._state===qu7)_._idleStart=zf6-H91,_._state=eI7;
      else if(_._state===eI7&&zf6>=_._idleStart+_._idleTimeout)_._state=X91,_._idleStart=-1,_._onTimeout(_._timerArg);
      if(_._state===X91){
        if(_._state=M91,--K!==0)Pn[q]=Pn[K]
      }else++q
    }if(Pn.length=K,Pn.length!==0)Ku7()
  }function Ku7(){
    if(Xn)Xn.refresh();
    else if(clearTimeout(Xn),Xn=setTimeout(P91,H91),Xn.unref)Xn.unref()
  }class W91{
    [J91]=!0;
    _state=M91;
    _idleTimeout=-1;
    _idleStart=-1;
    _onTimeout;
    _timerArg;
    constructor(q,K,_){
      this._onTimeout=q,this._idleTimeout=K,this._timerArg=_,this.refresh()
    }refresh(){
      if(this._state===M91)Pn.push(this);
      if(!Xn||Pn.length===1)Ku7();
      this._state=qu7
    }clear(){
      this._state=X91,this._idleStart=-1
    }
  }_u7.exports={
    setTimeout(q,K,_){
      return K<=j91?setTimeout(q,K,_):new W91(q,K,_)
    },clearTimeout(q){
      if(q[J91])q.clear();
      else clearTimeout(q)
    },setFastTimeout(q,K,_){
      return new W91(q,K,_)
    },clearFastTimeout(q){
      q.clear()
    },now(){
      return zf6
    },tick(q=0){
      zf6+=q-j91+1,P91(),P91()
    },reset(){
      zf6=0,Pn.length=0,clearTimeout(Xn),Xn=null
    },kFastTimer:J91
  }
} /* confidence: 30% */

/* original: P91 */ function helper_fn(){
  zf6+=H91;
  let q=0,K=Pn.length;
  while(q<K){
    let _=Pn[q];
    if(_._state===qu7)_._idleStart=zf6-H91,_._state=eI7;
    else if(_._state===eI7&&zf6>=_._idleStart+_._idleTimeout)_._state=X91,_._idleStart=-1,_._onTimeout(_._timerArg);
    if(_._state===X91){
      if(_._state=M91,--K!==0)Pn[q]=Pn[K]
    }else++q
  }if(Pn.length=K,Pn.length!==0)Ku7()
} /* confidence: 35% */

/* original: Ku7 */ function utility_fn(){
  if(Xn)Xn.refresh();
   /* confidence: 40% */

