// Module: _listeners

/* original: Dd4 */ var composed_value=B((ebA,Wd4)=>{
  Wd4.exports=QS8;
  function QS8(){
    this._listeners={
      
    }
  }QS8.prototype.on=function(K,_,z){
    return(this._listeners[K]||(this._listeners[K]=[])).push({
      fn:_,ctx:z||this
    }),this
  };
  QS8.prototype.off=function(K,_){
    if(K===void 0)this._listeners={
      
    };
    else if(_===void 0)this._listeners[K]=[];
    else{
      var z=this._listeners[K];
      for(var Y=0;
      Y<z.length;
      )if(z[Y].fn===_)z.splice(Y,1);
      else++Y
    }return this
  };
  QS8.prototype.emit=function(K){
    var _=this._listeners[K];
    if(_){
      var z=[],Y=1;
      for(;
      Y<arguments.length;
      )z.push(arguments[Y++]);
      for(Y=0;
      Y<_.length;
      )_[Y].fn.apply(_[Y++].ctx,z)
    }return this
  }
} /* confidence: 30% */

/* original: QS8 */ function utility_fn(){
  this._listeners={
    
  }
} /* confidence: 40% */

