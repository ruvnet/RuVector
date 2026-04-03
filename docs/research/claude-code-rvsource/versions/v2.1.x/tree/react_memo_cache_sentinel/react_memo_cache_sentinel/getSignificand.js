// Module: getSignificand

/* original: eQ1 */ var composed_value=B((Dp4)=>{
  Object.defineProperty(Dp4,"__esModule",{
    value:!0
  });
  Dp4.getSignificand=Dp4.getNormalBase2=Dp4.MIN_VALUE=Dp4.MAX_NORMAL_EXPONENT=Dp4.MIN_NORMAL_EXPONENT=Dp4.SIGNIFICAND_WIDTH=void 0;
  Dp4.SIGNIFICAND_WIDTH=52;
  var lfz=2146435072,nfz=1048575,tQ1=1023;
  Dp4.MIN_NORMAL_EXPONENT=-tQ1+1;
  Dp4.MAX_NORMAL_EXPONENT=tQ1;
  Dp4.MIN_VALUE=Math.pow(2,-1022);
  function ifz(q){
    let K=new DataView(new ArrayBuffer(8));
    return K.setFloat64(0,q),((K.getUint32(0)&lfz)>>20)-tQ1
  }Dp4.getNormalBase2=ifz;
  function rfz(q){
    let K=new DataView(new ArrayBuffer(8));
    K.setFloat64(0,q);
    let _=K.getUint32(0),z=K.getUint32(4);
    return(_&nfz)*Math.pow(2,32)+z
  }Dp4.getSignificand=rfz
} /* confidence: 30% */

/* original: lfz */ var lfz=2146435072,nfz=1048575,tQ1=1023;

/* original: Lp4 */ var composed_value=B((yp4)=>{
  Object.defineProperty(yp4,"__esModule",{
    value:!0
  });
  yp4.ExponentMapping=void 0;
  var oy6=eQ1(),zZz=PS8(),Vp4=WS8();
  class Np4{
    _shift;
    constructor(q){
      this._shift=-q
    }mapToIndex(q){
      if(q<oy6.MIN_VALUE)return this._minNormalLowerBoundaryIndex();
      let K=oy6.getNormalBase2(q),_=this._rightShift(oy6.getSignificand(q)-1,oy6.SIGNIFICAND_WIDTH);
      return K+_>>this._shift
    }lowerBoundary(q){
      let K=this._minNormalLowerBoundaryIndex();
      if(q<K)throw new Vp4.MappingError(`underflow: ${q} is < minimum lower boundary: ${K}`);
      let _=this._maxNormalLowerBoundaryIndex();
      if(q>_)throw new Vp4.MappingError(`overflow: ${q} is > maximum lower boundary: ${_}`);
      return zZz.ldexp(1,q<<this._shift)
    }get scale(){
      if(this._shift===0)return 0;
      return-this._shift
    }_minNormalLowerBoundaryIndex(){
      let q=oy6.MIN_NORMAL_EXPONENT>>this._shift;
      if(this._shift<2)q--;
      return q
    }_maxNormalLowerBoundaryIndex(){
      return oy6.MAX_NORMAL_EXPONENT>>this._shift
    }_rightShift(q,K){
      return Math.floor(q*Math.pow(2,-K))
    }
  }yp4.ExponentMapping=Np4
} /* confidence: 30% */

/* original: oy6 */ var composed_value=eQ1(),zZz=PS8(),Vp4=WS8(); /* confidence: 30% */

/* original: K */ let composed_value=oy6.getNormalBase2(q),_=this._rightShift(oy6.getSignificand(q)-1,oy6.SIGNIFICAND_WIDTH); /* confidence: 30% */

/* original: xp4 */ var composed_value=B((Cp4)=>{
  Object.defineProperty(Cp4,"__esModule",{
    value:!0
  });
  Cp4.LogarithmMapping=void 0;
  var ay6=eQ1(),hp4=PS8(),Rp4=WS8();
  class Sp4{
    _scale;
    _scaleFactor;
    _inverseFactor;
    constructor(q){
      this._scale=q,this._scaleFactor=hp4.ldexp(Math.LOG2E,q),this._inverseFactor=hp4.ldexp(Math.LN2,-q)
    }mapToIndex(q){
      if(q<=ay6.MIN_VALUE)return this._minNormalLowerBoundaryIndex()-1;
      if(ay6.getSignificand(q)===0)return(ay6.getNormalBase2(q)<<this._scale)-1;
      let K=Math.floor(Math.log(q)*this._scaleFactor),_=this._maxNormalLowerBoundaryIndex();
      if(K>=_)return _;
      return K
    }lowerBoundary(q){
      let K=this._maxNormalLowerBoundaryIndex();
      if(q>=K){
        if(q===K)return 2*Math.exp((q-(1<<this._scale))/this._scaleFactor);
        throw new Rp4.MappingError(`overflow: ${q} is > maximum lower boundary: ${K}`)
      }let _=this._minNormalLowerBoundaryIndex();
      if(q<=_){
        if(q===_)return ay6.MIN_VALUE;
        else if(q===_-1)return Math.exp((q+(1<<this._scale))/this._scaleFactor)/2;
        throw new Rp4.MappingError(`overflow: ${q} is < minimum lower boundary: ${_}`)
      }return Math.exp(q*this._inverseFactor)
    }get scale(){
      return this._scale
    }_minNormalLowerBoundaryIndex(){
      return ay6.MIN_NORMAL_EXPONENT<<this._scale
    }_maxNormalLowerBoundaryIndex(){
      return(ay6.MAX_NORMAL_EXPONENT+1<<this._scale)-1
    }
  }Cp4.LogarithmMapping=Sp4
} /* confidence: 30% */

/* original: ay6 */ var composed_value=eQ1(),hp4=PS8(),Rp4=WS8(); /* confidence: 30% */

/* original: YZz */ var composed_value=Lp4(),$Zz=xp4(),OZz=WS8(),Ip4=-10,up4=20,AZz=Array.from({
  length:31
} /* confidence: 30% */

/* original: ifz */ function helper_fn(q){
  let K=new DataView(new ArrayBuffer(8));
  return K.setFloat64(0,q),((K.getUint32(0)&lfz)>>20)-tQ1
} /* confidence: 35% */

/* original: Np4 */ class entity_class{
  _shift;
  constructor(q){
    this._shift=-q
  }mapToIndex(q){
    if(q<oy6.MIN_VALUE)return this._minNormalLowerBoundaryIndex();
    let K=oy6.getNormalBase2(q),_=this._rightShift(oy6.getSignificand(q)-1,oy6.SIGNIFICAND_WIDTH);
    return K+_>>this._shift
  }lowerBoundary(q){
    let K=this._minNormalLowerBoundaryIndex();
    if(q<K)throw new Vp4.MappingError(`underflow: ${q} is < minimum lower boundary: ${K}`);
    let _=this._maxNormalLowerBoundaryIndex();
    if(q>_)throw new Vp4.MappingError(`overflow: ${q} is > maximum lower boundary: ${_}`);
    return zZz.ldexp(1,q<<this._shift)
  }get scale(){
    if(this._shift===0)return 0;
    return-this._shift
  }_minNormalLowerBoundaryIndex(){
    let q=oy6.MIN_NORMAL_EXPONENT>>this._shift;
    if(this._shift<2)q--;
    return q
  }_maxNormalLowerBoundaryIndex(){
    return oy6.MAX_NORMAL_EXPONENT>>this._shift
  }_rightShift(q,K){
    return Math.floor(q*Math.pow(2,-K))
  }
} /* confidence: 45% */

/* original: Sp4 */ class entity_class{
  _scale;
  _scaleFactor;
  _inverseFactor;
  constructor(q){
    this._scale=q,this._scaleFactor=hp4.ldexp(Math.LOG2E,q),this._inverseFactor=hp4.ldexp(Math.LN2,-q)
  }mapToIndex(q){
    if(q<=ay6.MIN_VALUE)return this._minNormalLowerBoundaryIndex()-1;
    if(ay6.getSignificand(q)===0)return(ay6.getNormalBase2(q)<<this._scale)-1;
    let K=Math.floor(Math.log(q)*this._scaleFactor),_=this._maxNormalLowerBoundaryIndex();
    if(K>=_)return _;
    return K
  }lowerBoundary(q){
    let K=this._maxNormalLowerBoundaryIndex();
    if(q>=K){
      if(q===K)return 2*Math.exp((q-(1<<this._scale))/this._scaleFactor);
      throw new Rp4.MappingError(`overflow: ${q} is > maximum lower boundary: ${K}`)
    }let _=this._minNormalLowerBoundaryIndex();
    if(q<=_){
      if(q===_)return ay6.MIN_VALUE;
      else if(q===_-1)return Math.exp((q+(1<<this._scale))/this._scaleFactor)/2;
      throw new Rp4.MappingError(`overflow: ${q} is < minimum lower boundary: ${_}`)
    }return Math.exp(q*this._inverseFactor)
  }get scale(){
    return this._scale
  }_minNormalLowerBoundaryIndex(){
    return ay6.MIN_NORMAL_EXPONENT<<this._scale
  }_maxNormalLowerBoundaryIndex(){
    return(ay6.MAX_NORMAL_EXPONENT+1<<this._scale)-1
  }
} /* confidence: 45% */

