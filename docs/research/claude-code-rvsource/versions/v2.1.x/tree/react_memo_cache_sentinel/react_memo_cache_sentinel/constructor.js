// Module: constructor

/* original: tA5 */ var objectBoolean_objectDate_objec="[object Boolean]",eA5="[object Date]",qw5="[object Map]",Kw5="[object Number]",_w5="[object RegExp]",zw5="[object Set]",Yw5="[object String]",$w5="[object Symbol]",Ow5="[object ArrayBuffer]",Aw5="[object DataView]",ww5="[object Float32Array]",jw5="[object Float64Array]",Hw5="[object Int8Array]",Jw5="[object Int16Array]",Mw5="[object Int32Array]",Xw5="[object Uint8Array]",Pw5="[object Uint8ClampedArray]",Ww5="[object Uint16Array]",Dw5="[object Uint32Array]",pX7; /* confidence: 65% */

/* original: fw5 */ function helper_fn(q,K,_){
  var z=q.constructor;
  switch(K){
    case Ow5:return IP6(q);
    case tA5:case eA5:return new z(+q);
    case Aw5:return RX7(q,_);
    case ww5:case jw5:case Hw5:case Jw5:case Mw5:case Xw5:case Pw5:case Ww5:case Dw5:return Q_8(q,_);
    case qw5:return new z;
    case Kw5:case Yw5:return new z(q);
    case _w5:return CX7(q);
    case zw5:return new z;
    case $w5:return uX7(q)
  }
} /* confidence: 35% */

