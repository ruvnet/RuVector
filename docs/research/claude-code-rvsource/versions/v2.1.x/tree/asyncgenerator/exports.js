// Module: exports

/* original: U$8 */ var U$8=B((yD$,WV7)=>{
  WV7.exports=TypeError
}

/* original: yN7 */ var ObjectdefineProperty_u_boolean=B((qf$,NN7)=>{
  var oN5=vN7(),VN7=oN5("%Object.defineProperty%",!0),aN5=kN7()(),sN5=l$8(),tN5=U$8(),r$8=aN5?Symbol.toStringTag:null;
  NN7.exports=function(K,_){
    var z=arguments.length>2&&!!arguments[2]&&arguments[2].force,Y=arguments.length>2&&!!arguments[2]&&arguments[2].nonConfigurable;
    if(typeof z<"u"&&typeof z!=="boolean"||typeof Y<"u"&&typeof Y!=="boolean")throw new tN5("if provided, the `overrideIfSet` and `nonConfigurable` options must be booleans");
    if(r$8&&(z||!sN5(K,r$8)))if(VN7)VN7(K,r$8,{
      configurable:!Y,enumerable:!1,value:_,writable:!1
    });
    else K[r$8]=_
  }
} /* confidence: 65% */

