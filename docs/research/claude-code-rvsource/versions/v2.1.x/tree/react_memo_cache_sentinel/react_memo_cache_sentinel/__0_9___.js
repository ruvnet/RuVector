// Module: __0_9___

/* original: G37 */ var u3000u303Fu3040u309Fu30A0u30FF=B((N0Y)=>{
  var f48="(?:[u3000-u303F]|[u3040-u309F]|[u30A0-u30FF]|[uFF00-uFFEF]|[u4E00-u9FAF]|[u2605-u2606]|[u2190-u2195]|u203B|[u2010u2015u2018u2019u2025u2026u201Cu201Du2225u2260]|[u0391-u0451]|[u00A7u00A8u00B1u00B4u00D7u00F7])+";
  f48=f48.replace(/u/g,"\\u");
  var v0Y="(?:(?![A-Z0-9 $%*+\\-./:]|"+f48+`)(?:.|[\r
]))+`;
  N0Y.KANJI=new RegExp(f48,"g");
  N0Y.BYTE_KANJI=new RegExp("[^A-Z0-9 $%*+\\-./:]+","g");
  N0Y.BYTE=new RegExp(v0Y,"g");
  N0Y.NUMERIC=new RegExp("[0-9]+","g");
  N0Y.ALPHANUMERIC=new RegExp("[A-Z $%*+\\-./:]+","g");
  var T0Y=new RegExp("^"+f48+"$"),k0Y=new RegExp("^[0-9]+$"),V0Y=new RegExp("^[A-Z0-9 $%*+\\-./:]+$");
  N0Y.testKanji=function(K){
    return T0Y.test(K)
  };
  N0Y.testNumeric=function(K){
    return k0Y.test(K)
  };
  N0Y.testAlphanumeric=function(K){
    return V0Y.test(K)
  }
} /* confidence: 65% */

/* original: T0Y */ var __09_AZ09=new RegExp("^"+f48+"$"),k0Y=new RegExp("^[0-9]+$"),V0Y=new RegExp("^[A-Z0-9 $%*+\\-./:]+$"); /* confidence: 65% */

