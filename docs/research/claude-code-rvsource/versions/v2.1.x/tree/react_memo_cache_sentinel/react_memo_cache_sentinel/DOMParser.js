// Module: DOMParser

/* original: rJK */ var rJK=typeof window<"u"?window:{
  
};

/* original: f4Y */ var composed_value=W4Y()?rJK.DOMParser:D4Y(); /* confidence: 30% */

/* original: W4Y */ function helper_fn(){
  var q=rJK.DOMParser,K=!1;
  try{
    if(new q().parseFromString("","text/html"))K=!0
  }catch(_){
    
  }return K
} /* confidence: 35% */

