// Module: priority

/* original: hHK */ var hHK="!important";

/* original: Y */ var composed_value=RHK(q+":"+K); /* confidence: 30% */

/* original: RHK */ function StringTrimmer(q){
  let K={
    property:{
      
    },priority:{
      
    }
  };
  if(!q)return K;
  let _=n7Y(q);
  if(_.length<2)return K;
  for(let z=0;
  z<_.length;
  z+=2){
    let Y=_[z],$=_[z+1];
    if($.endsWith(hHK))K.priority[Y]="important",$=$.slice(0,-hHK.length).trim();
    K.property[Y]=$
  }return K
} /* confidence: 41% */

