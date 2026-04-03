// Module: blurred

/* original: ih1 */ var ih1="unknown",z74,nh1;

/* original: rh1 */ function focused_blurred(q){
  ih1=q?"focused":"blurred";
  for(let K of nh1)K();
  if(!q){
    for(let K of z74)K();
    z74.clear()
  }
} /* confidence: 65% */

/* original: JT6 */ function helper_fn(){
  return ih1!=="blurred"
} /* confidence: 35% */

/* original: oh1 */ function helper_fn(){
  return ih1
} /* confidence: 35% */

/* original: H74 */ function helper_fn(q){
  let K=Y6(6),{
    children:_
  }=q,z=qA6.useSyncExternalStore(Xv8,JT6),Y=qA6.useSyncExternalStore(Xv8,oh1),$;
  if(K[0]!==z||K[1]!==Y)$={
    isTerminalFocused:z,terminalFocusState:Y
  },K[0]=z,K[1]=Y,K[2]=$;
  else $=K[2];
  let O=$,A;
  if(K[3]!==_||K[4]!==O)A=qA6.default.createElement(sh1.Provider,{
    value:O
  },_),K[3]=_,K[4]=O,K[5]=A;
  else A=K[5];
  return A
} /* confidence: 35% */

