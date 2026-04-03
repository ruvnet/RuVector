// Module: kitty

/* original: K4K */ var K4K="Claude Code";

/* original: HFz */ function auto_iterm2_iterm2_iterm2_with(q,K,_){
  let z=K.title||K4K;
  try{
    switch(q){
      case"auto":return JFz(K,_);
      case"iterm2":return _.notifyITerm2(K),"iterm2";
      case"iterm2_with_bell":return _.notifyITerm2(K),_.notifyBell(),"iterm2_with_bell";
      case"kitty":return _.notifyKitty({
        ...K,title:z,id:_4K()
      }),"kitty";
      case"ghostty":return _.notifyGhostty({
        ...K,title:z
      }),"ghostty";
      case"terminal_bell":return _.notifyBell(),"terminal_bell";
      case"notifications_disabled":return"disabled";
      default:return"none"
    }
  }catch{
    return"error"
  }
} /* confidence: 65% */

/* original: JFz */ function Apple_Terminal_terminal_bell_n(q,K){
  let _=q.title||K4K;
  switch(Y7.terminal){
    case"Apple_Terminal":{
      if(await MFz())return K.notifyBell(),"terminal_bell";
      return"no_method_available"
    }case"iTerm.app":return K.notifyITerm2(q),"iterm2";
    case"kitty":return K.notifyKitty({
      ...q,title:_,id:_4K()
    }),"kitty";
    case"ghostty":return K.notifyGhostty({
      ...q,title:_
    }),"ghostty";
    default:return"no_method_available"
  }
} /* confidence: 65% */

