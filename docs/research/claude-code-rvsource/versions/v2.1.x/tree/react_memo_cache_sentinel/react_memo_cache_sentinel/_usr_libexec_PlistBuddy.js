// Module: _usr_libexec_PlistBuddy

/* original: LS6 */ function Library_Preferences_comappleTe(){
  return AjY(OjY(),"Library","Preferences","com.apple.Terminal.plist")
} /* confidence: 65% */

/* original: sTK */ function usrlibexecPlistBuddy_c_usrlibe(q){
  let{
    code:K
  }=await K1("/usr/libexec/PlistBuddy",["-c",`Add :'Window Settings':'${q}':useOptionAsMetaKey bool true`,LS6()]);
  if(K!==0){
    let{
      code:_
    }=await K1("/usr/libexec/PlistBuddy",["-c",`Set :'Window Settings':'${q}':useOptionAsMetaKey true`,LS6()]);
    if(_!==0)return j6(Error(`Failed to enable Option as Meta key for Terminal.app profile: ${q}`)),!1
  }return!0
} /* confidence: 65% */

/* original: tTK */ function usrlibexecPlistBuddy_c_usrlibe(q){
  let{
    code:K
  }=await K1("/usr/libexec/PlistBuddy",["-c",`Add :'Window Settings':'${q}':Bell bool false`,LS6()]);
  if(K!==0){
    let{
      code:_
    }=await K1("/usr/libexec/PlistBuddy",["-c",`Set :'Window Settings':'${q}':Bell false`,LS6()]);
    if(_!==0)return j6(Error(`Failed to disable audio bell for Terminal.app profile: ${q}`)),!1
  }return!0
} /* confidence: 65% */

