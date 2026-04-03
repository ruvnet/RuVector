// Module: osascript

/* original: Hy8 */ var Hy8={
  
};

/* original: Cx4 */ function darwin_tengu_collage_kaleidosc(){
  if(process.platform!=="darwin")return!1;
  if(L8("tengu_collage_kaleidoscope",!0))try{
    let{
      getNativeModule:K
    }=await Promise.resolve().then(() => (Jy8(),Hy8)),_=K()?.hasClipboardImage;
    if(_)return _()
  }catch(K){
    j6(K)
  }return(await x7("osascript",["-e","the clipboard as Â«class PNGfÂ»"])).code===0
} /* confidence: 65% */

