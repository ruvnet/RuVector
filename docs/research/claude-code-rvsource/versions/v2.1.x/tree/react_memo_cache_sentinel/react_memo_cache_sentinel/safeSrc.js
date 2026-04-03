// Module: safeSrc

/* original: ao6 */ var azAZ09_s_d_g_g_NUMERICIDENTIFI=B((QU,lP4)=>{
  var{
    MAX_SAFE_COMPONENT_LENGTH:bm1,MAX_SAFE_BUILD_LENGTH:va_,MAX_LENGTH:Ta_
  }=Wy8(),ka_=oo6();
  QU=lP4.exports={
    
  };
  var Va_=QU.re=[],Na_=QU.safeRe=[],a4=QU.src=[],ya_=QU.safeSrc=[],s4=QU.t={
    
  },Ea_=0,xm1="[a-zA-Z0-9-]",La_=[["\\s",1],["\\d",Ta_],[xm1,va_]],ha_=(q)=>{
    for(let[K,_]of La_)q=q.split(`${K}*`).join(`${K}{0,${_}}`).split(`${K}+`).join(`${K}{1,${_}}`);
    return q
  },B9=(q,K,_)=>{
    let z=ha_(K),Y=Ea_++;
    ka_(q,Y,K),s4[q]=Y,a4[Y]=K,ya_[Y]=z,Va_[Y]=new RegExp(K,_?"g":void 0),Na_[Y]=new RegExp(z,_?"g":void 0)
  };
  B9("NUMERICIDENTIFIER","0|[1-9]\\d*");
  B9("NUMERICIDENTIFIERLOOSE","\\d+");
  B9("NONNUMERICIDENTIFIER",`\\d*[a-zA-Z-]${xm1}*`);
  B9("MAINVERSION",`(${a4[s4.NUMERICIDENTIFIER]})\\.(${a4[s4.NUMERICIDENTIFIER]})\\.(${a4[s4.NUMERICIDENTIFIER]})`);
  B9("MAINVERSIONLOOSE",`(${a4[s4.NUMERICIDENTIFIERLOOSE]})\\.(${a4[s4.NUMERICIDENTIFIERLOOSE]})\\.(${a4[s4.NUMERICIDENTIFIERLOOSE]})`);
  B9("PRERELEASEIDENTIFIER",`(?:${a4[s4.NONNUMERICIDENTIFIER]}|${a4[s4.NUMERICIDENTIFIER]})`);
  B9("PRERELEASEIDENTIFIERLOOSE",`(?:${a4[s4.NONNUMERICIDENTIFIER]}|${a4[s4.NUMERICIDENTIFIERLOOSE]})`);
  B9("PRERELEASE",`(?:-(${a4[s4.PRERELEASEIDENTIFIER]}(?:\\.${a4[s4.PRERELEASEIDENTIFIER]})*))`);
  B9("PRERELEASELOOSE",`(?:-?(${a4[s4.PRERELEASEIDENTIFIERLOOSE]}(?:\\.${a4[s4.PRERELEASEIDENTIFIERLOOSE]})*))`);
  B9("BUILDIDENTIFIER",`${xm1}+`);
  B9("BUILD",`(?:\\+(${a4[s4.BUILDIDENTIFIER]}(?:\\.${a4[s4.BUILDIDENTIFIER]})*))`);
  B9("FULLPLAIN",`v?${a4[s4.MAINVERSION]}${a4[s4.PRERELEASE]}?${a4[s4.BUILD]}?`);
  B9("FULL",`^${a4[s4.FULLPLAIN]}$`);
  B9("LOOSEPLAIN",`[v=\\s]*${a4[s4.MAINVERSIONLOOSE]}${a4[s4.PRERELEASELOOSE]}?${a4[s4.BUILD]}?`);
  B9("LOOSE",`^${a4[s4.LOOSEPLAIN]}$`);
  B9("GTLT","((?:<|>)?=?)");
  B9("XRANGEIDENTIFIERLOOSE",`${a4[s4.NUMERICIDENTIFIERLOOSE]}|x|X|\\*`);
  B9("XRANGEIDENTIFIER",`${a4[s4.NUMERICIDENTIFIER]}|x|X|\\*`);
  B9("XRANGEPLAIN",`[v=\\s]*(${a4[s4.XRANGEIDENTIFIER]})(?:\\.(${a4[s4.XRANGEIDENTIFIER]})(?:\\.(${a4[s4.XRANGEIDENTIFIER]})(?:${a4[s4.PRERELEASE]})?${a4[s4.BUILD]}?)?)?`);
  B9("XRANGEPLAINLOOSE",`[v=\\s]*(${a4[s4.XRANGEIDENTIFIERLOOSE]})(?:\\.(${a4[s4.XRANGEIDENTIFIERLOOSE]})(?:\\.(${a4[s4.XRANGEIDENTIFIERLOOSE]})(?:${a4[s4.PRERELEASELOOSE]})?${a4[s4.BUILD]}?)?)?`);
  B9("XRANGE",`^${a4[s4.GTLT]}\\s*${a4[s4.XRANGEPLAIN]}$`);
  B9("XRANGELOOSE",`^${a4[s4.GTLT]}\\s*${a4[s4.XRANGEPLAINLOOSE]}$`);
  B9("COERCEPLAIN",`(^|[^\\d])(\\d{1,${bm1}})(?:\\.(\\d{1,${bm1}}))?(?:\\.(\\d{1,${bm1}}))?`);
  B9("COERCE",`${a4[s4.COERCEPLAIN]}(?:$|[^\\d])`);
  B9("COERCEFULL",a4[s4.COERCEPLAIN]+`(?:${a4[s4.PRERELEASE]})?(?:${a4[s4.BUILD]})?(?:$|[^\\d])`);
  B9("COERCERTL",a4[s4.COERCE],!0);
  B9("COERCERTLFULL",a4[s4.COERCEFULL],!0);
  B9("LONETILDE","(?:~>?)");
  B9("TILDETRIM",`(\\s*)${a4[s4.LONETILDE]}\\s+`,!0);
  QU.tildeTrimReplace="$1~";
  B9("TILDE",`^${a4[s4.LONETILDE]}${a4[s4.XRANGEPLAIN]}$`);
  B9("TILDELOOSE",`^${a4[s4.LONETILDE]}${a4[s4.XRANGEPLAINLOOSE]}$`);
  B9("LONECARET","(?:\\^)");
  B9("CARETTRIM",`(\\s*)${a4[s4.LONECARET]}\\s+`,!0);
  QU.caretTrimReplace="$1^";
  B9("CARET",`^${a4[s4.LONECARET]}${a4[s4.XRANGEPLAIN]}$`);
  B9("CARETLOOSE",`^${a4[s4.LONECARET]}${a4[s4.XRANGEPLAINLOOSE]}$`);
  B9("COMPARATORLOOSE",`^${a4[s4.GTLT]}\\s*(${a4[s4.LOOSEPLAIN]})$|^$`);
  B9("COMPARATOR",`^${a4[s4.GTLT]}\\s*(${a4[s4.FULLPLAIN]})$|^$`);
  B9("COMPARATORTRIM",`(\\s*)${a4[s4.GTLT]}\\s*(${a4[s4.LOOSEPLAIN]}|${a4[s4.XRANGEPLAIN]})`,!0);
  QU.comparatorTrimReplace="$1$2$3";
  B9("HYPHENRANGE",`^\\s*(${a4[s4.XRANGEPLAIN]})\\s+-\\s+(${a4[s4.XRANGEPLAIN]})\\s*$`);
  B9("HYPHENRANGELOOSE",`^\\s*(${a4[s4.XRANGEPLAINLOOSE]})\\s+-\\s+(${a4[s4.XRANGEPLAINLOOSE]})\\s*$`);
  B9("STAR","(<|>)?=?\\s*\\*");
  B9("GTE0","^\\s*>=\\s*0\\.0\\.0\\s*$");
  B9("GTE0PRE","^\\s*>=\\s*0\\.0\\.0-0\\s*$")
} /* confidence: 65% */

/* original: Va_ */ var Va_=QU.re=[],Na_=QU.safeRe=[],a4=QU.src=[],ya_=QU.safeSrc=[],s4=QU.t={
  
}

