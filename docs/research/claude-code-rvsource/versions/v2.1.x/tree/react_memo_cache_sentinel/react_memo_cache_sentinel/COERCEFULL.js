// Module: COERCEFULL

/* original: Hc6 */ var major_premajor_minor_preminor_=B((WHO,Wkq)=>{
  var ng9=Number.MAX_SAFE_INTEGER||9007199254740991,ig9=["major","premajor","minor","preminor","patch","prepatch","prerelease"];
  Wkq.exports={
    MAX_LENGTH:256,MAX_SAFE_COMPONENT_LENGTH:16,MAX_SAFE_BUILD_LENGTH:250,MAX_SAFE_INTEGER:ng9,RELEASE_TYPES:ig9,SEMVER_SPEC_VERSION:"2.0.0",FLAG_INCLUDE_PRERELEASE:1,FLAG_LOOSE:2
  }
} /* confidence: 65% */

/* original: JG6 */ var azAZ09_s_d_g_g_NUMERICIDENTIFI=B((NF,fkq)=>{
  var{
    MAX_SAFE_COMPONENT_LENGTH:gG1,MAX_SAFE_BUILD_LENGTH:og9,MAX_LENGTH:ag9
  }=Hc6(),sg9=Jc6();
  NF=fkq.exports={
    
  };
  var tg9=NF.re=[],eg9=NF.safeRe=[],n4=NF.src=[],qF9=NF.safeSrc=[],i4=NF.t={
    
  },KF9=0,FG1="[a-zA-Z0-9-]",_F9=[["\\s",1],["\\d",ag9],[FG1,og9]],zF9=(q)=>{
    for(let[K,_]of _F9)q=q.split(`${K}*`).join(`${K}{0,${_}}`).split(`${K}+`).join(`${K}{1,${_}}`);
    return q
  },x9=(q,K,_)=>{
    let z=zF9(K),Y=KF9++;
    sg9(q,Y,K),i4[q]=Y,n4[Y]=K,qF9[Y]=z,tg9[Y]=new RegExp(K,_?"g":void 0),eg9[Y]=new RegExp(z,_?"g":void 0)
  };
  x9("NUMERICIDENTIFIER","0|[1-9]\\d*");
  x9("NUMERICIDENTIFIERLOOSE","\\d+");
  x9("NONNUMERICIDENTIFIER",`\\d*[a-zA-Z-]${FG1}*`);
  x9("MAINVERSION",`(${n4[i4.NUMERICIDENTIFIER]})\\.(${n4[i4.NUMERICIDENTIFIER]})\\.(${n4[i4.NUMERICIDENTIFIER]})`);
  x9("MAINVERSIONLOOSE",`(${n4[i4.NUMERICIDENTIFIERLOOSE]})\\.(${n4[i4.NUMERICIDENTIFIERLOOSE]})\\.(${n4[i4.NUMERICIDENTIFIERLOOSE]})`);
  x9("PRERELEASEIDENTIFIER",`(?:${n4[i4.NUMERICIDENTIFIER]}|${n4[i4.NONNUMERICIDENTIFIER]})`);
  x9("PRERELEASEIDENTIFIERLOOSE",`(?:${n4[i4.NUMERICIDENTIFIERLOOSE]}|${n4[i4.NONNUMERICIDENTIFIER]})`);
  x9("PRERELEASE",`(?:-(${n4[i4.PRERELEASEIDENTIFIER]}(?:\\.${n4[i4.PRERELEASEIDENTIFIER]})*))`);
  x9("PRERELEASELOOSE",`(?:-?(${n4[i4.PRERELEASEIDENTIFIERLOOSE]}(?:\\.${n4[i4.PRERELEASEIDENTIFIERLOOSE]})*))`);
  x9("BUILDIDENTIFIER",`${FG1}+`);
  x9("BUILD",`(?:\\+(${n4[i4.BUILDIDENTIFIER]}(?:\\.${n4[i4.BUILDIDENTIFIER]})*))`);
  x9("FULLPLAIN",`v?${n4[i4.MAINVERSION]}${n4[i4.PRERELEASE]}?${n4[i4.BUILD]}?`);
  x9("FULL",`^${n4[i4.FULLPLAIN]}$`);
  x9("LOOSEPLAIN",`[v=\\s]*${n4[i4.MAINVERSIONLOOSE]}${n4[i4.PRERELEASELOOSE]}?${n4[i4.BUILD]}?`);
  x9("LOOSE",`^${n4[i4.LOOSEPLAIN]}$`);
  x9("GTLT","((?:<|>)?=?)");
  x9("XRANGEIDENTIFIERLOOSE",`${n4[i4.NUMERICIDENTIFIERLOOSE]}|x|X|\\*`);
  x9("XRANGEIDENTIFIER",`${n4[i4.NUMERICIDENTIFIER]}|x|X|\\*`);
  x9("XRANGEPLAIN",`[v=\\s]*(${n4[i4.XRANGEIDENTIFIER]})(?:\\.(${n4[i4.XRANGEIDENTIFIER]})(?:\\.(${n4[i4.XRANGEIDENTIFIER]})(?:${n4[i4.PRERELEASE]})?${n4[i4.BUILD]}?)?)?`);
  x9("XRANGEPLAINLOOSE",`[v=\\s]*(${n4[i4.XRANGEIDENTIFIERLOOSE]})(?:\\.(${n4[i4.XRANGEIDENTIFIERLOOSE]})(?:\\.(${n4[i4.XRANGEIDENTIFIERLOOSE]})(?:${n4[i4.PRERELEASELOOSE]})?${n4[i4.BUILD]}?)?)?`);
  x9("XRANGE",`^${n4[i4.GTLT]}\\s*${n4[i4.XRANGEPLAIN]}$`);
  x9("XRANGELOOSE",`^${n4[i4.GTLT]}\\s*${n4[i4.XRANGEPLAINLOOSE]}$`);
  x9("COERCEPLAIN",`(^|[^\\d])(\\d{1,${gG1}})(?:\\.(\\d{1,${gG1}}))?(?:\\.(\\d{1,${gG1}}))?`);
  x9("COERCE",`${n4[i4.COERCEPLAIN]}(?:$|[^\\d])`);
  x9("COERCEFULL",n4[i4.COERCEPLAIN]+`(?:${n4[i4.PRERELEASE]})?(?:${n4[i4.BUILD]})?(?:$|[^\\d])`);
  x9("COERCERTL",n4[i4.COERCE],!0);
  x9("COERCERTLFULL",n4[i4.COERCEFULL],!0);
  x9("LONETILDE","(?:~>?)");
  x9("TILDETRIM",`(\\s*)${n4[i4.LONETILDE]}\\s+`,!0);
  NF.tildeTrimReplace="$1~";
  x9("TILDE",`^${n4[i4.LONETILDE]}${n4[i4.XRANGEPLAIN]}$`);
  x9("TILDELOOSE",`^${n4[i4.LONETILDE]}${n4[i4.XRANGEPLAINLOOSE]}$`);
  x9("LONECARET","(?:\\^)");
  x9("CARETTRIM",`(\\s*)${n4[i4.LONECARET]}\\s+`,!0);
  NF.caretTrimReplace="$1^";
  x9("CARET",`^${n4[i4.LONECARET]}${n4[i4.XRANGEPLAIN]}$`);
  x9("CARETLOOSE",`^${n4[i4.LONECARET]}${n4[i4.XRANGEPLAINLOOSE]}$`);
  x9("COMPARATORLOOSE",`^${n4[i4.GTLT]}\\s*(${n4[i4.LOOSEPLAIN]})$|^$`);
  x9("COMPARATOR",`^${n4[i4.GTLT]}\\s*(${n4[i4.FULLPLAIN]})$|^$`);
  x9("COMPARATORTRIM",`(\\s*)${n4[i4.GTLT]}\\s*(${n4[i4.LOOSEPLAIN]}|${n4[i4.XRANGEPLAIN]})`,!0);
  NF.comparatorTrimReplace="$1$2$3";
  x9("HYPHENRANGE",`^\\s*(${n4[i4.XRANGEPLAIN]})\\s+-\\s+(${n4[i4.XRANGEPLAIN]})\\s*$`);
  x9("HYPHENRANGELOOSE",`^\\s*(${n4[i4.XRANGEPLAINLOOSE]})\\s+-\\s+(${n4[i4.XRANGEPLAINLOOSE]})\\s*$`);
  x9("STAR","(<|>)?=?\\s*\\*");
  x9("GTE0","^\\s*>=\\s*0\\.0\\.0\\s*$");
  x9("GTE0PRE","^\\s*>=\\s*0\\.0\\.0-0\\s*$")
} /* confidence: 65% */

