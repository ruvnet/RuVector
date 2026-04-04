// Module: killed

/* original: H7_ */ var H7_=1e4,pV1=null,MZ8=null;

/* original: _Iq */ function security_findgenericpassword_a(q){
  return new Promise((K)=>{
    j7_("security",["find-generic-password","-a",Ki(),"-w","-s",q],{
      encoding:"utf-8",timeout:H7_
    },(_,z)=>{
      K({
        stdout:_?null:z?.trim()||null,timedOut:Boolean(_&&"killed"in _&&_.killed)
      })
    })
  })
} /* confidence: 65% */

