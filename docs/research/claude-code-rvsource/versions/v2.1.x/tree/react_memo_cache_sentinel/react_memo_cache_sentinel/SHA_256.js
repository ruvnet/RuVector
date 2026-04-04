// Module: SHA_256

/* original: pSq */ var __esModule_u_SubtleCryptonotfo=B((uSq)=>{
  Object.defineProperty(uSq,"__esModule",{
    value:!0
  });
  uSq.BrowserCrypto=void 0;
  var CG6=Ok1(),je9=bG6();
  class pf8{
    constructor(){
      if(typeof window>"u"||window.crypto===void 0||window.crypto.subtle===void 0)throw Error("SubtleCrypto not found. Make sure it's an https:// website.")
    }async sha256DigestBase64(q){
      let K=new TextEncoder().encode(q),_=await window.crypto.subtle.digest("SHA-256",K);
      return CG6.fromByteArray(new Uint8Array(_))
    }randomBytesBase64(q){
      let K=new Uint8Array(q);
      return window.crypto.getRandomValues(K),CG6.fromByteArray(K)
    }static padBase64(q){
      while(q.length%4!==0)q+="=";
      return q
    }async verify(q,K,_){
      let z={
        name:"RSASSA-PKCS1-v1_5",hash:{
          name:"SHA-256"
        }
      },Y=new TextEncoder().encode(K),$=CG6.toByteArray(pf8.padBase64(_)),O=await window.crypto.subtle.importKey("jwk",q,z,!0,["verify"]);
      return await window.crypto.subtle.verify(z,O,$,Y)
    }async sign(q,K){
      let _={
        name:"RSASSA-PKCS1-v1_5",hash:{
          name:"SHA-256"
        }
      },z=new TextEncoder().encode(K),Y=await window.crypto.subtle.importKey("jwk",q,_,!0,["sign"]),$=await window.crypto.subtle.sign(_,Y,z);
      return CG6.fromByteArray(new Uint8Array($))
    }decodeBase64StringUtf8(q){
      let K=CG6.toByteArray(pf8.padBase64(q));
      return new TextDecoder().decode(K)
    }encodeBase64StringUtf8(q){
      let K=new TextEncoder().encode(q);
      return CG6.fromByteArray(K)
    }async sha256DigestHex(q){
      let K=new TextEncoder().encode(q),_=await window.crypto.subtle.digest("SHA-256",K);
      return(0,je9.fromArrayBufferToHex)(_)
    }async signWithHmacSha256(q,K){
      let _=typeof q==="string"?q:String.fromCharCode(...new Uint16Array(q)),z=new TextEncoder,Y=await window.crypto.subtle.importKey("raw",z.encode(_),{
        name:"HMAC",hash:{
          name:"SHA-256"
        }
      },!1,["sign"]);
      return window.crypto.subtle.sign("HMAC",Y,z.encode(K))
    }
  }uSq.BrowserCrypto=pf8
} /* confidence: 65% */

/* original: CG6 */ var composed_value=Ok1(),je9=bG6(); /* confidence: 30% */

/* original: bG6 */ var __esModule_u_u_u_0=B((dSq)=>{
  Object.defineProperty(dSq,"__esModule",{
    value:!0
  });
  dSq.createCrypto=Pe9;
  dSq.hasBrowserCrypto=QSq;
  dSq.fromArrayBufferToHex=We9;
  var Me9=pSq(),Xe9=USq();
  function Pe9(){
    if(QSq())return new Me9.BrowserCrypto;
    return new Xe9.NodeCrypto
  }function QSq(){
    return typeof window<"u"&&typeof window.crypto<"u"&&typeof window.crypto.subtle<"u"
  }function We9(q){
    return Array.from(new Uint8Array(q)).map((_)=>{
      return _.toString(16).padStart(2,"0")
    }).join("")
  }
} /* confidence: 65% */

/* original: Me9 */ var composed_value=pSq(),Xe9=USq(); /* confidence: 30% */

/* original: Ek1 */ var headers=B((UCq)=>{
  Object.defineProperty(UCq,"__esModule",{
    value:!0
  });
  UCq.OAuthClientAuthHandler=void 0;
  UCq.getErrorFromOAuthErrorResponse=M6_;
  var gCq=U6("querystring"),H6_=bG6(),J6_=["PUT","POST","PATCH"];
  class FCq{
    constructor(q){
      this.clientAuthentication=q,this.crypto=(0,H6_.createCrypto)()
    }applyClientAuthenticationOptions(q,K){
      if(this.injectAuthenticatedHeaders(q,K),!K)this.injectAuthenticatedRequestBody(q)
    }injectAuthenticatedHeaders(q,K){
      var _;
      if(K)q.headers=q.headers||{
        
      },Object.assign(q.headers,{
        Authorization:`Bearer ${K}}`
      });
      else if(((_=this.clientAuthentication)===null||_===void 0?void 0:_.confidentialClientType)==="basic"){
        q.headers=q.headers||{
          
        };
        let z=this.clientAuthentication.clientId,Y=this.clientAuthentication.clientSecret||"",$=this.crypto.encodeBase64StringUtf8(`${z}:${Y}`);
        Object.assign(q.headers,{
          Authorization:`Basic ${$}`
        })
      }
    }injectAuthenticatedRequestBody(q){
      var K;
      if(((K=this.clientAuthentication)===null||K===void 0?void 0:K.confidentialClientType)==="request-body"){
        let _=(q.method||"GET").toUpperCase();
        if(J6_.indexOf(_)!==-1){
          let z,Y=q.headers||{
            
          };
          for(let $ in Y)if($.toLowerCase()==="content-type"&&Y[$]){
            z=Y[$].toLowerCase();
            break
          }if(z==="application/x-www-form-urlencoded"){
            q.data=q.data||"";
            let $=gCq.parse(q.data);
            Object.assign($,{
              client_id:this.clientAuthentication.clientId,client_secret:this.clientAuthentication.clientSecret||""
            }),q.data=gCq.stringify($)
          }else if(z==="application/json")q.data=q.data||{
            
          },Object.assign(q.data,{
            client_id:this.clientAuthentication.clientId,client_secret:this.clientAuthentication.clientSecret||""
          });
          else throw Error(`${z} content-types are not supported with ${this.clientAuthentication.confidentialClientType} client authentication`)
        }else throw Error(`${_} HTTP method does not support ${this.clientAuthentication.confidentialClientType} client authentication`)
      }
    }static get RETRY_CONFIG(){
      return{
        retry:!0,retryConfig:{
          httpMethodsToRetry:["GET","PUT","POST","HEAD","OPTIONS","DELETE"]
        }
      }
    }
  }UCq.OAuthClientAuthHandler=FCq;
  function M6_(q,K){
    let{
      error:_,error_description:z,error_uri:Y
    }=q,$=`Error code ${_}`;
    if(typeof z<"u")$+=`: ${z}`;
    if(typeof Y<"u")$+=` - ${Y}`;
    let O=Error($);
    if(K){
      let A=Object.keys(K);
      if(K.stack)A.push("stack");
      A.forEach((w)=>{
        if(w!=="message")Object.defineProperty(O,w,{
          value:K[w],writable:!1,enumerable:!0
        })
      })
    }return O
  }
} /* confidence: 70% */

/* original: gCq */ var querystring_PUT_POST_PATCH=U6("querystring"),H6_=bG6(),J6_=["PUT","POST","PATCH"]; /* confidence: 65% */

/* original: $ */ let composed_value=gCq.parse(q.data); /* confidence: 30% */

/* original: Obq */ var AWS4HMACSHA256_aws4_request=bG6(),$bq="AWS4-HMAC-SHA256",x6_="aws4_request"; /* confidence: 65% */

/* original: QSq */ function u_u_u(){
  return typeof window<"u"&&typeof window.crypto<"u"&&typeof window.crypto.subtle<"u"
} /* confidence: 65% */

/* original: pf8 */ class u_SubtleCryptonotfoundMakesure{
  constructor(){
    if(typeof window>"u"||window.crypto===void 0||window.crypto.subtle===void 0)throw Error("SubtleCrypto not found. Make sure it's an https:// website.")
  }async sha256DigestBase64(q){
    let K=new TextEncoder().encode(q),_=await window.crypto.subtle.digest("SHA-256",K);
    return CG6.fromByteArray(new Uint8Array(_))
  }randomBytesBase64(q){
    let K=new Uint8Array(q);
    return window.crypto.getRandomValues(K),CG6.fromByteArray(K)
  }static padBase64(q){
    while(q.length%4!==0)q+="=";
    return q
  }async verify(q,K,_){
    let z={
      name:"RSASSA-PKCS1-v1_5",hash:{
        name:"SHA-256"
      }
    },Y=new TextEncoder().encode(K),$=CG6.toByteArray(u_SubtleCryptonotfoundMakesure.padBase64(_)),O=await window.crypto.subtle.importKey("jwk",q,z,!0,["verify"]);
    return await window.crypto.subtle.verify(z,O,$,Y)
  }async sign(q,K){
    let _={
      name:"RSASSA-PKCS1-v1_5",hash:{
        name:"SHA-256"
      }
    },z=new TextEncoder().encode(K),Y=await window.crypto.subtle.importKey("jwk",q,_,!0,["sign"]),$=await window.crypto.subtle.sign(_,Y,z);
    return CG6.fromByteArray(new Uint8Array($))
  }decodeBase64StringUtf8(q){
    let K=CG6.toByteArray(u_SubtleCryptonotfoundMakesure.padBase64(q));
    return new TextDecoder().decode(K)
  }encodeBase64StringUtf8(q){
    let K=new TextEncoder().encode(q);
    return CG6.fromByteArray(K)
  }async sha256DigestHex(q){
    let K=new TextEncoder().encode(q),_=await window.crypto.subtle.digest("SHA-256",K);
    return(0,je9.fromArrayBufferToHex)(_)
  }async signWithHmacSha256(q,K){
    let _=typeof q==="string"?q:String.fromCharCode(...new Uint16Array(q)),z=new TextEncoder,Y=await window.crypto.subtle.importKey("raw",z.encode(_),{
      name:"HMAC",hash:{
        name:"SHA-256"
      }
    },!1,["sign"]);
    return window.crypto.subtle.sign("HMAC",Y,z.encode(K))
  }
} /* confidence: 65% */

/* original: FCq */ class headers{
  constructor(q){
    this.clientAuthentication=q,this.crypto=(0,H6_.createCrypto)()
  }applyClientAuthenticationOptions(q,K){
    if(this.injectAuthenticatedHeaders(q,K),!K)this.injectAuthenticatedRequestBody(q)
  }injectAuthenticatedHeaders(q,K){
    var _;
    if(K)q.headers=q.headers||{
      
    },Object.assign(q.headers,{
      Authorization:`Bearer ${K}}`
    });
    else if(((_=this.clientAuthentication)===null||_===void 0?void 0:_.confidentialClientType)==="basic"){
      q.headers=q.headers||{
        
      };
      let z=this.clientAuthentication.clientId,Y=this.clientAuthentication.clientSecret||"",$=this.crypto.encodeBase64StringUtf8(`${z}:${Y}`);
      Object.assign(q.headers,{
        Authorization:`Basic ${$}`
      })
    }
  }injectAuthenticatedRequestBody(q){
    var K;
    if(((K=this.clientAuthentication)===null||K===void 0?void 0:K.confidentialClientType)==="request-body"){
      let _=(q.method||"GET").toUpperCase();
      if(J6_.indexOf(_)!==-1){
        let z,Y=q.headers||{
          
        };
        for(let $ in Y)if($.toLowerCase()==="content-type"&&Y[$]){
          z=Y[$].toLowerCase();
          break
        }if(z==="application/x-www-form-urlencoded"){
          q.data=q.data||"";
          let $=gCq.parse(q.data);
          Object.assign($,{
            client_id:this.clientAuthentication.clientId,client_secret:this.clientAuthentication.clientSecret||""
          }),q.data=gCq.stringify($)
        }else if(z==="application/json")q.data=q.data||{
          
        },Object.assign(q.data,{
          client_id:this.clientAuthentication.clientId,client_secret:this.clientAuthentication.clientSecret||""
        });
        else throw Error(`${z} content-types are not supported with ${this.clientAuthentication.confidentialClientType} client authentication`)
      }else throw Error(`${_} HTTP method does not support ${this.clientAuthentication.confidentialClientType} client authentication`)
    }
  }static get RETRY_CONFIG(){
    return{
      retry:!0,retryConfig:{
        httpMethodsToRetry:["GET","PUT","POST","HEAD","OPTIONS","DELETE"]
      }
    }
  }
} /* confidence: 70% */

