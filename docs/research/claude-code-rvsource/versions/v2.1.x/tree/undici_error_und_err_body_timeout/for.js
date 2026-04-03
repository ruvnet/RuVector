// Module: for

/* original: t$ */ var error_handler=B((lQ$,GI7)=>{
  var Rx7=Symbol.for("undici.error.UND_ERR");
  class DJ extends Error{
    constructor(q){
      super(q);
      this.name="UndiciError",this.code="UND_ERR"
    }static[Symbol.hasInstance](q){
      return q&&q[Rx7]===!0
    }[Rx7]=!0
  }var Sx7=Symbol.for("undici.error.UND_ERR_CONNECT_TIMEOUT");
  class tx7 extends DJ{
    constructor(q){
      super(q);
      this.name="ConnectTimeoutError",this.message=q||"Connect Timeout Error",this.code="UND_ERR_CONNECT_TIMEOUT"
    }static[Symbol.hasInstance](q){
      return q&&q[Sx7]===!0
    }[Sx7]=!0
  }var Cx7=Symbol.for("undici.error.UND_ERR_HEADERS_TIMEOUT");
  class ex7 extends DJ{
    constructor(q){
      super(q);
      this.name="HeadersTimeoutError",this.message=q||"Headers Timeout Error",this.code="UND_ERR_HEADERS_TIMEOUT"
    }static[Symbol.hasInstance](q){
      return q&&q[Cx7]===!0
    }[Cx7]=!0
  }var bx7=Symbol.for("undici.error.UND_ERR_HEADERS_OVERFLOW");
  class qI7 extends DJ{
    constructor(q){
      super(q);
      this.name="HeadersOverflowError",this.message=q||"Headers Overflow Error",this.code="UND_ERR_HEADERS_OVERFLOW"
    }static[Symbol.hasInstance](q){
      return q&&q[bx7]===!0
    }[bx7]=!0
  }var xx7=Symbol.for("undici.error.UND_ERR_BODY_TIMEOUT");
  class KI7 extends DJ{
    constructor(q){
      super(q);
      this.name="BodyTimeoutError",this.message=q||"Body Timeout Error",this.code="UND_ERR_BODY_TIMEOUT"
    }static[Symbol.hasInstance](q){
      return q&&q[xx7]===!0
    }[xx7]=!0
  }var Ix7=Symbol.for("undici.error.UND_ERR_RESPONSE_STATUS_CODE");
  class _I7 extends DJ{
    constructor(q,K,_,z){
      super(q);
      this.name="ResponseStatusCodeError",this.message=q||"Response Status Code Error",this.code="UND_ERR_RESPONSE_STATUS_CODE",this.body=z,this.status=K,this.statusCode=K,this.headers=_
    }static[Symbol.hasInstance](q){
      return q&&q[Ix7]===!0
    }[Ix7]=!0
  }var ux7=Symbol.for("undici.error.UND_ERR_INVALID_ARG");
  class zI7 extends DJ{
    constructor(q){
      super(q);
      this.name="InvalidArgumentError",this.message=q||"Invalid Argument Error",this.code="UND_ERR_INVALID_ARG"
    }static[Symbol.hasInstance](q){
      return q&&q[ux7]===!0
    }[ux7]=!0
  }var mx7=Symbol.for("undici.error.UND_ERR_INVALID_RETURN_VALUE");
  class YI7 extends DJ{
    constructor(q){
      super(q);
      this.name="InvalidReturnValueError",this.message=q||"Invalid Return Value Error",this.code="UND_ERR_INVALID_RETURN_VALUE"
    }static[Symbol.hasInstance](q){
      return q&&q[mx7]===!0
    }[mx7]=!0
  }var px7=Symbol.for("undici.error.UND_ERR_ABORT");
  class t31 extends DJ{
    constructor(q){
      super(q);
      this.name="AbortError",this.message=q||"The operation was aborted",this.code="UND_ERR_ABORT"
    }static[Symbol.hasInstance](q){
      return q&&q[px7]===!0
    }[px7]=!0
  }var Bx7=Symbol.for("undici.error.UND_ERR_ABORTED");
  class $I7 extends t31{
    constructor(q){
      super(q);
      this.name="AbortError",this.message=q||"Request aborted",this.code="UND_ERR_ABORTED"
    }static[Symbol.hasInstance](q){
      return q&&q[Bx7]===!0
    }[Bx7]=!0
  }var gx7=Symbol.for("undici.error.UND_ERR_INFO");
  class OI7 extends DJ{
    constructor(q){
      super(q);
      this.name="InformationalError",this.message=q||"Request information",this.code="UND_ERR_INFO"
    }static[Symbol.hasInstance](q){
      return q&&q[gx7]===!0
    }[gx7]=!0
  }var Fx7=Symbol.for("undici.error.UND_ERR_REQ_CONTENT_LENGTH_MISMATCH");
  class AI7 extends DJ{
    constructor(q){
      super(q);
      this.name="RequestContentLengthMismatchError",this.message=q||"Request body length does not match content-length header",this.code="UND_ERR_REQ_CONTENT_LENGTH_MISMATCH"
    }static[Symbol.hasInstance](q){
      return q&&q[Fx7]===!0
    }[Fx7]=!0
  }var Ux7=Symbol.for("undici.error.UND_ERR_RES_CONTENT_LENGTH_MISMATCH");
  class wI7 extends DJ{
    constructor(q){
      super(q);
      this.name="ResponseContentLengthMismatchError",this.message=q||"Response body length does not match content-length header",this.code="UND_ERR_RES_CONTENT_LENGTH_MISMATCH"
    }static[Symbol.hasInstance](q){
      return q&&q[Ux7]===!0
    }[Ux7]=!0
  }var Qx7=Symbol.for("undici.error.UND_ERR_DESTROYED");
  class jI7 extends DJ{
    constructor(q){
      super(q);
      this.name="ClientDestroyedError",this.message=q||"The client is destroyed",this.code="UND_ERR_DESTROYED"
    }static[Symbol.hasInstance](q){
      return q&&q[Qx7]===!0
    }[Qx7]=!0
  }var dx7=Symbol.for("undici.error.UND_ERR_CLOSED");
  class HI7 extends DJ{
    constructor(q){
      super(q);
      this.name="ClientClosedError",this.message=q||"The client is closed",this.code="UND_ERR_CLOSED"
    }static[Symbol.hasInstance](q){
      return q&&q[dx7]===!0
    }[dx7]=!0
  }var cx7=Symbol.for("undici.error.UND_ERR_SOCKET");
  class JI7 extends DJ{
    constructor(q,K){
      super(q);
      this.name="SocketError",this.message=q||"Socket error",this.code="UND_ERR_SOCKET",this.socket=K
    }static[Symbol.hasInstance](q){
      return q&&q[cx7]===!0
    }[cx7]=!0
  }var lx7=Symbol.for("undici.error.UND_ERR_NOT_SUPPORTED");
  class MI7 extends DJ{
    constructor(q){
      super(q);
      this.name="NotSupportedError",this.message=q||"Not supported error",this.code="UND_ERR_NOT_SUPPORTED"
    }static[Symbol.hasInstance](q){
      return q&&q[lx7]===!0
    }[lx7]=!0
  }var nx7=Symbol.for("undici.error.UND_ERR_BPL_MISSING_UPSTREAM");
  class XI7 extends DJ{
    constructor(q){
      super(q);
      this.name="MissingUpstreamError",this.message=q||"No upstream has been added to the BalancedPool",this.code="UND_ERR_BPL_MISSING_UPSTREAM"
    }static[Symbol.hasInstance](q){
      return q&&q[nx7]===!0
    }[nx7]=!0
  }var ix7=Symbol.for("undici.error.UND_ERR_HTTP_PARSER");
  class PI7 extends Error{
    constructor(q,K,_){
      super(q);
      this.name="HTTPParserError",this.code=K?`HPE_${K}`:void 0,this.data=_?_.toString():void 0
    }static[Symbol.hasInstance](q){
      return q&&q[ix7]===!0
    }[ix7]=!0
  }var rx7=Symbol.for("undici.error.UND_ERR_RES_EXCEEDED_MAX_SIZE");
  class WI7 extends DJ{
    constructor(q){
      super(q);
      this.name="ResponseExceededMaxSizeError",this.message=q||"Response content exceeded max size",this.code="UND_ERR_RES_EXCEEDED_MAX_SIZE"
    }static[Symbol.hasInstance](q){
      return q&&q[rx7]===!0
    }[rx7]=!0
  }var ox7=Symbol.for("undici.error.UND_ERR_REQ_RETRY");
  class DI7 extends DJ{
    constructor(q,K,{
      headers:_,data:z
    }){
      super(q);
      this.name="RequestRetryError",this.message=q||"Request retry error",this.code="UND_ERR_REQ_RETRY",this.statusCode=K,this.data=z,this.headers=_
    }static[Symbol.hasInstance](q){
      return q&&q[ox7]===!0
    }[ox7]=!0
  }var ax7=Symbol.for("undici.error.UND_ERR_RESPONSE");
  class fI7 extends DJ{
    constructor(q,K,{
      headers:_,data:z
    }){
      super(q);
      this.name="ResponseError",this.message=q||"Response error",this.code="UND_ERR_RESPONSE",this.statusCode=K,this.data=z,this.headers=_
    }static[Symbol.hasInstance](q){
      return q&&q[ax7]===!0
    }[ax7]=!0
  }var sx7=Symbol.for("undici.error.UND_ERR_PRX_TLS");
  class ZI7 extends DJ{
    constructor(q,K,_){
      super(K,{
        cause:q,..._??{
          
        }
      });
      this.name="SecureProxyConnectionError",this.message=K||"Secure Proxy Connection failed",this.code="UND_ERR_PRX_TLS",this.cause=q
    }static[Symbol.hasInstance](q){
      return q&&q[sx7]===!0
    }[sx7]=!0
  }GI7.exports={
    AbortError:t31,HTTPParserError:PI7,UndiciError:DJ,HeadersTimeoutError:ex7,HeadersOverflowError:qI7,BodyTimeoutError:KI7,RequestContentLengthMismatchError:AI7,ConnectTimeoutError:tx7,ResponseStatusCodeError:_I7,InvalidArgumentError:zI7,InvalidReturnValueError:YI7,RequestAbortedError:$I7,ClientDestroyedError:jI7,ClientClosedError:HI7,InformationalError:OI7,SocketError:JI7,NotSupportedError:MI7,ResponseContentLengthMismatchError:wI7,BalancedPoolMissingUpstreamError:XI7,ResponseExceededMaxSizeError:WI7,RequestRetryError:DI7,ResponseError:fI7,SecureProxyConnectionError:ZI7
  }
} /* confidence: 95% */

/* original: xx7 */ var xx7=Symbol.for("undici.error.UND_ERR_BODY_TIMEOUT");

/* original: E_1 */ var error_handler=B((Bd$,fg7)=>{
  var{
    UndiciError:fQ5
  }=t$(),Dg7=Symbol.for("undici.error.UND_MOCK_ERR_MOCK_NOT_MATCHED");
  class y_1 extends fQ5{
    constructor(q){
      super(q);
      Error.captureStackTrace(this,y_1),this.name="MockNotMatchedError",this.message=q||"The request does not match any registered mock dispatches",this.code="UND_MOCK_ERR_MOCK_NOT_MATCHED"
    }static[Symbol.hasInstance](q){
      return q&&q[Dg7]===!0
    }[Dg7]=!0
  }fg7.exports={
    MockNotMatchedError:y_1
  }
} /* confidence: 95% */

