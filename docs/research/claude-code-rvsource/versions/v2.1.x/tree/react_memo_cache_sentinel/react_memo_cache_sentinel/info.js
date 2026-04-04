// Module: info

/* original: K2 */ var auth_handler=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */Q1={
    LIBRARY_NAME:"MSAL.JS",SKU:"msal.js.common",DEFAULT_AUTHORITY:"https://login.microsoftonline.com/common/",DEFAULT_AUTHORITY_HOST:"login.microsoftonline.com",DEFAULT_COMMON_TENANT:"common",ADFS:"adfs",DSTS:"dstsv2",AAD_INSTANCE_DISCOVERY_ENDPT:"https://login.microsoftonline.com/common/discovery/instance?api-version=1.1&authorization_endpoint=",CIAM_AUTH_URL:".ciamlogin.com",AAD_TENANT_DOMAIN_SUFFIX:".onmicrosoft.com",RESOURCE_DELIM:"|",NO_ACCOUNT:"NO_ACCOUNT",CLAIMS:"claims",CONSUMER_UTID:"9188040d-6c67-4c5b-b112-36a304b66dad",OPENID_SCOPE:"openid",PROFILE_SCOPE:"profile",OFFLINE_ACCESS_SCOPE:"offline_access",EMAIL_SCOPE:"email",CODE_GRANT_TYPE:"authorization_code",RT_GRANT_TYPE:"refresh_token",S256_CODE_CHALLENGE_METHOD:"S256",URL_FORM_CONTENT_TYPE:"application/x-www-form-urlencoded;charset=utf-8",AUTHORIZATION_PENDING:"authorization_pending",NOT_DEFINED:"not_defined",EMPTY_STRING:"",NOT_APPLICABLE:"N/A",NOT_AVAILABLE:"Not Available",FORWARD_SLASH:"/",IMDS_ENDPOINT:"http://169.254.169.254/metadata/instance/compute/location",IMDS_VERSION:"2020-06-01",IMDS_TIMEOUT:2000,AZURE_REGION_AUTO_DISCOVER_FLAG:"TryAutoDetect",REGIONAL_AUTH_PUBLIC_CLOUD_SUFFIX:"login.microsoft.com",KNOWN_PUBLIC_CLOUDS:["login.microsoftonline.com","login.windows.net","login.microsoft.com","sts.windows.net"],SHR_NONCE_VALIDITY:240,INVALID_INSTANCE:"invalid_instance"
  },K9={
    SUCCESS:200,SUCCESS_RANGE_START:200,SUCCESS_RANGE_END:299,REDIRECT:302,CLIENT_ERROR:400,CLIENT_ERROR_RANGE_START:400,BAD_REQUEST:400,UNAUTHORIZED:401,NOT_FOUND:404,REQUEST_TIMEOUT:408,GONE:410,TOO_MANY_REQUESTS:429,CLIENT_ERROR_RANGE_END:499,SERVER_ERROR:500,SERVER_ERROR_RANGE_START:500,SERVICE_UNAVAILABLE:503,GATEWAY_TIMEOUT:504,SERVER_ERROR_RANGE_END:599,MULTI_SIDED_ERROR:600
  },SG=[Q1.OPENID_SCOPE,Q1.PROFILE_SCOPE,Q1.OFFLINE_ACCESS_SCOPE],uZ1=[...SG,Q1.EMAIL_SCOPE],q2={
    CONTENT_TYPE:"Content-Type",CONTENT_LENGTH:"Content-Length",RETRY_AFTER:"Retry-After",CCS_HEADER:"X-AnchorMailbox",WWWAuthenticate:"WWW-Authenticate",AuthenticationInfo:"Authentication-Info",X_MS_REQUEST_ID:"x-ms-request-id",X_MS_HTTP_VERSION:"x-ms-httpver"
  },$N={
    COMMON:"common",ORGANIZATIONS:"organizations",CONSUMERS:"consumers"
  },pY6={
    ACCESS_TOKEN:"access_token",XMS_CC:"xms_cc"
  },r86={
    LOGIN:"login",SELECT_ACCOUNT:"select_account",CONSENT:"consent",NONE:"none",CREATE:"create",NO_SESSION:"no_session"
  },WW8={
    PLAIN:"plain",S256:"S256"
  },B06={
    CODE:"code",IDTOKEN_TOKEN:"id_token token",IDTOKEN_TOKEN_REFRESHTOKEN:"id_token token refresh_token"
  },DF={
    QUERY:"query",FRAGMENT:"fragment",FORM_POST:"form_post"
  },ON={
    IMPLICIT_GRANT:"implicit",AUTHORIZATION_CODE_GRANT:"authorization_code",CLIENT_CREDENTIALS_GRANT:"client_credentials",RESOURCE_OWNER_PASSWORD_GRANT:"password",REFRESH_TOKEN_GRANT:"refresh_token",DEVICE_CODE_GRANT:"device_code",JWT_BEARER:"urn:ietf:params:oauth:grant-type:jwt-bearer"
  },BY6={
    MSSTS_ACCOUNT_TYPE:"MSSTS",ADFS_ACCOUNT_TYPE:"ADFS",MSAV1_ACCOUNT_TYPE:"MSA",GENERIC_ACCOUNT_TYPE:"Generic"
  },Xi={
    CACHE_KEY_SEPARATOR:"-",CLIENT_INFO_SEPARATOR:"."
  },xO={
    ID_TOKEN:"IdToken",ACCESS_TOKEN:"AccessToken",ACCESS_TOKEN_WITH_AUTH_SCHEME:"AccessToken_With_AuthScheme",REFRESH_TOKEN:"RefreshToken"
  },g06={
    CACHE_KEY:"authority-metadata",REFRESH_TIME_SECONDS:86400
  },RT={
    CONFIG:"config",CACHE:"cache",NETWORK:"network",HARDCODED_VALUES:"hardcoded_values"
  },gP={
    SCHEMA_VERSION:5,MAX_LAST_HEADER_BYTES:330,MAX_CACHED_ERRORS:50,CACHE_KEY:"server-telemetry",CATEGORY_SEPARATOR:"|",VALUE_SEPARATOR:",",OVERFLOW_TRUE:"1",OVERFLOW_FALSE:"0",UNKNOWN_ERROR:"unknown_error"
  },Vz={
    BEARER:"Bearer",POP:"pop",SSH:"ssh-cert"
  },fF={
    DEFAULT_THROTTLE_TIME_SECONDS:60,DEFAULT_MAX_THROTTLE_TIME_SECONDS:3600,THROTTLING_PREFIX:"throttling",X_MS_LIB_CAPABILITY_VALUE:"retry-after, h429"
  },lQ6={
    INVALID_GRANT_ERROR:"invalid_grant",CLIENT_MISMATCH_ERROR:"client_mismatch"
  },nQ6={
    username:"username",password:"password"
  },gY6={
    FAILED_AUTO_DETECTION:"1",INTERNAL_CACHE:"2",ENVIRONMENT_VARIABLE:"3",IMDS:"4"
  },DW8={
    CONFIGURED_NO_AUTO_DETECTION:"2",AUTO_DETECTION_REQUESTED_SUCCESSFUL:"4",AUTO_DETECTION_REQUESTED_FAILED:"5"
  },fw={
    NOT_APPLICABLE:"0",FORCE_REFRESH_OR_CLAIMS:"1",NO_CACHED_ACCESS_TOKEN:"2",CACHED_ACCESS_TOKEN_EXPIRED:"3",PROACTIVELY_REFRESHED:"4"
  },jZ={
    BASE64:"base64",HEX:"hex",UTF8:"utf-8"
  }
} /* confidence: 95% */

/* original: mZ1 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: pS */ var error_handler=L(()=>{
  K2();
  mZ1();
  /*! @azure/msal-common v15.13.1 2025-10-29 */fW8={
    [iQ6]:"Unexpected error in authentication.",[rQ6]:"Post request failed from the network, could be a 4xx/5xx or a network unavailability. Please check the exact error code for details."
  },pZ1={
    unexpectedError:{
      code:iQ6,desc:fW8[iQ6]
    },postRequestFailed:{
      code:rQ6,desc:fW8[rQ6]
    }
  };
  _9=class _9 extends Error{
    constructor(q,K,_){
      let z=K?`${q}: ${K}`:q;
      super(z);
      Object.setPrototypeOf(this,_9.prototype),this.errorCode=q||Q1.EMPTY_STRING,this.errorMessage=K||Q1.EMPTY_STRING,this.subError=_||Q1.EMPTY_STRING,this.name="AuthError"
    }setCorrelationId(q){
      this.correlationId=q
    }
  }
} /* confidence: 95% */

/* original: wM */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: hX */ var config=L(()=>{
  pS();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */z9={
    [a86]:"The client info could not be parsed/decoded correctly",[FY6]:"The client info was empty",[s86]:"Token cannot be parsed",[UY6]:"The token is null or empty",[ST]:"Endpoints cannot be resolved",[QY6]:"Network request failed",[dY6]:"Could not retrieve endpoints. Check your authority and verify the .well-known/openid-configuration endpoint returns the required endpoints.",[cY6]:"The hash parameters could not be deserialized",[Pu]:"State was not the expected format",[lY6]:"State mismatch error",[t86]:"State not found",[nY6]:"Nonce mismatch error",[Pi]:"Max Age was requested and the ID token is missing the auth_time variable. auth_time is an optional claim and is not enabled by default - it must be enabled. See https://aka.ms/msaljs/optional-claims for more information.",[iY6]:"Max Age is set to 0, or too much time has elapsed since the last end-user authentication.",[oQ6]:"The cache contains multiple tokens satisfying the requirements. Call AcquireToken again providing more requirements such as authority or account.",[aQ6]:"The cache contains multiple accounts satisfying the given parameters. Please pass more info to obtain the correct account",[rY6]:"The cache contains multiple appMetadata satisfying the given parameters. Please pass more info to obtain the correct appMetadata",[oY6]:"Token request cannot be made without authorization code or refresh token.",[aY6]:"Cannot remove null or empty scope from ScopeSet",[sY6]:"Cannot append ScopeSet",[e86]:"Empty input ScopeSet cannot be processed",[sQ6]:"Caller has cancelled token endpoint polling during device code flow by setting DeviceCodeRequest.cancel = true.",[tQ6]:"Device code is expired.",[eQ6]:"Device code stopped polling for unknown reasons.",[Wi]:"Please pass an account object, silent flow is not supported without account information",[tY6]:"Cache record object was null or undefined.",[Di]:"Invalid environment when attempting to create cache entry",[qd6]:"No account found in cache for given key.",[q16]:"No crypto object detected.",[Kd6]:"Unexpected credential type.",[_d6]:"Client assertion must meet requirements described in https://tools.ietf.org/html/rfc7515",[zd6]:"Client credential (secret, certificate, or assertion) must not be empty when creating a confidential client. An application should at most have one credential",[fi]:"Cannot return token from cache because it must be refreshed. This may be due to one of the following reasons: forceRefresh parameter is set to true, claims have been requested, there is no cached access token or it is expired.",[Yd6]:"User defined timeout for device code polling reached",[eY6]:"Cannot generate a POP jwt if the token_claims are not populated",[q$6]:"Server response does not contain an authorization code to proceed",[$d6]:"Could not remove the credential's binding key from storage.",[K$6]:"The provided authority does not support logout",[_$6]:"A keyId value is missing from the requested bound token's cache record and is required to match the token to it's stored binding key.",[Od6]:"No network connectivity. Check your internet connection.",[Ad6]:"User cancelled the flow.",[wd6]:"A tenant id - not common, organizations, or consumers - must be specified when using the client_credentials flow.",[D_]:"This method has not been implemented",[jd6]:"The nested app auth bridge is disabled"
  },gZ1={
    clientInfoDecodingError:{
      code:a86,desc:z9[a86]
    },clientInfoEmptyError:{
      code:FY6,desc:z9[FY6]
    },tokenParsingError:{
      code:s86,desc:z9[s86]
    },nullOrEmptyToken:{
      code:UY6,desc:z9[UY6]
    },endpointResolutionError:{
      code:ST,desc:z9[ST]
    },networkError:{
      code:QY6,desc:z9[QY6]
    },unableToGetOpenidConfigError:{
      code:dY6,desc:z9[dY6]
    },hashNotDeserialized:{
      code:cY6,desc:z9[cY6]
    },invalidStateError:{
      code:Pu,desc:z9[Pu]
    },stateMismatchError:{
      code:lY6,desc:z9[lY6]
    },stateNotFoundError:{
      code:t86,desc:z9[t86]
    },nonceMismatchError:{
      code:nY6,desc:z9[nY6]
    },authTimeNotFoundError:{
      code:Pi,desc:z9[Pi]
    },maxAgeTranspired:{
      code:iY6,desc:z9[iY6]
    },multipleMatchingTokens:{
      code:oQ6,desc:z9[oQ6]
    },multipleMatchingAccounts:{
      code:aQ6,desc:z9[aQ6]
    },multipleMatchingAppMetadata:{
      code:rY6,desc:z9[rY6]
    },tokenRequestCannotBeMade:{
      code:oY6,desc:z9[oY6]
    },removeEmptyScopeError:{
      code:aY6,desc:z9[aY6]
    },appendScopeSetError:{
      code:sY6,desc:z9[sY6]
    },emptyInputScopeSetError:{
      code:e86,desc:z9[e86]
    },DeviceCodePollingCancelled:{
      code:sQ6,desc:z9[sQ6]
    },DeviceCodeExpired:{
      code:tQ6,desc:z9[tQ6]
    },DeviceCodeUnknownError:{
      code:eQ6,desc:z9[eQ6]
    },NoAccountInSilentRequest:{
      code:Wi,desc:z9[Wi]
    },invalidCacheRecord:{
      code:tY6,desc:z9[tY6]
    },invalidCacheEnvironment:{
      code:Di,desc:z9[Di]
    },noAccountFound:{
      code:qd6,desc:z9[qd6]
    },noCryptoObj:{
      code:q16,desc:z9[q16]
    },unexpectedCredentialType:{
      code:Kd6,desc:z9[Kd6]
    },invalidAssertion:{
      code:_d6,desc:z9[_d6]
    },invalidClientCredential:{
      code:zd6,desc:z9[zd6]
    },tokenRefreshRequired:{
      code:fi,desc:z9[fi]
    },userTimeoutReached:{
      code:Yd6,desc:z9[Yd6]
    },tokenClaimsRequired:{
      code:eY6,desc:z9[eY6]
    },noAuthorizationCodeFromServer:{
      code:q$6,desc:z9[q$6]
    },bindingKeyNotRemovedError:{
      code:$d6,desc:z9[$d6]
    },logoutNotSupported:{
      code:K$6,desc:z9[K$6]
    },keyIdMissing:{
      code:_$6,desc:z9[_$6]
    },noNetworkConnectivity:{
      code:Od6,desc:z9[Od6]
    },userCanceledError:{
      code:Ad6,desc:z9[Ad6]
    },missingTenantIdError:{
      code:wd6,desc:z9[wd6]
    },nestedAppAuthBridgeDisabled:{
      code:jd6,desc:z9[jd6]
    }
  };
  K16=class K16 extends _9{
    constructor(q,K){
      super(q,K?`${z9[q]}: ${K}`:z9[q]);
      this.name="ClientAuthError",Object.setPrototypeOf(this,K16.prototype)
    }
  }
} /* confidence: 95% */

/* original: FZ1 */ var composed_value=L(()=>{
  hX();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */Q06={
    createNewGuid:()=>{
      throw J7(D_)
    },base64Decode:()=>{
      throw J7(D_)
    },base64Encode:()=>{
      throw J7(D_)
    },base64UrlEncode:()=>{
      throw J7(D_)
    },encodeKid:()=>{
      throw J7(D_)
    },async getPublicKeyThumbprint(){
      throw J7(D_)
    },async removeTokenBindingKey(){
      throw J7(D_)
    },async clearKeystore(){
      throw J7(D_)
    },async signJwt(){
      throw J7(D_)
    },async hashString(){
      throw J7(D_)
    }
  }
} /* confidence: 30% */

/* original: ZW8 */ var error_handler=L(()=>{
  K2();
  /*! @azure/msal-common v15.13.1 2025-10-29 */(function(q){
    q[q.Error=0]="Error",q[q.Warning=1]="Warning",q[q.Info=2]="Info",q[q.Verbose=3]="Verbose",q[q.Trace=4]="Trace"
  })(vH||(vH={
    
  }))
} /* confidence: 95% */

/* original: vW8 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: TW8 */ var none_httpsloginmicrosoftonline=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */Zi={
    None:"none",AzurePublic:"https://login.microsoftonline.com",AzurePpe:"https://login.windows-ppe.net",AzureChina:"https://login.chinacloudapi.cn",AzureGermany:"https://login.microsoftonline.de",AzureUsGovernment:"https://login.microsoftonline.us"
  }
} /* confidence: 65% */

/* original: vi */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: Y16 */ var config=L(()=>{
  pS();
  vi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */Nj={
    [z$6]:"A redirect URI is required for all calls, and none has been set.",[Hd6]:"Could not parse the given claims request object.",[Y$6]:"Authority URIs must use https.  Please see here for valid authority configuration options: https://docs.microsoft.com/en-us/azure/active-directory/develop/msal-js-initializing-client-applications#configuration-options",[ZF]:"URL could not be parsed into appropriate segments.",[$$6]:"URL was empty or null.",[O$6]:"Scopes cannot be passed as null, undefined or empty array because they are required to obtain an access token.",[_16]:"Given claims parameter must be a stringified JSON object.",[A$6]:"Token request was empty and not found in cache.",[w$6]:"The logout request was null or undefined.",[Jd6]:'code_challenge_method passed is invalid. Valid values are "plain" and "S256".',[j$6]:"Both params: code_challenge and code_challenge_method are to be passed if to be sent in the request",[z16]:"Invalid cloudDiscoveryMetadata provided. Must be a stringified JSON object containing tenant_discovery_endpoint and metadata fields",[H$6]:"Invalid authorityMetadata provided. Must by a stringified JSON object containing authorization_endpoint, token_endpoint, issuer fields.",[J$6]:"The provided authority is not a trusted authority. Please include this authority in the knownAuthorities config parameter.",[Gi]:"Missing sshJwk in SSH certificate request. A stringified JSON Web Key is required when using the SSH authentication scheme.",[Md6]:"Missing sshKid in SSH certificate request. A string that uniquely identifies the public SSH key is required when using the SSH authentication scheme.",[Xd6]:"Unable to find an authentication header containing server nonce. Either the Authentication-Info or WWW-Authenticate headers must be present in order to obtain a server nonce.",[Pd6]:"Invalid authentication header provided",[Wd6]:"Cannot set OIDCOptions parameter. Please change the protocol mode to OIDC or use a non-Microsoft authority.",[Dd6]:"Cannot set allowPlatformBroker parameter to true when not in AAD protocol mode.",[fd6]:"Authority mismatch error. Authority provided in login request or PublicClientApplication config does not match the environment of the provided account. Please use a matching account or make an interactive request to login to this authority.",[Gd6]:"Invalid authorize post body parameters provided. If you are using authorizePostBodyParameters, the request method must be POST. Please check the request method and parameters.",[Zd6]:"Invalid request method for EAR protocol mode. The request method cannot be GET when using EAR protocol mode. Please change the request method to POST."
  },UZ1={
    redirectUriNotSet:{
      code:z$6,desc:Nj[z$6]
    },claimsRequestParsingError:{
      code:Hd6,desc:Nj[Hd6]
    },authorityUriInsecure:{
      code:Y$6,desc:Nj[Y$6]
    },urlParseError:{
      code:ZF,desc:Nj[ZF]
    },urlEmptyError:{
      code:$$6,desc:Nj[$$6]
    },emptyScopesError:{
      code:O$6,desc:Nj[O$6]
    },invalidClaimsRequest:{
      code:_16,desc:Nj[_16]
    },tokenRequestEmptyError:{
      code:A$6,desc:Nj[A$6]
    },logoutRequestEmptyError:{
      code:w$6,desc:Nj[w$6]
    },invalidCodeChallengeMethod:{
      code:Jd6,desc:Nj[Jd6]
    },invalidCodeChallengeParams:{
      code:j$6,desc:Nj[j$6]
    },invalidCloudDiscoveryMetadata:{
      code:z16,desc:Nj[z16]
    },invalidAuthorityMetadata:{
      code:H$6,desc:Nj[H$6]
    },untrustedAuthority:{
      code:J$6,desc:Nj[J$6]
    },missingSshJwk:{
      code:Gi,desc:Nj[Gi]
    },missingSshKid:{
      code:Md6,desc:Nj[Md6]
    },missingNonceAuthenticationHeader:{
      code:Xd6,desc:Nj[Xd6]
    },invalidAuthenticationHeader:{
      code:Pd6,desc:Nj[Pd6]
    },cannotSetOIDCOptions:{
      code:Wd6,desc:Nj[Wd6]
    },cannotAllowPlatformBroker:{
      code:Dd6,desc:Nj[Dd6]
    },authorityMismatch:{
      code:fd6,desc:Nj[fd6]
    },invalidAuthorizePostBodyParameters:{
      code:Gd6,desc:Nj[Gd6]
    },invalidRequestMethodForEAR:{
      code:Zd6,desc:Nj[Zd6]
    }
  };
  l06=class l06 extends _9{
    constructor(q){
      super(q,Nj[q]);
      this.name="ClientConfigurationError",Object.setPrototypeOf(this,l06.prototype)
    }
  }
} /* confidence: 95% */

/* original: $16 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: vd6 */ var composed_value=L(()=>{
  Y16();
  $16();
  hX();
  K2();
  vi();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: i06 */ var composed_value=L(()=>{
  hX();
  K2();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: VW8 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: QZ1 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */BS={
    Default:0,Adfs:1,Dsts:2,Ciam:3
  }
} /* confidence: 30% */

/* original: dZ1 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: kd6 */ var AAD_OIDC_EAR=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */CG={
    AAD:"AAD",OIDC:"OIDC",EAR:"EAR"
  }
} /* confidence: 65% */

/* original: yW8 */ var composed_value=L(()=>{
  K2();
  i06();
  VW8();
  hX();
  QZ1();
  dZ1();
  kd6();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: r06 */ var composed_value=L(()=>{
  hX();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: M$6 */ var composed_value=L(()=>{
  hX();
  $16();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: O16 */ var composed_value=L(()=>{
  Y16();
  $16();
  K2();
  M$6();
  vi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: oZ1 */ var loginmicrosoftonlinecom_httpsl=L(()=>{
  O16();
  K2();
  /*! @azure/msal-common v15.13.1 2025-10-29 */RGq={
    endpointMetadata:{
      "login.microsoftonline.com":{
        token_endpoint:"https://login.microsoftonline.com/{tenantid}/oauth2/v2.0/token",jwks_uri:"https://login.microsoftonline.com/{tenantid}/discovery/v2.0/keys",issuer:"https://login.microsoftonline.com/{tenantid}/v2.0",authorization_endpoint:"https://login.microsoftonline.com/{tenantid}/oauth2/v2.0/authorize",end_session_endpoint:"https://login.microsoftonline.com/{tenantid}/oauth2/v2.0/logout"
      },"login.chinacloudapi.cn":{
        token_endpoint:"https://login.chinacloudapi.cn/{tenantid}/oauth2/v2.0/token",jwks_uri:"https://login.chinacloudapi.cn/{tenantid}/discovery/v2.0/keys",issuer:"https://login.partner.microsoftonline.cn/{tenantid}/v2.0",authorization_endpoint:"https://login.chinacloudapi.cn/{tenantid}/oauth2/v2.0/authorize",end_session_endpoint:"https://login.chinacloudapi.cn/{tenantid}/oauth2/v2.0/logout"
      },"login.microsoftonline.us":{
        token_endpoint:"https://login.microsoftonline.us/{tenantid}/oauth2/v2.0/token",jwks_uri:"https://login.microsoftonline.us/{tenantid}/discovery/v2.0/keys",issuer:"https://login.microsoftonline.us/{tenantid}/v2.0",authorization_endpoint:"https://login.microsoftonline.us/{tenantid}/oauth2/v2.0/authorize",end_session_endpoint:"https://login.microsoftonline.us/{tenantid}/oauth2/v2.0/logout"
      }
    },instanceDiscoveryMetadata:{
      metadata:[{
        preferred_network:"login.microsoftonline.com",preferred_cache:"login.windows.net",aliases:["login.microsoftonline.com","login.windows.net","login.microsoft.com","sts.windows.net"]
      },{
        preferred_network:"login.partner.microsoftonline.cn",preferred_cache:"login.partner.microsoftonline.cn",aliases:["login.partner.microsoftonline.cn","login.chinacloudapi.cn"]
      },{
        preferred_network:"login.microsoftonline.de",preferred_cache:"login.microsoftonline.de",aliases:["login.microsoftonline.de"]
      },{
        preferred_network:"login.microsoftonline.us",preferred_cache:"login.microsoftonline.us",aliases:["login.microsoftonline.us","login.usgovcloudapi.net"]
      },{
        preferred_network:"login-us.microsoftonline.com",preferred_cache:"login-us.microsoftonline.com",aliases:["login-us.microsoftonline.com"]
      }]
    }
  },nZ1=RGq.endpointMetadata,iZ1=RGq.instanceDiscoveryMetadata,rZ1=new Set;
  iZ1.metadata.forEach((q)=>{
    q.aliases.forEach((K)=>{
      rZ1.add(K)
    })
  })
} /* confidence: 65% */

/* original: bGq */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: IGq */ var error_handler=L(()=>{
  pS();
  bGq();
  /*! @azure/msal-common v15.13.1 2025-10-29 */sZ1={
    [aZ1]:"Exceeded cache storage capacity.",[LW8]:"Unexpected error occurred when using cache storage."
  };
  yd6=class yd6 extends _9{
    constructor(q,K){
      let _=K||(sZ1[q]?sZ1[q]:sZ1[LW8]);
      super(`${q}: ${_}`);
      Object.setPrototypeOf(this,yd6.prototype),this.name="CacheError",this.errorCode=q,this.errorMessage=_
    }
  }
} /* confidence: 95% */

/* original: tZ1 */ var composed_value=L(()=>{
  K2();
  vd6();
  yW8();
  hX();
  VW8();
  r06();
  vW8();
  oZ1();
  IGq();
  pS();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */hW8=class hW8 extends X$6{
    async setAccount(){
      throw J7(D_)
    }getAccount(){
      throw J7(D_)
    }async setIdTokenCredential(){
      throw J7(D_)
    }getIdTokenCredential(){
      throw J7(D_)
    }async setAccessTokenCredential(){
      throw J7(D_)
    }getAccessTokenCredential(){
      throw J7(D_)
    }async setRefreshTokenCredential(){
      throw J7(D_)
    }getRefreshTokenCredential(){
      throw J7(D_)
    }setAppMetadata(){
      throw J7(D_)
    }getAppMetadata(){
      throw J7(D_)
    }setServerTelemetry(){
      throw J7(D_)
    }getServerTelemetry(){
      throw J7(D_)
    }setAuthorityMetadata(){
      throw J7(D_)
    }getAuthorityMetadata(){
      throw J7(D_)
    }getAuthorityMetadataKeys(){
      throw J7(D_)
    }setThrottlingCache(){
      throw J7(D_)
    }getThrottlingCache(){
      throw J7(D_)
    }removeItem(){
      throw J7(D_)
    }getKeys(){
      throw J7(D_)
    }getAccountKeys(){
      throw J7(D_)
    }getTokenKeys(){
      throw J7(D_)
    }generateCredentialKey(){
      throw J7(D_)
    }generateAccountKey(){
      throw J7(D_)
    }
  }
} /* confidence: 30% */

/* original: Wu */ var action_creator=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */R1={
    AcquireTokenByCode:"acquireTokenByCode",AcquireTokenByRefreshToken:"acquireTokenByRefreshToken",AcquireTokenSilent:"acquireTokenSilent",AcquireTokenSilentAsync:"acquireTokenSilentAsync",AcquireTokenPopup:"acquireTokenPopup",AcquireTokenPreRedirect:"acquireTokenPreRedirect",AcquireTokenRedirect:"acquireTokenRedirect",CryptoOptsGetPublicKeyThumbprint:"cryptoOptsGetPublicKeyThumbprint",CryptoOptsSignJwt:"cryptoOptsSignJwt",SilentCacheClientAcquireToken:"silentCacheClientAcquireToken",SilentIframeClientAcquireToken:"silentIframeClientAcquireToken",AwaitConcurrentIframe:"awaitConcurrentIframe",SilentRefreshClientAcquireToken:"silentRefreshClientAcquireToken",SsoSilent:"ssoSilent",StandardInteractionClientGetDiscoveredAuthority:"standardInteractionClientGetDiscoveredAuthority",FetchAccountIdWithNativeBroker:"fetchAccountIdWithNativeBroker",NativeInteractionClientAcquireToken:"nativeInteractionClientAcquireToken",BaseClientCreateTokenRequestHeaders:"baseClientCreateTokenRequestHeaders",NetworkClientSendPostRequestAsync:"networkClientSendPostRequestAsync",RefreshTokenClientExecutePostToTokenEndpoint:"refreshTokenClientExecutePostToTokenEndpoint",AuthorizationCodeClientExecutePostToTokenEndpoint:"authorizationCodeClientExecutePostToTokenEndpoint",BrokerHandhshake:"brokerHandshake",AcquireTokenByRefreshTokenInBroker:"acquireTokenByRefreshTokenInBroker",AcquireTokenByBroker:"acquireTokenByBroker",RefreshTokenClientExecuteTokenRequest:"refreshTokenClientExecuteTokenRequest",RefreshTokenClientAcquireToken:"refreshTokenClientAcquireToken",RefreshTokenClientAcquireTokenWithCachedRefreshToken:"refreshTokenClientAcquireTokenWithCachedRefreshToken",RefreshTokenClientAcquireTokenByRefreshToken:"refreshTokenClientAcquireTokenByRefreshToken",RefreshTokenClientCreateTokenRequestBody:"refreshTokenClientCreateTokenRequestBody",AcquireTokenFromCache:"acquireTokenFromCache",SilentFlowClientAcquireCachedToken:"silentFlowClientAcquireCachedToken",SilentFlowClientGenerateResultFromCacheRecord:"silentFlowClientGenerateResultFromCacheRecord",AcquireTokenBySilentIframe:"acquireTokenBySilentIframe",InitializeBaseRequest:"initializeBaseRequest",InitializeSilentRequest:"initializeSilentRequest",InitializeClientApplication:"initializeClientApplication",InitializeCache:"initializeCache",SilentIframeClientTokenHelper:"silentIframeClientTokenHelper",SilentHandlerInitiateAuthRequest:"silentHandlerInitiateAuthRequest",SilentHandlerMonitorIframeForHash:"silentHandlerMonitorIframeForHash",SilentHandlerLoadFrame:"silentHandlerLoadFrame",SilentHandlerLoadFrameSync:"silentHandlerLoadFrameSync",StandardInteractionClientCreateAuthCodeClient:"standardInteractionClientCreateAuthCodeClient",StandardInteractionClientGetClientConfiguration:"standardInteractionClientGetClientConfiguration",StandardInteractionClientInitializeAuthorizationRequest:"standardInteractionClientInitializeAuthorizationRequest",GetAuthCodeUrl:"getAuthCodeUrl",GetStandardParams:"getStandardParams",HandleCodeResponseFromServer:"handleCodeResponseFromServer",HandleCodeResponse:"handleCodeResponse",HandleResponseEar:"handleResponseEar",HandleResponsePlatformBroker:"handleResponsePlatformBroker",HandleResponseCode:"handleResponseCode",UpdateTokenEndpointAuthority:"updateTokenEndpointAuthority",AuthClientAcquireToken:"authClientAcquireToken",AuthClientExecuteTokenRequest:"authClientExecuteTokenRequest",AuthClientCreateTokenRequestBody:"authClientCreateTokenRequestBody",PopTokenGenerateCnf:"popTokenGenerateCnf",PopTokenGenerateKid:"popTokenGenerateKid",HandleServerTokenResponse:"handleServerTokenResponse",DeserializeResponse:"deserializeResponse",AuthorityFactoryCreateDiscoveredInstance:"authorityFactoryCreateDiscoveredInstance",AuthorityResolveEndpointsAsync:"authorityResolveEndpointsAsync",AuthorityResolveEndpointsFromLocalSources:"authorityResolveEndpointsFromLocalSources",AuthorityGetCloudDiscoveryMetadataFromNetwork:"authorityGetCloudDiscoveryMetadataFromNetwork",AuthorityUpdateCloudDiscoveryMetadata:"authorityUpdateCloudDiscoveryMetadata",AuthorityGetEndpointMetadataFromNetwork:"authorityGetEndpointMetadataFromNetwork",AuthorityUpdateEndpointMetadata:"authorityUpdateEndpointMetadata",AuthorityUpdateMetadataWithRegionalInformation:"authorityUpdateMetadataWithRegionalInformation",RegionDiscoveryDetectRegion:"regionDiscoveryDetectRegion",RegionDiscoveryGetRegionFromIMDS:"regionDiscoveryGetRegionFromIMDS",RegionDiscoveryGetCurrentVersion:"regionDiscoveryGetCurrentVersion",AcquireTokenByCodeAsync:"acquireTokenByCodeAsync",GetEndpointMetadataFromNetwork:"getEndpointMetadataFromNetwork",GetCloudDiscoveryMetadataFromNetworkMeasurement:"getCloudDiscoveryMetadataFromNetworkMeasurement",HandleRedirectPromiseMeasurement:"handleRedirectPromise",HandleNativeRedirectPromiseMeasurement:"handleNativeRedirectPromise",UpdateCloudDiscoveryMetadataMeasurement:"updateCloudDiscoveryMetadataMeasurement",UsernamePasswordClientAcquireToken:"usernamePasswordClientAcquireToken",NativeMessageHandlerHandshake:"nativeMessageHandlerHandshake",NativeGenerateAuthResult:"nativeGenerateAuthResult",RemoveHiddenIframe:"removeHiddenIframe",ClearTokensAndKeysWithClaims:"clearTokensAndKeysWithClaims",CacheManagerGetRefreshToken:"cacheManagerGetRefreshToken",ImportExistingCache:"importExistingCache",SetUserData:"setUserData",LocalStorageUpdated:"localStorageUpdated",GeneratePkceCodes:"generatePkceCodes",GenerateCodeVerifier:"generateCodeVerifier",GenerateCodeChallengeFromVerifier:"generateCodeChallengeFromVerifier",Sha256Digest:"sha256Digest",GetRandomValues:"getRandomValues",GenerateHKDF:"generateHKDF",GenerateBaseKey:"generateBaseKey",Base64Decode:"base64Decode",UrlEncodeArr:"urlEncodeArr",Encrypt:"encrypt",Decrypt:"decrypt",GenerateEarKey:"generateEarKey",DecryptEarResponse:"decryptEarResponse"
  },E_O=new Map([[R1.AcquireTokenByCode,"ATByCode"],[R1.AcquireTokenByRefreshToken,"ATByRT"],[R1.AcquireTokenSilent,"ATS"],[R1.AcquireTokenSilentAsync,"ATSAsync"],[R1.AcquireTokenPopup,"ATPopup"],[R1.AcquireTokenRedirect,"ATRedirect"],[R1.CryptoOptsGetPublicKeyThumbprint,"CryptoGetPKThumb"],[R1.CryptoOptsSignJwt,"CryptoSignJwt"],[R1.SilentCacheClientAcquireToken,"SltCacheClientAT"],[R1.SilentIframeClientAcquireToken,"SltIframeClientAT"],[R1.SilentRefreshClientAcquireToken,"SltRClientAT"],[R1.SsoSilent,"SsoSlt"],[R1.StandardInteractionClientGetDiscoveredAuthority,"StdIntClientGetDiscAuth"],[R1.FetchAccountIdWithNativeBroker,"FetchAccIdWithNtvBroker"],[R1.NativeInteractionClientAcquireToken,"NtvIntClientAT"],[R1.BaseClientCreateTokenRequestHeaders,"BaseClientCreateTReqHead"],[R1.NetworkClientSendPostRequestAsync,"NetClientSendPost"],[R1.RefreshTokenClientExecutePostToTokenEndpoint,"RTClientExecPost"],[R1.AuthorizationCodeClientExecutePostToTokenEndpoint,"AuthCodeClientExecPost"],[R1.BrokerHandhshake,"BrokerHandshake"],[R1.AcquireTokenByRefreshTokenInBroker,"ATByRTInBroker"],[R1.AcquireTokenByBroker,"ATByBroker"],[R1.RefreshTokenClientExecuteTokenRequest,"RTClientExecTReq"],[R1.RefreshTokenClientAcquireToken,"RTClientAT"],[R1.RefreshTokenClientAcquireTokenWithCachedRefreshToken,"RTClientATWithCachedRT"],[R1.RefreshTokenClientAcquireTokenByRefreshToken,"RTClientATByRT"],[R1.RefreshTokenClientCreateTokenRequestBody,"RTClientCreateTReqBody"],[R1.AcquireTokenFromCache,"ATFromCache"],[R1.SilentFlowClientAcquireCachedToken,"SltFlowClientATCached"],[R1.SilentFlowClientGenerateResultFromCacheRecord,"SltFlowClientGenResFromCache"],[R1.AcquireTokenBySilentIframe,"ATBySltIframe"],[R1.InitializeBaseRequest,"InitBaseReq"],[R1.InitializeSilentRequest,"InitSltReq"],[R1.InitializeClientApplication,"InitClientApplication"],[R1.InitializeCache,"InitCache"],[R1.ImportExistingCache,"importCache"],[R1.SetUserData,"setUserData"],[R1.LocalStorageUpdated,"localStorageUpdated"],[R1.SilentIframeClientTokenHelper,"SIClientTHelper"],[R1.SilentHandlerInitiateAuthRequest,"SHandlerInitAuthReq"],[R1.SilentHandlerMonitorIframeForHash,"SltHandlerMonitorIframeForHash"],[R1.SilentHandlerLoadFrame,"SHandlerLoadFrame"],[R1.SilentHandlerLoadFrameSync,"SHandlerLoadFrameSync"],[R1.StandardInteractionClientCreateAuthCodeClient,"StdIntClientCreateAuthCodeClient"],[R1.StandardInteractionClientGetClientConfiguration,"StdIntClientGetClientConf"],[R1.StandardInteractionClientInitializeAuthorizationRequest,"StdIntClientInitAuthReq"],[R1.GetAuthCodeUrl,"GetAuthCodeUrl"],[R1.HandleCodeResponseFromServer,"HandleCodeResFromServer"],[R1.HandleCodeResponse,"HandleCodeResp"],[R1.HandleResponseEar,"HandleRespEar"],[R1.HandleResponseCode,"HandleRespCode"],[R1.HandleResponsePlatformBroker,"HandleRespPlatBroker"],[R1.UpdateTokenEndpointAuthority,"UpdTEndpointAuth"],[R1.AuthClientAcquireToken,"AuthClientAT"],[R1.AuthClientExecuteTokenRequest,"AuthClientExecTReq"],[R1.AuthClientCreateTokenRequestBody,"AuthClientCreateTReqBody"],[R1.PopTokenGenerateCnf,"PopTGenCnf"],[R1.PopTokenGenerateKid,"PopTGenKid"],[R1.HandleServerTokenResponse,"HandleServerTRes"],[R1.DeserializeResponse,"DeserializeRes"],[R1.AuthorityFactoryCreateDiscoveredInstance,"AuthFactCreateDiscInst"],[R1.AuthorityResolveEndpointsAsync,"AuthResolveEndpointsAsync"],[R1.AuthorityResolveEndpointsFromLocalSources,"AuthResolveEndpointsFromLocal"],[R1.AuthorityGetCloudDiscoveryMetadataFromNetwork,"AuthGetCDMetaFromNet"],[R1.AuthorityUpdateCloudDiscoveryMetadata,"AuthUpdCDMeta"],[R1.AuthorityGetEndpointMetadataFromNetwork,"AuthUpdCDMetaFromNet"],[R1.AuthorityUpdateEndpointMetadata,"AuthUpdEndpointMeta"],[R1.AuthorityUpdateMetadataWithRegionalInformation,"AuthUpdMetaWithRegInfo"],[R1.RegionDiscoveryDetectRegion,"RegDiscDetectReg"],[R1.RegionDiscoveryGetRegionFromIMDS,"RegDiscGetRegFromIMDS"],[R1.RegionDiscoveryGetCurrentVersion,"RegDiscGetCurrentVer"],[R1.AcquireTokenByCodeAsync,"ATByCodeAsync"],[R1.GetEndpointMetadataFromNetwork,"GetEndpointMetaFromNet"],[R1.GetCloudDiscoveryMetadataFromNetworkMeasurement,"GetCDMetaFromNet"],[R1.HandleRedirectPromiseMeasurement,"HandleRedirectPromise"],[R1.HandleNativeRedirectPromiseMeasurement,"HandleNtvRedirectPromise"],[R1.UpdateCloudDiscoveryMetadataMeasurement,"UpdateCDMeta"],[R1.UsernamePasswordClientAcquireToken,"UserPassClientAT"],[R1.NativeMessageHandlerHandshake,"NtvMsgHandlerHandshake"],[R1.NativeGenerateAuthResult,"NtvGenAuthRes"],[R1.RemoveHiddenIframe,"RemoveHiddenIframe"],[R1.ClearTokensAndKeysWithClaims,"ClearTAndKeysWithClaims"],[R1.CacheManagerGetRefreshToken,"CacheManagerGetRT"],[R1.GeneratePkceCodes,"GenPkceCodes"],[R1.GenerateCodeVerifier,"GenCodeVerifier"],[R1.GenerateCodeChallengeFromVerifier,"GenCodeChallengeFromVerifier"],[R1.Sha256Digest,"Sha256Digest"],[R1.GetRandomValues,"GetRandomValues"],[R1.GenerateHKDF,"genHKDF"],[R1.GenerateBaseKey,"genBaseKey"],[R1.Base64Decode,"b64Decode"],[R1.UrlEncodeArr,"urlEncArr"],[R1.Encrypt,"encrypt"],[R1.Decrypt,"decrypt"],[R1.GenerateEarKey,"genEarKey"],[R1.DecryptEarResponse,"decryptEarResp"]]),uGq={
    NotStarted:0,InProgress:1,Completed:2
  }
} /* confidence: 95% */

/* original: q01 */ var composed_value=L(()=>{
  Wu();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: SW8 */ var composed_value=L(()=>{
  FZ1();
  ZW8();
  K2();
  vW8();
  TW8();
  tZ1();
  kd6();
  hX();
  q01();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */bm9={
    tokenRenewalOffsetSeconds:F06,preventCorsPreflight:!1
  },xm9={
    loggerCallback:()=>{
      
    },piiLoggingEnabled:!1,logLevel:vH.Info,correlationId:Q1.EMPTY_STRING
  },Im9={
    claimsBasedCachingEnabled:!1
  },um9={
    async sendGetRequestAsync(){
      throw J7(D_)
    },async sendPostRequestAsync(){
      throw J7(D_)
    }
  },mm9={
    sku:Q1.SKU,version:d06,cpu:Q1.EMPTY_STRING,os:Q1.EMPTY_STRING
  },pm9={
    clientSecret:Q1.EMPTY_STRING,clientAssertion:void 0
  },Bm9={
    azureCloudInstance:Zi.None,tenant:`${Q1.DEFAULT_COMMON_TENANT}`
  },gm9={
    application:{
      appName:"",appVersion:""
    }
  }
} /* confidence: 30% */

/* original: Ed6 */ var home_account_id_UPN=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */CT={
    HOME_ACCOUNT_ID:"home_account_id",UPN:"UPN"
  }
} /* confidence: 65% */

/* original: s06 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: e06 */ var composed_value=L(()=>{
  K2();
  s06();
  vd6();
  Y16();
  vi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: gGq */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: UGq */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: dGq */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: yi */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: lGq */ var composed_value=L(()=>{
  K2();
  Wu();
  yi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */Fd6.IMDS_OPTIONS={
    headers:{
      Metadata:"true"
    }
  } /* confidence: 30% */

/* original: w16 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: FW8 */ var composed_value=L(()=>{
  r06();
  hX();
  K2();
  w16();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: QW8 */ var tenant_tenantid=L(()=>{
  QZ1();
  gGq();
  O16();
  hX();
  K2();
  oZ1();
  Y16();
  kd6();
  TW8();
  UGq();
  dGq();
  lGq();
  pS();
  Wu();
  yi();
  FW8();
  wM();
  vi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */FP.reservedTenantDomains=new Set(["{tenant}","{tenantid}",$N.COMMON,$N.CONSUMERS,$N.ORGANIZATIONS])
} /* confidence: 65% */

/* original: KG1 */ var composed_value=L(()=>{
  QW8();
  hX();
  Wu();
  yi();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: _G6 */ var error_handler=L(()=>{
  pS();
  /*! @azure/msal-common v15.13.1 2025-10-29 */xT=class xT extends _9{
    constructor(q,K,_,z,Y){
      super(q,K,_);
      this.name="ServerError",this.errorNo=z,this.status=Y,Object.setPrototypeOf(this,xT.prototype)
    }
  }
} /* confidence: 95% */

/* original: cW8 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: iGq */ var composed_value=L(()=>{
  K2();
  _G6();
  cW8();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: rGq */ var error_handler=L(()=>{
  pS();
  /*! @azure/msal-common v15.13.1 2025-10-29 */lW8=class lW8 extends _9{
    constructor(q,K,_){
      super(q.errorCode,q.errorMessage,q.subError);
      Object.setPrototypeOf(this,lW8.prototype),this.name="NetworkError",this.error=q,this.httpStatus=K,this.responseHeaders=_
    }
  }
} /* confidence: 95% */

/* original: dd6 */ var composed_value=L(()=>{
  SW8();
  ZW8();
  K2();
  vW8();
  Ed6();
  i06();
  e06();
  M$6();
  KG1();
  Wu();
  iGq();
  pS();
  hX();
  rGq();
  yi();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: rW8 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: nd6 */ var action_creator=L(()=>{
  K2();
  pS();
  rW8();
  /*! @azure/msal-common v15.13.1 2025-10-29 */oGq=[_G1,zG1,YG1,H16,nW8],vp9=["message_only","additional_action","basic_action","user_password_expired","consent_required","bad_token"],oW8={
    [j16]:"No refresh token found in the cache. Please sign-in.",[cd6]:"The requested account is not available in the native broker. It may have been deleted or logged out. Please sign-in again using an interactive API.",[ld6]:"Refresh token has expired.",[H16]:"Identity provider returned bad_token due to an expired or invalid refresh token. Please invoke an interactive API to resolve.",[nW8]:"`canShowUI` flag in Edge was set to false. User interaction required on web page. Please invoke an interactive API to resolve."
  },$G1={
    noTokensFoundError:{
      code:j16,desc:oW8[j16]
    },native_account_unavailable:{
      code:cd6,desc:oW8[cd6]
    },bad_token:{
      code:H16,desc:oW8[H16]
    }
  };
  XL=class XL extends _9{
    constructor(q,K,_,z,Y,$,O,A){
      super(q,K,_);
      Object.setPrototypeOf(this,XL.prototype),this.timestamp=z||Q1.EMPTY_STRING,this.traceId=Y||Q1.EMPTY_STRING,this.correlationId=$||Q1.EMPTY_STRING,this.claims=O||Q1.EMPTY_STRING,this.name="InteractionRequiredAuthError",this.errorNo=A
    }
  }
} /* confidence: 95% */

/* original: aGq */ var composed_value=L(()=>{
  K2();
  hX();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: eW8 */ var composed_value=L(()=>{
  w16();
  O16();
  Wu();
  yi();
  /*! @azure/msal-common v15.13.1 2025-10-29 */Tp9={
    SW:"sw"
  }
} /* confidence: 30% */

/* original: OG1 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: id6 */ var composed_value=L(()=>{
  hX();
  _G6();
  vd6();
  yW8();
  nd6();
  aGq();
  K2();
  eW8();
  OG1();
  Wu();
  r06();
  dZ1();
  VW8();
  FW8();
  w16();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: qD8 */ var composed_value=L(()=>{
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: qvq */ var composed_value=L(()=>{
  dd6();
  w16();
  hX();
  id6();
  K2();
  $16();
  r06();
  Wu();
  yi();
  QW8();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */_D8=class _D8 extends bG{
    constructor(q,K){
      super(q,K)
    }async acquireCachedToken(q){
      this.performanceClient?.addQueueMeasurement(R1.SilentFlowClientAcquireCachedToken,q.correlationId);
      let K=fw.NOT_APPLICABLE;
      if(q.forceRefresh||!this.config.cacheOptions.claimsBasedCachingEnabled&&!Zw.isEmptyObj(q.claims))throw this.setCacheOutcome(fw.FORCE_REFRESH_OR_CLAIMS,q.correlationId),J7(fi);
      if(!q.account)throw J7(Wi);
      let _=q.account.tenantId||nGq(q.authority),z=this.cacheManager.getTokenKeys(),Y=this.cacheManager.getAccessToken(q.account,q,z,_);
      if(!Y)throw this.setCacheOutcome(fw.NO_CACHED_ACCESS_TOKEN,q.correlationId),J7(fi);
      else if(o01(Y.cachedAt)||qG6(Y.expiresOn,this.config.systemOptions.tokenRenewalOffsetSeconds))throw this.setCacheOutcome(fw.CACHED_ACCESS_TOKEN_EXPIRED,q.correlationId),J7(fi);
      else if(Y.refreshOn&&qG6(Y.refreshOn,0))K=fw.PROACTIVELY_REFRESHED;
      let $=q.authority||this.authority.getPreferredCache(),O={
        account:this.cacheManager.getAccount(this.cacheManager.generateAccountKey(q.account),q.correlationId),accessToken:Y,idToken:this.cacheManager.getIdToken(q.account,q.correlationId,z,_,this.performanceClient),refreshToken:null,appMetadata:this.cacheManager.readAppMetadataFromCache($)
      };
      if(this.setCacheOutcome(K,q.correlationId),this.config.serverTelemetryManager)this.config.serverTelemetryManager.incrementCacheHits();
      return[await ez(this.generateResultFromCacheRecord.bind(this),R1.SilentFlowClientGenerateResultFromCacheRecord,this.logger,this.performanceClient,q.correlationId)(O,q),K]
    }setCacheOutcome(q,K){
      if(this.serverTelemetryManager?.setCacheOutcome(q),this.performanceClient?.addFields({
        cacheOutcome:q
      },K),q!==fw.NOT_APPLICABLE)this.logger.info(`Token refresh is required due to cache outcome: ${q}`)
    }async generateResultFromCacheRecord(q,K){
      this.performanceClient?.addQueueMeasurement(R1.SilentFlowClientGenerateResultFromCacheRecord,K.correlationId);
      let _;
      if(q.idToken)_=Ti(q.idToken.secret,this.config.cryptoInterface.base64Decode);
      if(K.maxAge||K.maxAge===0){
        let z=_?.auth_time;
        if(!z)throw J7(Pi);
        Vd6(z,K.maxAge)
      }return kJ.generateAuthenticationResult(this.cryptoUtils,this.authority,q,!0,K,_)
    }
  }
} /* confidence: 30% */

/* original: _vq */ var composed_value=L(()=>{
  e06();
  s06();
  K2();
  i06();
  M$6();
  O16();
  hX();
  nd6();
  _G6();
  wM();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: $vq */ var composed_value=L(()=>{
  K2();
  pS();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: IO */ var composed_value=L(()=>{
  tGq();
  eGq();
  qvq();
  dd6();
  Ed6();
  QW8();
  TW8();
  kd6();
  tZ1();
  yW8();
  O16();
  FZ1();
  _vq();
  e06();
  id6();
  vd6();
  ZW8();
  nd6();
  rW8();
  pS();
  mZ1();
  _G6();
  hX();
  wM();
  Y16();
  vi();
  K2();
  $16();
  $vq();
  r06();
  KG1();
  FW8();
  w16();
  M$6();
  s06();
  OG1();
  qD8();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: zD8 */ var composed_value=L(()=>{
  IO();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: Ovq */ var composed_value=L(()=>{
  PW8();
  zD8();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: R2 */ var auth_handler=L(()=>{
  IO();
  /*! @azure/msal-node v3.8.1 2025-10-29 */wG1=`https://login.microsoftonline.com/${pp9}/`,wN={
    AUTHORIZATION_HEADER_NAME:"Authorization",METADATA_HEADER_NAME:"Metadata",APP_SERVICE_SECRET_HEADER_NAME:"X-IDENTITY-HEADER",ML_AND_SF_SECRET_HEADER_NAME:"secret"
  },SX={
    API_VERSION:"api-version",RESOURCE:"resource",SHA256_TOKEN_TO_REFRESH:"token_sha256_to_refresh",XMS_CC:"xms_cc"
  },D3={
    AZURE_POD_IDENTITY_AUTHORITY_HOST:"AZURE_POD_IDENTITY_AUTHORITY_HOST",DEFAULT_IDENTITY_CLIENT_ID:"DEFAULT_IDENTITY_CLIENT_ID",IDENTITY_ENDPOINT:"IDENTITY_ENDPOINT",IDENTITY_HEADER:"IDENTITY_HEADER",IDENTITY_SERVER_THUMBPRINT:"IDENTITY_SERVER_THUMBPRINT",IMDS_ENDPOINT:"IMDS_ENDPOINT",MSI_ENDPOINT:"MSI_ENDPOINT",MSI_SECRET:"MSI_SECRET"
  },m3={
    APP_SERVICE:"AppService",AZURE_ARC:"AzureArc",CLOUD_SHELL:"CloudShell",DEFAULT_TO_IMDS:"DefaultToImds",IMDS:"Imds",MACHINE_LEARNING:"MachineLearning",SERVICE_FABRIC:"ServiceFabric"
  },TH={
    SYSTEM_ASSIGNED:"system-assigned",USER_ASSIGNED_CLIENT_ID:"user-assigned-client-id",USER_ASSIGNED_RESOURCE_ID:"user-assigned-resource-id",USER_ASSIGNED_OBJECT_ID:"user-assigned-object-id"
  },h2={
    GET:"get",POST:"post"
  },YD8={
    SUCCESS_RANGE_START:K9.SUCCESS_RANGE_START,SUCCESS_RANGE_END:K9.SUCCESS_RANGE_END,SERVER_ERROR:K9.SERVER_ERROR
  },Jvq={
    SHA256:"sha256"
  },$D8={
    CV_CHARSET:"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
  },jG1={
    KEY_SEPARATOR:"-"
  },uT={
    MSAL_SKU:"msal.js.node",JWT_BEARER_ASSERTION_TYPE:"urn:ietf:params:oauth:client-assertion-type:jwt-bearer",AUTHORIZATION_PENDING:"authorization_pending",HTTP_PROTOCOL:"http://",LOCALHOST:"localhost"
  },Li={
    acquireTokenSilent:62,acquireTokenByUsernamePassword:371,acquireTokenByDeviceCode:671,acquireTokenByClientCredential:771,acquireTokenByCode:871,acquireTokenByRefreshToken:872
  },FS={
    RSA_256:"RS256",PSS_256:"PS256",X5T_256:"x5t#S256",X5T:"x5t",X5C:"x5c",AUDIENCE:"aud",EXPIRATION_TIME:"exp",ISSUER:"iss",SUBJECT:"sub",NOT_BEFORE:"nbf",JWT_ID:"jti"
  },OD8={
    INTERVAL_MS:100,TIMEOUT_MS:5000
  }
} /* confidence: 95% */

/* original: Zvq */ var composed_value=L(()=>{
  IO();
  R2();
  Xvq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: N$6 */ var azure_pod_identity_authority_h=L(()=>{
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */V$6={
    [D3.AZURE_POD_IDENTITY_AUTHORITY_HOST]:"azure_pod_identity_authority_host_url_malformed",[D3.IDENTITY_ENDPOINT]:"identity_endpoint_url_malformed",[D3.IMDS_ENDPOINT]:"imds_endpoint_url_malformed",[D3.MSI_ENDPOINT]:"msi_endpoint_url_malformed"
  }
} /* confidence: 65% */

/* original: $G6 */ var env_config=L(()=>{
  IO();
  N$6();
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */Bp9={
    [AD8]:"The file path in the WWW-Authenticate header does not contain a .key file.",[wD8]:"The file path in the WWW-Authenticate header is not in a valid Windows or Linux Format.",[X16]:"More than one ManagedIdentityIdType was provided.",[jD8]:"The secret in the file on the file path in the WWW-Authenticate header is greater than 4096 bytes.",[HD8]:"The platform is not supported by Azure Arc. Azure Arc only supports Windows and Linux.",[Gvq]:"A ManagedIdentityId id was not provided.",[V$6.AZURE_POD_IDENTITY_AUTHORITY_HOST]:`The Managed Identity's '${D3.AZURE_POD_IDENTITY_AUTHORITY_HOST}' environment variable is malformed.`,[V$6.IDENTITY_ENDPOINT]:`The Managed Identity's '${D3.IDENTITY_ENDPOINT}' environment variable is malformed.`,[V$6.IMDS_ENDPOINT]:`The Managed Identity's '${D3.IMDS_ENDPOINT}' environment variable is malformed.`,[V$6.MSI_ENDPOINT]:`The Managed Identity's '${D3.MSI_ENDPOINT}' environment variable is malformed.`,[vvq]:"Authentication unavailable. The request to the managed identity endpoint timed out.",[JD8]:"Azure Arc Managed Identities can only be system assigned.",[MD8]:"Cloud Shell Managed Identities can only be system assigned.",[XD8]:"Unable to create a Managed Identity source based on environment variables.",[sd6]:"Unable to read the secret file.",[Tvq]:"Service Fabric user assigned managed identity ClientId or ResourceId is not configurable at runtime.",[PD8]:"A 401 response was received form the Azure Arc Managed Identity, but the www-authenticate header is missing.",[WD8]:"A 401 response was received form the Azure Arc Managed Identity, but the www-authenticate header is in an unsupported format."
  };
  JG1=class JG1 extends _9{
    constructor(q){
      super(q,Bp9[q]);
      this.name="ManagedIdentityError",Object.setPrototypeOf(this,JG1.prototype)
    }
  }
} /* confidence: 95% */

/* original: kvq */ var composed_value=L(()=>{
  $G6();
  R2();
  N$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: PG1 */ var composed_value=L(()=>{
  IO();
  Zvq();
  kvq();
  td6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */gp9={
    clientId:Q1.EMPTY_STRING,authority:Q1.DEFAULT_AUTHORITY,clientSecret:Q1.EMPTY_STRING,clientAssertion:Q1.EMPTY_STRING,clientCertificate:{
      thumbprint:Q1.EMPTY_STRING,thumbprintSha256:Q1.EMPTY_STRING,privateKey:Q1.EMPTY_STRING,x5c:Q1.EMPTY_STRING
    },knownAuthorities:[],cloudDiscoveryMetadata:Q1.EMPTY_STRING,authorityMetadata:Q1.EMPTY_STRING,clientCapabilities:[],protocolMode:CG.AAD,azureCloudOptions:{
      azureCloudInstance:Zi.None,tenant:Q1.EMPTY_STRING
    },skipAuthorityMetadataCache:!1,encodeExtraQueryParams:!1
  },Fp9={
    claimsBasedCachingEnabled:!1
  },XG1={
    loggerCallback:()=>{
      
    },piiLoggingEnabled:!1,logLevel:vH.Info
  },Up9={
    loggerOptions:XG1,networkClient:new ad6,proxyUrl:Q1.EMPTY_STRING,customAgentOptions:{
      
    },disableInternalRetries:!1
  },Qp9={
    application:{
      appName:Q1.EMPTY_STRING,appVersion:Q1.EMPTY_STRING
    }
  }
} /* confidence: 30% */

/* original: _c6 */ var composed_value=L(()=>{
  IO();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: ZD8 */ var composed_value=L(()=>{
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: ZTq */ var composed_value=L(()=>{
  IO();
  R2();
  _c6();
  ZD8();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: zc6 */ var composed_value=L(()=>{
  IO();
  TG1();
  _c6();
  ZTq();
  ZD8();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: GD8 */ var composed_value=L(()=>{
  K2();
  q01();
  /*! @azure/msal-common v15.13.1 2025-10-29 */
} /* confidence: 30% */

/* original: TTq */ var composed_value=L(()=>{
  IO();
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: VG1 */ var composed_value=L(()=>{
  vD8();
  IO();
  zD8();
  PW8();
  zc6();
  TG1();
  /*! @azure/msal-node v3.8.1 2025-10-29 */Yc6={
    Account:{
      
    },IdToken:{
      
    },AccessToken:{
      
    },RefreshToken:{
      
    },AppMetadata:{
      
    }
  }
} /* confidence: 30% */

/* original: oD8 */ var composed_value=L(()=>{
  IO();
  _c6();
  R2();
  tNq=w6(sNq(),1);
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: qyq */ var composed_value=L(()=>{
  IO();
  R2();
  XG6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: sD8 */ var composed_value=L(()=>{
  IO();
  PG1();
  zc6();
  vD8();
  R2();
  VG1();
  oD8();
  XG6();
  td6();
  wv1();
  qyq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: Kyq */ var composed_value=L(()=>{
  IO();
  td6();
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: _yq */ var NativeBrokerimplementationwasp=L(()=>{
  R2();
  IO();
  sD8();
  td6();
  Kyq();
  Hv1();
  XG6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */vc6=class vc6 extends R$6{
    constructor(q){
      super(q);
      if(this.config.broker.nativeBrokerPlugin)if(this.config.broker.nativeBrokerPlugin.isBrokerAvailable)this.nativeBrokerPlugin=this.config.broker.nativeBrokerPlugin,this.nativeBrokerPlugin.setLogger(this.config.system.loggerOptions);
      else this.logger.warning("NativeBroker implementation was provided but the broker is unavailable.");
      this.skus=J16.makeExtraSkuString({
        libraryName:uT.MSAL_SKU,libraryVersion:vu
      })
    }async acquireTokenByDeviceCode(q){
      this.logger.info("acquireTokenByDeviceCode called",q.correlationId);
      let K=Object.assign(q,await this.initializeBaseRequest(q)),_=this.initializeServerTelemetryManager(Li.acquireTokenByDeviceCode,K.correlationId);
      try{
        let z=await this.createAuthority(K.authority,K.correlationId,void 0,q.azureCloudOptions),Y=await this.buildOauthClientConfiguration(z,K.correlationId,"",_),$=new Gc6(Y);
        return this.logger.verbose("Device code client created",K.correlationId),await $.acquireToken(K)
      }catch(z){
        if(z instanceof _9)z.setCorrelationId(K.correlationId);
        throw _.cacheFailedRequest(z),z
      }
    }async acquireTokenInteractive(q){
      let K=q.correlationId||this.cryptoProvider.createNewGuid();
      this.logger.trace("acquireTokenInteractive called",K);
      let{
        openBrowser:_,successTemplate:z,errorTemplate:Y,windowHandle:$,loopbackClient:O,...A
      }=q;
      if(this.nativeBrokerPlugin){
        let X={
          ...A,clientId:this.config.auth.clientId,scopes:q.scopes||SG,redirectUri:q.redirectUri||"",authority:q.authority||this.config.auth.authority,correlationId:K,extraParameters:{
            ...A.extraQueryParameters,...A.tokenQueryParameters,[P$6.X_CLIENT_EXTRA_SKU]:this.skus
          },accountId:A.account?.nativeAccountId
        };
        return this.nativeBrokerPlugin.acquireTokenInteractive(X,$)
      }if(q.redirectUri){
        if(!this.config.broker.nativeBrokerPlugin)throw yj.createRedirectUriNotSupportedError();
        q.redirectUri=""
      }let{
        verifier:w,challenge:j
      }=await this.cryptoProvider.generatePkceCodes(),H=O||new jv1,J={
        
      },M=null;
      try{
        let X=H.listenForAuthCode(z,Y).then((Z)=>{
          J=Z
        }).catch((Z)=>{
          M=Z
        }),P=await this.waitForRedirectUri(H),W={
          ...A,correlationId:K,scopes:q.scopes||SG,redirectUri:P,responseMode:DF.QUERY,codeChallenge:j,codeChallengeMethod:WW8.S256
        },D=await this.getAuthCodeUrl(W);
        if(await _(D),await X,M)throw M;
        if(J.error)throw new xT(J.error,J.error_description,J.suberror);
        else if(!J.code)throw yj.createNoAuthCodeInResponseError();
        let f=J.client_info,G={
          code:J.code,codeVerifier:w,clientInfo:f||Q1.EMPTY_STRING,...W
        };
        return await this.acquireTokenByCode(G)
      }finally{
        H.closeServer()
      }
    }async acquireTokenSilent(q){
      let K=q.correlationId||this.cryptoProvider.createNewGuid();
      if(this.logger.trace("acquireTokenSilent called",K),this.nativeBrokerPlugin){
        let _={
          ...q,clientId:this.config.auth.clientId,scopes:q.scopes||SG,redirectUri:q.redirectUri||"",authority:q.authority||this.config.auth.authority,correlationId:K,extraParameters:{
            ...q.tokenQueryParameters,[P$6.X_CLIENT_EXTRA_SKU]:this.skus
          },accountId:q.account.nativeAccountId,forceRefresh:q.forceRefresh||!1
        };
        return this.nativeBrokerPlugin.acquireTokenSilent(_)
      }if(q.redirectUri){
        if(!this.config.broker.nativeBrokerPlugin)throw yj.createRedirectUriNotSupportedError();
        q.redirectUri=""
      }return super.acquireTokenSilent(q)
    }async signOut(q){
      if(this.nativeBrokerPlugin&&q.account.nativeAccountId){
        let K={
          clientId:this.config.auth.clientId,accountId:q.account.nativeAccountId,correlationId:q.correlationId||this.cryptoProvider.createNewGuid()
        };
        await this.nativeBrokerPlugin.signOut(K)
      }await this.getTokenCache().removeAccount(q.account,q.correlationId)
    }async getAllAccounts(){
      if(this.nativeBrokerPlugin){
        let q=this.cryptoProvider.createNewGuid();
        return this.nativeBrokerPlugin.getAllAccounts(this.config.auth.clientId,q)
      }return this.getTokenCache().getAllAccounts()
    }async waitForRedirectUri(q){
      return new Promise((K,_)=>{
        let z=0,Y=setInterval(()=>{
          if(OD8.TIMEOUT_MS/OD8.INTERVAL_MS<z){
            clearInterval(Y),_(yj.createLoopbackServerTimeoutError());
            return
          }try{
            let $=q.getRedirectUri();
            clearInterval(Y),K($);
            return
          }catch($){
            if($ instanceof _9&&$.errorCode===CX.noLoopbackServerExists.code){
              z++;
              return
            }clearInterval(Y),_($);
            return
          }
        },OD8.INTERVAL_MS)
      })
    }
  }
} /* confidence: 65% */

/* original: Oyq */ var composed_value=L(()=>{
  IO();
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: b$6 */ var clientid_client_id_object_id_m=L(()=>{
  IO();
  R2();
  $G6();
  $yq();
  Oyq();
  N$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */C$6={
    MANAGED_IDENTITY_CLIENT_ID_2017:"clientid",MANAGED_IDENTITY_CLIENT_ID:"client_id",MANAGED_IDENTITY_OBJECT_ID:"object_id",MANAGED_IDENTITY_RESOURCE_ID_IMDS:"msi_res_id",MANAGED_IDENTITY_RESOURCE_ID_NON_IMDS:"mi_res_id"
  };
  JN.getValidatedEnvVariableUrlString=(q,K,_,z)=>{
    try{
      return new b9(K).urlString
    }catch(Y){
      throw z.info(`[Managed Identity] ${_} managed identity is unavailable because the '${q}' environment variable is malformed.`),jM(V$6[q])
    }
  }
} /* confidence: 65% */

/* original: wyq */ var composed_value=L(()=>{
  GD8();
  Ayq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */$l9=[K9.NOT_FOUND,K9.REQUEST_TIMEOUT,K9.TOO_MANY_REQUESTS,K9.SERVER_ERROR,K9.SERVICE_UNAVAILABLE,K9.GATEWAY_TIMEOUT]
} /* confidence: 30% */

/* original: x$6 */ var composed_value=L(()=>{
  IO();
  wyq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: jyq */ var headers=L(()=>{
  b$6();
  R2();
  x$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */I$6=class I$6 extends JN{
    constructor(q,K,_,z,Y,$,O){
      super(q,K,_,z,Y);
      this.identityEndpoint=$,this.identityHeader=O
    }static getEnvironmentVariables(){
      let q=process.env[D3.IDENTITY_ENDPOINT],K=process.env[D3.IDENTITY_HEADER];
      return[q,K]
    }static tryCreate(q,K,_,z,Y){
      let[$,O]=I$6.getEnvironmentVariables();
      if(!$||!O)return q.info(`[Managed Identity] ${m3.APP_SERVICE} managed identity is unavailable because one or both of the '${D3.IDENTITY_HEADER}' and '${D3.IDENTITY_ENDPOINT}' environment variables are not defined.`),null;
      let A=I$6.getValidatedEnvVariableUrlString(D3.IDENTITY_ENDPOINT,$,m3.APP_SERVICE,q);
      return q.info(`[Managed Identity] Environment variables validation passed for ${m3.APP_SERVICE} managed identity. Endpoint URI: ${A}. Creating ${m3.APP_SERVICE} managed identity.`),new I$6(q,K,_,z,Y,$,O)
    }createRequest(q,K){
      let _=new PL(h2.GET,this.identityEndpoint);
      if(_.headers[wN.APP_SERVICE_SECRET_HEADER_NAME]=this.identityHeader,_.queryParameters[SX.API_VERSION]=Ol9,_.queryParameters[SX.RESOURCE]=q,K.idType!==TH.SYSTEM_ASSIGNED)_.queryParameters[this.getManagedIdentityUserAssignedIdQueryParameterKey(K.idType)]=K.id;
      return _
    }
  }
} /* confidence: 70% */

/* original: Pyq */ var auth_handler=L(()=>{
  IO();
  x$6();
  b$6();
  $G6();
  R2();
  N$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */Xyq={
    win32:`${process.env.ProgramData}\\AzureConnectedMachineAgent\\Tokens\\`,linux:"/var/opt/azcmagent/tokens/"
  },Ml9={
    win32:`${process.env.ProgramFiles}\\AzureConnectedMachineAgent\\himds.exe`,linux:"/opt/azcmagent/bin/himds"
  };
  W16=class W16 extends JN{
    constructor(q,K,_,z,Y,$){
      super(q,K,_,z,Y);
      this.identityEndpoint=$
    }static getEnvironmentVariables(){
      let q=process.env[D3.IDENTITY_ENDPOINT],K=process.env[D3.IMDS_ENDPOINT];
      if(!q||!K){
        let _=Ml9[process.platform];
        try{
          Al9(_,Hyq.F_OK|Hyq.R_OK),q=Jyq,K=Myq
        }catch(z){
          
        }
      }return[q,K]
    }static tryCreate(q,K,_,z,Y,$){
      let[O,A]=W16.getEnvironmentVariables();
      if(!O||!A)return q.info(`[Managed Identity] ${m3.AZURE_ARC} managed identity is unavailable through environment variables because one or both of '${D3.IDENTITY_ENDPOINT}' and '${D3.IMDS_ENDPOINT}' are not defined. ${m3.AZURE_ARC} managed identity is also unavailable through file detection.`),null;
      if(A===Myq)q.info(`[Managed Identity] ${m3.AZURE_ARC} managed identity is available through file detection. Defaulting to known ${m3.AZURE_ARC} endpoint: ${Jyq}. Creating ${m3.AZURE_ARC} managed identity.`);
      else{
        let w=W16.getValidatedEnvVariableUrlString(D3.IDENTITY_ENDPOINT,O,m3.AZURE_ARC,q);
        w.endsWith("/")&&w.slice(0,-1),W16.getValidatedEnvVariableUrlString(D3.IMDS_ENDPOINT,A,m3.AZURE_ARC,q),q.info(`[Managed Identity] Environment variables validation passed for ${m3.AZURE_ARC} managed identity. Endpoint URI: ${w}. Creating ${m3.AZURE_ARC} managed identity.`)
      }if($.idType!==TH.SYSTEM_ASSIGNED)throw jM(JD8);
      return new W16(q,K,_,z,Y,O)
    }createRequest(q){
      let K=new PL(h2.GET,this.identityEndpoint.replace("localhost","127.0.0.1"));
      return K.headers[wN.METADATA_HEADER_NAME]="true",K.queryParameters[SX.API_VERSION]=Jl9,K.queryParameters[SX.RESOURCE]=q,K
    }async getServerTokenResponseAsync(q,K,_,z){
      let Y;
      if(q.status===K9.UNAUTHORIZED){
        let $=q.headers["www-authenticate"];
        if(!$)throw jM(PD8);
        if(!$.includes("Basic realm="))throw jM(WD8);
        let O=$.split("Basic realm=")[1];
        if(!Xyq.hasOwnProperty(process.platform))throw jM(HD8);
        let A=Xyq[process.platform],w=Hl9.basename(O);
        if(!w.endsWith(".key"))throw jM(AD8);
        if(A+w!==O)throw jM(wD8);
        let j;
        try{
          j=await wl9(O).size
        }catch(M){
          throw jM(sd6)
        }if(j>Mvq)throw jM(jD8);
        let H;
        try{
          H=jl9(O,jZ.UTF8)
        }catch(M){
          throw jM(sd6)
        }let J=`Basic ${H}`;
        this.logger.info("[Managed Identity] Adding authorization header to the request."),_.headers[wN.AUTHORIZATION_HEADER_NAME]=J;
        try{
          Y=await K.sendGetRequestAsync(_.computeUri(),z)
        }catch(M){
          if(M instanceof _9)throw M;
          else throw J7(UA.networkError)
        }
      }return this.getServerTokenResponse(Y||q)
    }
  }
} /* confidence: 95% */

/* original: Wyq */ var headers=L(()=>{
  x$6();
  b$6();
  R2();
  $G6();
  N$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */u$6=class u$6 extends JN{
    constructor(q,K,_,z,Y,$){
      super(q,K,_,z,Y);
      this.msiEndpoint=$
    }static getEnvironmentVariables(){
      return[process.env[D3.MSI_ENDPOINT]]
    }static tryCreate(q,K,_,z,Y,$){
      let[O]=u$6.getEnvironmentVariables();
      if(!O)return q.info(`[Managed Identity] ${m3.CLOUD_SHELL} managed identity is unavailable because the '${D3.MSI_ENDPOINT} environment variable is not defined.`),null;
      let A=u$6.getValidatedEnvVariableUrlString(D3.MSI_ENDPOINT,O,m3.CLOUD_SHELL,q);
      if(q.info(`[Managed Identity] Environment variable validation passed for ${m3.CLOUD_SHELL} managed identity. Endpoint URI: ${A}. Creating ${m3.CLOUD_SHELL} managed identity.`),$.idType!==TH.SYSTEM_ASSIGNED)throw jM(MD8);
      return new u$6(q,K,_,z,Y,O)
    }createRequest(q){
      let K=new PL(h2.POST,this.msiEndpoint);
      return K.headers[wN.METADATA_HEADER_NAME]="true",K.bodyParameters[SX.RESOURCE]=q,K
    }
  }
} /* confidence: 70% */

/* original: fyq */ var composed_value=L(()=>{
  GD8();
  Dyq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */Xl9=[K9.NOT_FOUND,K9.REQUEST_TIMEOUT,K9.GONE,K9.TOO_MANY_REQUESTS]
} /* confidence: 30% */

/* original: Gyq */ var headers=L(()=>{
  x$6();
  b$6();
  R2();
  fyq();
  /*! @azure/msal-node v3.8.1 2025-10-29 */vl9=`http://169.254.169.254${Zyq}`;
  Vc6=class Vc6 extends JN{
    constructor(q,K,_,z,Y,$){
      super(q,K,_,z,Y);
      this.identityEndpoint=$
    }static tryCreate(q,K,_,z,Y){
      let $;
      if(process.env[D3.AZURE_POD_IDENTITY_AUTHORITY_HOST])q.info(`[Managed Identity] Environment variable ${D3.AZURE_POD_IDENTITY_AUTHORITY_HOST} for ${m3.IMDS} returned endpoint: ${process.env[D3.AZURE_POD_IDENTITY_AUTHORITY_HOST]}`),$=Vc6.getValidatedEnvVariableUrlString(D3.AZURE_POD_IDENTITY_AUTHORITY_HOST,`${process.env[D3.AZURE_POD_IDENTITY_AUTHORITY_HOST]}${Zyq}`,m3.IMDS,q);
      else q.info(`[Managed Identity] Unable to find ${D3.AZURE_POD_IDENTITY_AUTHORITY_HOST} environment variable for ${m3.IMDS}, using the default endpoint.`),$=vl9;
      return new Vc6(q,K,_,z,Y,$)
    }createRequest(q,K){
      let _=new PL(h2.GET,this.identityEndpoint);
      if(_.headers[wN.METADATA_HEADER_NAME]="true",_.queryParameters[SX.API_VERSION]=Tl9,_.queryParameters[SX.RESOURCE]=q,K.idType!==TH.SYSTEM_ASSIGNED)_.queryParameters[this.getManagedIdentityUserAssignedIdQueryParameterKey(K.idType,!0)]=K.id;
      return _.retryPolicy=new m$6,_
    }
  }
} /* confidence: 70% */

/* original: vyq */ var headers=L(()=>{
  x$6();
  b$6();
  R2();
  /*! @azure/msal-node v3.8.1 2025-10-29 */p$6=class p$6 extends JN{
    constructor(q,K,_,z,Y,$,O){
      super(q,K,_,z,Y);
      this.identityEndpoint=$,this.identityHeader=O
    }static getEnvironmentVariables(){
      let q=process.env[D3.IDENTITY_ENDPOINT],K=process.env[D3.IDENTITY_HEADER],_=process.env[D3.IDENTITY_SERVER_THUMBPRINT];
      return[q,K,_]
    }static tryCreate(q,K,_,z,Y,$){
      let[O,A,w]=p$6.getEnvironmentVariables();
      if(!O||!A||!w)return q.info(`[Managed Identity] ${m3.SERVICE_FABRIC} managed identity is unavailable because one or all of the '${D3.IDENTITY_HEADER}', '${D3.IDENTITY_ENDPOINT}' or '${D3.IDENTITY_SERVER_THUMBPRINT}' environment variables are not defined.`),null;
      let j=p$6.getValidatedEnvVariableUrlString(D3.IDENTITY_ENDPOINT,O,m3.SERVICE_FABRIC,q);
      if(q.info(`[Managed Identity] Environment variables validation passed for ${m3.SERVICE_FABRIC} managed identity. Endpoint URI: ${j}. Creating ${m3.SERVICE_FABRIC} managed identity.`),$.idType!==TH.SYSTEM_ASSIGNED)q.warning(`[Managed Identity] ${m3.SERVICE_FABRIC} user assigned managed identity is configured in the cluster, not during runtime. See also: https://learn.microsoft.com/en-us/azure/service-fabric/configure-existing-cluster-enable-managed-identity-token-service.`);
      return new p$6(q,K,_,z,Y,O,A)
    }createRequest(q,K){
      let _=new PL(h2.GET,this.identityEndpoint);
      if(_.headers[wN.ML_AND_SF_SECRET_HEADER_NAME]=this.identityHeader,_.queryParameters[SX.API_VERSION]=kl9,_.queryParameters[SX.RESOURCE]=q,K.idType!==TH.SYSTEM_ASSIGNED)_.queryParameters[this.getManagedIdentityUserAssignedIdQueryParameterKey(K.idType)]=K.id;
      return _
    }
  }
} /* confidence: 70% */

/* original: Tyq */ var headers=L(()=>{
  b$6();
  R2();
  x$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */Nl9=`Only client id is supported for user-assigned managed identity in ${m3.MACHINE_LEARNING}.`;
  B$6=class B$6 extends JN{
    constructor(q,K,_,z,Y,$,O){
      super(q,K,_,z,Y);
      this.msiEndpoint=$,this.secret=O
    }static getEnvironmentVariables(){
      let q=process.env[D3.MSI_ENDPOINT],K=process.env[D3.MSI_SECRET];
      return[q,K]
    }static tryCreate(q,K,_,z,Y){
      let[$,O]=B$6.getEnvironmentVariables();
      if(!$||!O)return q.info(`[Managed Identity] ${m3.MACHINE_LEARNING} managed identity is unavailable because one or both of the '${D3.MSI_ENDPOINT}' and '${D3.MSI_SECRET}' environment variables are not defined.`),null;
      let A=B$6.getValidatedEnvVariableUrlString(D3.MSI_ENDPOINT,$,m3.MACHINE_LEARNING,q);
      return q.info(`[Managed Identity] Environment variables validation passed for ${m3.MACHINE_LEARNING} managed identity. Endpoint URI: ${A}. Creating ${m3.MACHINE_LEARNING} managed identity.`),new B$6(q,K,_,z,Y,$,O)
    }createRequest(q,K){
      let _=new PL(h2.GET,this.msiEndpoint);
      if(_.headers[wN.METADATA_HEADER_NAME]="true",_.headers[wN.ML_AND_SF_SECRET_HEADER_NAME]=this.secret,_.queryParameters[SX.API_VERSION]=Vl9,_.queryParameters[SX.RESOURCE]=q,K.idType===TH.SYSTEM_ASSIGNED)_.queryParameters[C$6.MANAGED_IDENTITY_CLIENT_ID_2017]=process.env[D3.DEFAULT_IDENTITY_CLIENT_ID];
      else if(K.idType===TH.USER_ASSIGNED_CLIENT_ID)_.queryParameters[this.getManagedIdentityUserAssignedIdQueryParameterKey(K.idType,!1,!0)]=K.id;
      else throw Error(Nl9);
      return _
    }
  }
} /* confidence: 70% */

/* original: kyq */ var composed_value=L(()=>{
  jyq();
  Pyq();
  Wyq();
  Gyq();
  vyq();
  $G6();
  R2();
  Tyq();
  N$6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: Vyq */ var composed_value=L(()=>{
  IO();
  PG1();
  XG6();
  zc6();
  tD8();
  kyq();
  vD8();
  R2();
  ZD8();
  /*! @azure/msal-node v3.8.1 2025-10-29 */yl9=[m3.SERVICE_FABRIC]
} /* confidence: 30% */

/* original: Nyq */ var composed_value=L(()=>{
  IO();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: qf8 */ var composed_value=L(()=>{
  Ovq();
  _yq();
  zyq();
  sD8();
  tD8();
  Hv1();
  Jv1();
  Vyq();
  wv1();
  oD8();
  VG1();
  Nyq();
  R2();
  zc6();
  IO();
  XG6();
  /*! @azure/msal-node v3.8.1 2025-10-29 */
} /* confidence: 30% */

/* original: yyq */ var composed_value=L(()=>{
  qf8()
} /* confidence: 30% */

/* original: v15 */ function utility_fn(){
  return null
} /* confidence: 40% */

