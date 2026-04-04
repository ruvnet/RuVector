// Module: message

/* original: z */ let composed_value=async()=>{
  let $=t7(),O=$?.accessToken&&OD()?{
    accessToken:$.accessToken
  }:q?{
    apiKey:q
  }:null;
  if(!O)throw Error("No auth available");
  return ry9(O)
}; /* confidence: 30% */

/* original: YD */ function helper_fn(){
  let K=w8().oauthAccount?.organizationUuid;
  if(K)return K;
  let _=t7()?.accessToken;
  if(_===void 0||!OD())return null;
  let Y=(await sg(_))?.organization?.uuid;
  if(!Y)return null;
  return Y
} /* confidence: 35% */

/* original: OD */ function utility_fn(){
  return t7()?.scopes?.includes(g_6)??!1
} /* confidence: 40% */

/* original: V5K */ function helper_fn(q){
  let{
    accessToken:K,orgUUID:_
  }=await dH(),z={
    ...eY(K),"x-organization-uuid":_
  },Y=`${m7().BASE_API_URL}/api/oauth/organizations/${_}/admin_requests`;
  return(await O1.post(Y,q,{
    headers:z
  })).data
} /* confidence: 35% */

/* original: y5K */ function helper_fn(q){
  let{
    accessToken:K,orgUUID:_
  }=await dH(),z={
    ...eY(K),"x-organization-uuid":_
  },Y=`${m7().BASE_API_URL}/api/oauth/organizations/${_}/admin_requests/eligibility?request_type=${q}`;
  return(await O1.get(Y,{
    headers:z
  })).data
} /* confidence: 35% */

/* original: uI8 */ function team_enterprise_message_Youror(){
  if(!w8().hasVisitedExtraUsage)S8((Y)=>({
    ...Y,hasVisitedExtraUsage:!0
  }));
  h5K();
  let q=jK(),K=q==="team"||q==="enterprise";
  if(!ag()&&K){
    let Y;
    try{
      Y=(await xL6())?.extra_usage
    }catch($){
      j6($)
    }if(Y?.is_enabled&&Y.monthly_limit===null)return{
      type:"message",value:"Your organization already has unlimited extra usage. No request needed."
    };
    try{
      if((await y5K("limit_increase"))?.is_allowed===!1)return{
        type:"message",value:"Please contact your admin to manage extra usage settings."
      }
    }catch($){
      j6($)
    }try{
      let $=await N5K("limit_increase",["pending","dismissed"]);
      if($&&$.length>0)return{
        type:"message",value:"You have already submitted a request for extra usage to your admin."
      }
    }catch($){
      j6($)
    }try{
      return await V5K({
        request_type:"limit_increase",details:null
      }),{
        type:"message",value:Y?.is_enabled?"Request sent to your admin to increase extra usage.":"Request sent to your admin to enable extra usage."
      }
    }catch($){
      j6($)
    }return{
      type:"message",value:"Please contact your admin to manage extra usage settings."
    }
  }let z=K?"https://claude.ai/admin-settings/usage":"https://claude.ai/settings/usage";
  try{
    let Y=await p3(z);
    return{
      type:"browser-opened",url:z,opened:Y
    }
  }catch(Y){
    return j6(Y),{
      type:"message",value:`Failed to open browser. Please visit ${z} to manage extra usage.`
    }
  }
} /* confidence: 65% */

/* original: aK7 */ function RemoteControlrequiresaclaudeai(){
  if(!sK7())return"Remote Control requires a claude.ai subscription. Run `claude auth login` to sign in with your claude.ai account.";
  if(!rJY())return"Remote Control requires a full-scope login token. Long-lived tokens (from `claude setup-token` or CLAUDE_CODE_OAUTH_TOKEN) are limited to inference-only for security reasons. Run `claude auth login` to use Remote Control.";
  if(!oJY()?.organizationUuid)return"Unable to determine your organization for Remote Control eligibility. Run `claude auth login` to refresh your account information.";
  if(!await ZN("tengu_ccr_bridge"))return"Remote Control is not yet enabled for your account.";
  return null
} /* confidence: 65% */

/* original: rJY */ function helper_fn(){
  try{
    return OD()
  } /* confidence: 35% */

/* original: mkY */ function claude_code_guest_pass_xorgani(q="claude_code_guest_pass"){
  let{
    accessToken:K,orgUUID:_
  }=await dH(),z={
    ...eY(K),"x-organization-uuid":_
  },Y=`${m7().BASE_API_URL}/api/oauth/organizations/${_}/referral/eligibility`;
  return(await O1.get(Y,{
    headers:z,params:{
      campaign:q
    },timeout:5000
  })).data
} /* confidence: 65% */

/* original: bbK */ function claude_code_guest_pass_xorgani(q="claude_code_guest_pass"){
  let{
    accessToken:K,orgUUID:_
  }=await dH(),z={
    ...eY(K),"x-organization-uuid":_
  },Y=`${m7().BASE_API_URL}/api/oauth/organizations/${_}/referral/redemptions`;
  return(await O1.get(Y,{
    headers:z,params:{
      campaign:q
    },timeout:1e4
  })).data
} /* confidence: 65% */

/* original: AUK */ function status(){
  let q,K;
  try{
    ({
      accessToken:q,orgUUID:K
    }=await dH())
  }catch{
    return!1
  }if(await dCY())return!0;
  let _=`${m7().BASE_API_URL}/v1/environment_providers/cloud/create`,z={
    ...eY(q),"x-organization-uuid":K
  };
  try{
    let Y=await O1.post(_,{
      name:"Default",kind:"anthropic_cloud",description:"Default - trusted network access",config:{
        environment_type:"anthropic",cwd:"/home/user",init_script:null,environment:{
          
        },languages:[{
          name:"python",version:"3.11"
        },{
          name:"node",version:"20"
        }],network_config:{
          allowed_hosts:[],allow_default_hosts:!0
        }
      }
    },{
      headers:z,timeout:15000,validateStatus:()=>!0
    });
    return Y.status>=200&&Y.status<300
  }catch{
    return!1
  }
} /* confidence: 70% */

/* original: wUK */ function helper_fn(){
  try{
    return await dH(),!0
  }catch{
    return!1
  }
} /* confidence: 35% */

/* original: cCY */ function auth_handler(){
  if(!await wUK())return{
    status:"not_signed_in"
  };
  let q=await Xc8();
  if(q==="not_installed")return{
    status:"gh_not_installed"
  };
  if(q==="not_authenticated")return{
    status:"gh_not_authenticated"
  };
  let{
    stdout:K
  }=await Yg("gh",["auth","token"],{
    stdout:"pipe",stderr:"ignore",timeout:5000,reject:!1
  }),_=K.trim();
  if(!_)return{
    status:"gh_not_authenticated"
  };
  return{
    status:"has_gh_token",token:new bz7(_)
  }
} /* confidence: 95% */

