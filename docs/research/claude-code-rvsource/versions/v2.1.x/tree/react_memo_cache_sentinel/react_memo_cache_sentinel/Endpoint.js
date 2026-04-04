// Module: Endpoint

/* original: x8q */ var __esModule_required_fn_argv_re=B((C8q)=>{
  Object.defineProperty(C8q,"__esModule",{
    value:!0
  });
  C8q.ruleSet=void 0;
  var L8q="required",jS="fn",HS="argv",VZ6="ref",f8q=!0,Z8q="isSet",iF6="booleanEquals",TZ6="error",kZ6="endpoint",cn="tree",Uw1="PartitionResult",Qw1="getAttr",G8q={
    [L8q]:!1,type:"string"
  },v8q={
    [L8q]:!0,default:!1,type:"boolean"
  },T8q={
    [VZ6]:"Endpoint"
  },h8q={
    [jS]:iF6,[HS]:[{
      [VZ6]:"UseFIPS"
    },!0]
  },R8q={
    [jS]:iF6,[HS]:[{
      [VZ6]:"UseDualStack"
    },!0]
  },wS={
    
  },k8q={
    [jS]:Qw1,[HS]:[{
      [VZ6]:Uw1
    },"supportsFIPS"]
  },S8q={
    [VZ6]:Uw1
  },V8q={
    [jS]:iF6,[HS]:[!0,{
      [jS]:Qw1,[HS]:[S8q,"supportsDualStack"]
    }]
  },N8q=[h8q],y8q=[R8q],E8q=[{
    [VZ6]:"Region"
  }],Sf3={
    version:"1.0",parameters:{
      Region:G8q,UseDualStack:v8q,UseFIPS:v8q,Endpoint:G8q
    },rules:[{
      conditions:[{
        [jS]:Z8q,[HS]:[T8q]
      }],rules:[{
        conditions:N8q,error:"Invalid Configuration: FIPS and custom endpoint are not supported",type:TZ6
      },{
        conditions:y8q,error:"Invalid Configuration: Dualstack and custom endpoint are not supported",type:TZ6
      },{
        endpoint:{
          url:T8q,properties:wS,headers:wS
        },type:kZ6
      }],type:cn
    },{
      conditions:[{
        [jS]:Z8q,[HS]:E8q
      }],rules:[{
        conditions:[{
          [jS]:"aws.partition",[HS]:E8q,assign:Uw1
        }],rules:[{
          conditions:[h8q,R8q],rules:[{
            conditions:[{
              [jS]:iF6,[HS]:[f8q,k8q]
            },V8q],rules:[{
              endpoint:{
                url:"https://portal.sso-fips.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:wS,headers:wS
              },type:kZ6
            }],type:cn
          },{
            error:"FIPS and DualStack are enabled, but this partition does not support one or both",type:TZ6
          }],type:cn
        },{
          conditions:N8q,rules:[{
            conditions:[{
              [jS]:iF6,[HS]:[k8q,f8q]
            }],rules:[{
              conditions:[{
                [jS]:"stringEquals",[HS]:[{
                  [jS]:Qw1,[HS]:[S8q,"name"]
                },"aws-us-gov"]
              }],endpoint:{
                url:"https://portal.sso.{Region}.amazonaws.com",properties:wS,headers:wS
              },type:kZ6
            },{
              endpoint:{
                url:"https://portal.sso-fips.{Region}.{PartitionResult#dnsSuffix}",properties:wS,headers:wS
              },type:kZ6
            }],type:cn
          },{
            error:"FIPS is enabled but this partition does not support FIPS",type:TZ6
          }],type:cn
        },{
          conditions:y8q,rules:[{
            conditions:[V8q],rules:[{
              endpoint:{
                url:"https://portal.sso.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:wS,headers:wS
              },type:kZ6
            }],type:cn
          },{
            error:"DualStack is enabled but this partition does not support DualStack",type:TZ6
          }],type:cn
        },{
          endpoint:{
            url:"https://portal.sso.{Region}.{PartitionResult#dnsSuffix}",properties:wS,headers:wS
          },type:kZ6
        }],type:cn
      }],type:cn
    },{
      error:"Invalid Configuration: Missing Region",type:TZ6
    }]
  };
  C8q.ruleSet=Sf3
} /* confidence: 65% */

/* original: L8q */ var required_fn_argv_ref_isSet_boo="required",jS="fn",HS="argv",VZ6="ref",f8q=!0,Z8q="isSet",iF6="booleanEquals",TZ6="error",kZ6="endpoint",cn="tree",Uw1="PartitionResult",Qw1="getAttr",G8q={
  [required_fn_argv_ref_isSet_boo]:!1,type:"string"
} /* confidence: 65% */

/* original: m8q */ var __esModule_Endpoint_Region_Use=B((I8q)=>{
  Object.defineProperty(I8q,"__esModule",{
    value:!0
  });
  I8q.defaultEndpointResolver=void 0;
  var Cf3=Sg(),dw1=cI(),bf3=x8q(),xf3=new dw1.EndpointCache({
    size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
  }),If3=(q,K={
    
  })=>{
    return xf3.get(q,()=>(0,dw1.resolveEndpoint)(bf3.ruleSet,{
      endpointParams:q,logger:K.logger
    }))
  };
  I8q.defaultEndpointResolver=If3;
  dw1.customEndpointFunctions.aws=Cf3.awsEndpointFunctions
} /* confidence: 65% */

/* original: Cf3 */ var Endpoint_Region_UseDualStack_U=Sg(),dw1=cI(),bf3=x8q(),xf3=new dw1.EndpointCache({
  size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
} /* confidence: 65% */

