// Module: UseDualStack

/* original: oe7 */ var __esModule_required_fn_argv_re=B((ie7)=>{
  Object.defineProperty(ie7,"__esModule",{
    value:!0
  });
  ie7.ruleSet=void 0;
  var de7="required",OS="fn",AS="argv",ZZ6="ref",xe7=!0,Ie7="isSet",UF6="booleanEquals",DZ6="error",fZ6="endpoint",Qn="tree",Xw1="PartitionResult",Pw1="getAttr",ue7={
    [de7]:!1,type:"string"
  },me7={
    [de7]:!0,default:!1,type:"boolean"
  },pe7={
    [ZZ6]:"Endpoint"
  },ce7={
    [OS]:UF6,[AS]:[{
      [ZZ6]:"UseFIPS"
    },!0]
  },le7={
    [OS]:UF6,[AS]:[{
      [ZZ6]:"UseDualStack"
    },!0]
  },$S={
    
  },Be7={
    [OS]:Pw1,[AS]:[{
      [ZZ6]:Xw1
    },"supportsFIPS"]
  },ne7={
    [ZZ6]:Xw1
  },ge7={
    [OS]:UF6,[AS]:[!0,{
      [OS]:Pw1,[AS]:[ne7,"supportsDualStack"]
    }]
  },Fe7=[ce7],Ue7=[le7],Qe7=[{
    [ZZ6]:"Region"
  }],PX3={
    version:"1.0",parameters:{
      Region:ue7,UseDualStack:me7,UseFIPS:me7,Endpoint:ue7
    },rules:[{
      conditions:[{
        [OS]:Ie7,[AS]:[pe7]
      }],rules:[{
        conditions:Fe7,error:"Invalid Configuration: FIPS and custom endpoint are not supported",type:DZ6
      },{
        conditions:Ue7,error:"Invalid Configuration: Dualstack and custom endpoint are not supported",type:DZ6
      },{
        endpoint:{
          url:pe7,properties:$S,headers:$S
        },type:fZ6
      }],type:Qn
    },{
      conditions:[{
        [OS]:Ie7,[AS]:Qe7
      }],rules:[{
        conditions:[{
          [OS]:"aws.partition",[AS]:Qe7,assign:Xw1
        }],rules:[{
          conditions:[ce7,le7],rules:[{
            conditions:[{
              [OS]:UF6,[AS]:[xe7,Be7]
            },ge7],rules:[{
              endpoint:{
                url:"https://oidc-fips.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:$S,headers:$S
              },type:fZ6
            }],type:Qn
          },{
            error:"FIPS and DualStack are enabled, but this partition does not support one or both",type:DZ6
          }],type:Qn
        },{
          conditions:Fe7,rules:[{
            conditions:[{
              [OS]:UF6,[AS]:[Be7,xe7]
            }],rules:[{
              conditions:[{
                [OS]:"stringEquals",[AS]:[{
                  [OS]:Pw1,[AS]:[ne7,"name"]
                },"aws-us-gov"]
              }],endpoint:{
                url:"https://oidc.{Region}.amazonaws.com",properties:$S,headers:$S
              },type:fZ6
            },{
              endpoint:{
                url:"https://oidc-fips.{Region}.{PartitionResult#dnsSuffix}",properties:$S,headers:$S
              },type:fZ6
            }],type:Qn
          },{
            error:"FIPS is enabled but this partition does not support FIPS",type:DZ6
          }],type:Qn
        },{
          conditions:Ue7,rules:[{
            conditions:[ge7],rules:[{
              endpoint:{
                url:"https://oidc.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:$S,headers:$S
              },type:fZ6
            }],type:Qn
          },{
            error:"DualStack is enabled but this partition does not support DualStack",type:DZ6
          }],type:Qn
        },{
          endpoint:{
            url:"https://oidc.{Region}.{PartitionResult#dnsSuffix}",properties:$S,headers:$S
          },type:fZ6
        }],type:Qn
      }],type:Qn
    },{
      error:"Invalid Configuration: Missing Region",type:DZ6
    }]
  };
  ie7.ruleSet=PX3
} /* confidence: 65% */

/* original: de7 */ var required_fn_argv_ref_isSet_boo="required",OS="fn",AS="argv",ZZ6="ref",xe7=!0,Ie7="isSet",UF6="booleanEquals",DZ6="error",fZ6="endpoint",Qn="tree",Xw1="PartitionResult",Pw1="getAttr",ue7={
  [required_fn_argv_ref_isSet_boo]:!1,type:"string"
} /* confidence: 65% */

/* original: te7 */ var __esModule_Endpoint_Region_Use=B((ae7)=>{
  Object.defineProperty(ae7,"__esModule",{
    value:!0
  });
  ae7.defaultEndpointResolver=void 0;
  var WX3=Sg(),Ww1=cI(),DX3=oe7(),fX3=new Ww1.EndpointCache({
    size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
  }),ZX3=(q,K={
    
  })=>{
    return fX3.get(q,()=>(0,Ww1.resolveEndpoint)(DX3.ruleSet,{
      endpointParams:q,logger:K.logger
    }))
  };
  ae7.defaultEndpointResolver=ZX3;
  Ww1.customEndpointFunctions.aws=WX3.awsEndpointFunctions
} /* confidence: 65% */

/* original: WX3 */ var Endpoint_Region_UseDualStack_U=Sg(),Ww1=cI(),DX3=oe7(),fX3=new Ww1.EndpointCache({
  size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
} /* confidence: 65% */

