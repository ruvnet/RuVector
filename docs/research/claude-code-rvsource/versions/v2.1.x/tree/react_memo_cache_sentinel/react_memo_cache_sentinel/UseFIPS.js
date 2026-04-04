// Module: UseFIPS

/* original: tzq */ var __esModule_required_fn_argv_re=B((azq)=>{
  Object.defineProperty(azq,"__esModule",{
    value:!0
  });
  azq.ruleSet=void 0;
  var izq="required",lg="fn",ng="argv",nZ6="ref",pzq=!0,Bzq="isSet",bU6="booleanEquals",lZ6="error",CU6="endpoint",VT="tree",IM1="PartitionResult",gzq={
    [izq]:!1,type:"string"
  },Fzq={
    [izq]:!0,default:!1,type:"boolean"
  },Uzq={
    [nZ6]:"Endpoint"
  },rzq={
    [lg]:bU6,[ng]:[{
      [nZ6]:"UseFIPS"
    },!0]
  },ozq={
    [lg]:bU6,[ng]:[{
      [nZ6]:"UseDualStack"
    },!0]
  },cg={
    
  },Qzq={
    [lg]:"getAttr",[ng]:[{
      [nZ6]:IM1
    },"supportsFIPS"]
  },dzq={
    [lg]:bU6,[ng]:[!0,{
      [lg]:"getAttr",[ng]:[{
        [nZ6]:IM1
      },"supportsDualStack"]
    }]
  },czq=[rzq],lzq=[ozq],nzq=[{
    [nZ6]:"Region"
  }],H_9={
    version:"1.0",parameters:{
      Region:gzq,UseDualStack:Fzq,UseFIPS:Fzq,Endpoint:gzq
    },rules:[{
      conditions:[{
        [lg]:Bzq,[ng]:[Uzq]
      }],rules:[{
        conditions:czq,error:"Invalid Configuration: FIPS and custom endpoint are not supported",type:lZ6
      },{
        rules:[{
          conditions:lzq,error:"Invalid Configuration: Dualstack and custom endpoint are not supported",type:lZ6
        },{
          endpoint:{
            url:Uzq,properties:cg,headers:cg
          },type:CU6
        }],type:VT
      }],type:VT
    },{
      rules:[{
        conditions:[{
          [lg]:Bzq,[ng]:nzq
        }],rules:[{
          conditions:[{
            [lg]:"aws.partition",[ng]:nzq,assign:IM1
          }],rules:[{
            conditions:[rzq,ozq],rules:[{
              conditions:[{
                [lg]:bU6,[ng]:[pzq,Qzq]
              },dzq],rules:[{
                rules:[{
                  endpoint:{
                    url:"https://bedrock-runtime-fips.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:cg,headers:cg
                  },type:CU6
                }],type:VT
              }],type:VT
            },{
              error:"FIPS and DualStack are enabled, but this partition does not support one or both",type:lZ6
            }],type:VT
          },{
            conditions:czq,rules:[{
              conditions:[{
                [lg]:bU6,[ng]:[Qzq,pzq]
              }],rules:[{
                rules:[{
                  endpoint:{
                    url:"https://bedrock-runtime-fips.{Region}.{PartitionResult#dnsSuffix}",properties:cg,headers:cg
                  },type:CU6
                }],type:VT
              }],type:VT
            },{
              error:"FIPS is enabled but this partition does not support FIPS",type:lZ6
            }],type:VT
          },{
            conditions:lzq,rules:[{
              conditions:[dzq],rules:[{
                rules:[{
                  endpoint:{
                    url:"https://bedrock-runtime.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:cg,headers:cg
                  },type:CU6
                }],type:VT
              }],type:VT
            },{
              error:"DualStack is enabled but this partition does not support DualStack",type:lZ6
            }],type:VT
          },{
            rules:[{
              endpoint:{
                url:"https://bedrock-runtime.{Region}.{PartitionResult#dnsSuffix}",properties:cg,headers:cg
              },type:CU6
            }],type:VT
          }],type:VT
        }],type:VT
      },{
        error:"Invalid Configuration: Missing Region",type:lZ6
      }],type:VT
    }]
  };
  azq.ruleSet=H_9
} /* confidence: 65% */

/* original: izq */ var required_fn_argv_ref_isSet_boo="required",lg="fn",ng="argv",nZ6="ref",pzq=!0,Bzq="isSet",bU6="booleanEquals",lZ6="error",CU6="endpoint",VT="tree",IM1="PartitionResult",gzq={
  [required_fn_argv_ref_isSet_boo]:!1,type:"string"
} /* confidence: 65% */

/* original: KYq */ var __esModule_Endpoint_Region_Use=B((ezq)=>{
  Object.defineProperty(ezq,"__esModule",{
    value:!0
  });
  ezq.defaultEndpointResolver=void 0;
  var J_9=Sg(),uM1=cI(),M_9=tzq(),X_9=new uM1.EndpointCache({
    size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
  }),P_9=(q,K={
    
  })=>{
    return X_9.get(q,()=>(0,uM1.resolveEndpoint)(M_9.ruleSet,{
      endpointParams:q,logger:K.logger
    }))
  };
  ezq.defaultEndpointResolver=P_9;
  uM1.customEndpointFunctions.aws=J_9.awsEndpointFunctions
} /* confidence: 65% */

/* original: J_9 */ var Endpoint_Region_UseDualStack_U=Sg(),uM1=cI(),M_9=tzq(),X_9=new uM1.EndpointCache({
  size:50,params:["Endpoint","Region","UseDualStack","UseFIPS"]
} /* confidence: 65% */

