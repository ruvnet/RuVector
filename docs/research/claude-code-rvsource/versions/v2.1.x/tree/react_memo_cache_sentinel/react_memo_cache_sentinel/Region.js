// Module: Region

/* original: Kqq */ var __esModule_required_type_fn_ar=B((e7q)=>{
  Object.defineProperty(e7q,"__esModule",{
    value:!0
  });
  e7q.ruleSet=void 0;
  var d7q="required",M_="type",I$="fn",u$="argv",J86="ref",b7q=!1,N21=!0,H86="booleanEquals",tf="stringEquals",c7q="sigv4",l7q="sts",n7q="us-east-1",Gj="endpoint",x7q="https://sts.{Region}.{PartitionResult#dnsSuffix}",ug="tree",RZ6="error",E21="getAttr",I7q={
    [d7q]:!1,[M_]:"string"
  },y21={
    [d7q]:!0,default:!1,[M_]:"boolean"
  },i7q={
    [J86]:"Endpoint"
  },u7q={
    [I$]:"isSet",[u$]:[{
      [J86]:"Region"
    }]
  },ef={
    [J86]:"Region"
  },m7q={
    [I$]:"aws.partition",[u$]:[ef],assign:"PartitionResult"
  },r7q={
    [J86]:"UseFIPS"
  },o7q={
    [J86]:"UseDualStack"
  },VG={
    url:"https://sts.amazonaws.com",properties:{
      authSchemes:[{
        name:c7q,signingName:l7q,signingRegion:n7q
      }]
    },headers:{
      
    }
  },zL={
    
  },p7q={
    conditions:[{
      [I$]:tf,[u$]:[ef,"aws-global"]
    }],[Gj]:VG,[M_]:Gj
  },a7q={
    [I$]:H86,[u$]:[r7q,!0]
  },s7q={
    [I$]:H86,[u$]:[o7q,!0]
  },B7q={
    [I$]:E21,[u$]:[{
      [J86]:"PartitionResult"
    },"supportsFIPS"]
  },t7q={
    [J86]:"PartitionResult"
  },g7q={
    [I$]:H86,[u$]:[!0,{
      [I$]:E21,[u$]:[t7q,"supportsDualStack"]
    }]
  },F7q=[{
    [I$]:"isSet",[u$]:[i7q]
  }],U7q=[a7q],Q7q=[s7q],zT3={
    version:"1.0",parameters:{
      Region:I7q,UseDualStack:y21,UseFIPS:y21,Endpoint:I7q,UseGlobalEndpoint:y21
    },rules:[{
      conditions:[{
        [I$]:H86,[u$]:[{
          [J86]:"UseGlobalEndpoint"
        },N21]
      },{
        [I$]:"not",[u$]:F7q
      },u7q,m7q,{
        [I$]:H86,[u$]:[r7q,b7q]
      },{
        [I$]:H86,[u$]:[o7q,b7q]
      }],rules:[{
        conditions:[{
          [I$]:tf,[u$]:[ef,"ap-northeast-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"ap-south-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"ap-southeast-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"ap-southeast-2"]
        }],endpoint:VG,[M_]:Gj
      },p7q,{
        conditions:[{
          [I$]:tf,[u$]:[ef,"ca-central-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"eu-central-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"eu-north-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"eu-west-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"eu-west-2"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"eu-west-3"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"sa-east-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,n7q]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"us-east-2"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"us-west-1"]
        }],endpoint:VG,[M_]:Gj
      },{
        conditions:[{
          [I$]:tf,[u$]:[ef,"us-west-2"]
        }],endpoint:VG,[M_]:Gj
      },{
        endpoint:{
          url:x7q,properties:{
            authSchemes:[{
              name:c7q,signingName:l7q,signingRegion:"{Region}"
            }]
          },headers:zL
        },[M_]:Gj
      }],[M_]:ug
    },{
      conditions:F7q,rules:[{
        conditions:U7q,error:"Invalid Configuration: FIPS and custom endpoint are not supported",[M_]:RZ6
      },{
        conditions:Q7q,error:"Invalid Configuration: Dualstack and custom endpoint are not supported",[M_]:RZ6
      },{
        endpoint:{
          url:i7q,properties:zL,headers:zL
        },[M_]:Gj
      }],[M_]:ug
    },{
      conditions:[u7q],rules:[{
        conditions:[m7q],rules:[{
          conditions:[a7q,s7q],rules:[{
            conditions:[{
              [I$]:H86,[u$]:[N21,B7q]
            },g7q],rules:[{
              endpoint:{
                url:"https://sts-fips.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:zL,headers:zL
              },[M_]:Gj
            }],[M_]:ug
          },{
            error:"FIPS and DualStack are enabled, but this partition does not support one or both",[M_]:RZ6
          }],[M_]:ug
        },{
          conditions:U7q,rules:[{
            conditions:[{
              [I$]:H86,[u$]:[B7q,N21]
            }],rules:[{
              conditions:[{
                [I$]:tf,[u$]:[{
                  [I$]:E21,[u$]:[t7q,"name"]
                },"aws-us-gov"]
              }],endpoint:{
                url:"https://sts.{Region}.amazonaws.com",properties:zL,headers:zL
              },[M_]:Gj
            },{
              endpoint:{
                url:"https://sts-fips.{Region}.{PartitionResult#dnsSuffix}",properties:zL,headers:zL
              },[M_]:Gj
            }],[M_]:ug
          },{
            error:"FIPS is enabled but this partition does not support FIPS",[M_]:RZ6
          }],[M_]:ug
        },{
          conditions:Q7q,rules:[{
            conditions:[g7q],rules:[{
              endpoint:{
                url:"https://sts.{Region}.{PartitionResult#dualStackDnsSuffix}",properties:zL,headers:zL
              },[M_]:Gj
            }],[M_]:ug
          },{
            error:"DualStack is enabled but this partition does not support DualStack",[M_]:RZ6
          }],[M_]:ug
        },p7q,{
          endpoint:{
            url:x7q,properties:zL,headers:zL
          },[M_]:Gj
        }],[M_]:ug
      }],[M_]:ug
    },{
      error:"Invalid Configuration: Missing Region",[M_]:RZ6
    }]
  };
  e7q.ruleSet=zT3
} /* confidence: 65% */

/* original: d7q */ var required_type_fn_argv_ref_bool="required",M_="type",I$="fn",u$="argv",J86="ref",b7q=!1,N21=!0,H86="booleanEquals",tf="stringEquals",c7q="sigv4",l7q="sts",n7q="us-east-1",Gj="endpoint",x7q="https://sts.{Region}.{PartitionResult#dnsSuffix}",ug="tree",RZ6="error",E21="getAttr",I7q={
  [required_type_fn_argv_ref_bool]:!1,[M_]:"string"
} /* confidence: 65% */

/* original: Yqq */ var __esModule_Endpoint_Region_Use=B((_qq)=>{
  Object.defineProperty(_qq,"__esModule",{
    value:!0
  });
  _qq.defaultEndpointResolver=void 0;
  var YT3=Sg(),L21=cI(),$T3=Kqq(),OT3=new L21.EndpointCache({
    size:50,params:["Endpoint","Region","UseDualStack","UseFIPS","UseGlobalEndpoint"]
  }),AT3=(q,K={
    
  })=>{
    return OT3.get(q,()=>(0,L21.resolveEndpoint)($T3.ruleSet,{
      endpointParams:q,logger:K.logger
    }))
  };
  _qq.defaultEndpointResolver=AT3;
  L21.customEndpointFunctions.aws=YT3.awsEndpointFunctions
} /* confidence: 65% */

/* original: YT3 */ var Endpoint_Region_UseDualStack_U=Sg(),L21=cI(),$T3=Kqq(),OT3=new L21.EndpointCache({
  size:50,params:["Endpoint","Region","UseDualStack","UseFIPS","UseGlobalEndpoint"]
} /* confidence: 65% */

