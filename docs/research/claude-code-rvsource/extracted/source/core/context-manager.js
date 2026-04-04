// ===================================================================
// Module: context-manager
// Source: @anthropic-ai/claude-code@2.1.91
// Confidence: 0.082
// Fragments: 49
// Extracted: 2026-04-03T03:17:18.016Z
// ===================================================================

if(G8.slowOperations.some((K)=>q-K.timestamp>=Ko8)){if(G8.slowOperations=G8.slowOperations.filter((K)=>q-K.timestamp<Ko8),G8.slowOperations.length===0)return cJ7}return G8.slowOperations}function xB(){return G8.mainThreadAgentType}function Pl(q){G8.mainThreadAgentType=q}function _5(){return G8.isRemoteMode}function Wa8(q){G8.isRemoteMode=q}function Da8(){return G8.systemPromptSectionCache}function fa8(q,K){G8.systemPromptSectionCache.set(q,K)}function Za8(){G8.systemPromptSectionCache.clear()}function Ga8(){return G8.lastEmittedDate}function DP6(q){G8.lastEmittedDate=q}function t0(){return G8.additionalDirectoriesForClaudeMd}function PI6(q){G8.additionalDirectoriesForClaudeMd=q}function wJ(){return G8.allowedChannels}function Wl(q){G8.allowedChannels=q}function s98(){return G8.hasDevChannels}function t98(q){G8.hasDevChannels=q}function va8(){return G8.promptCache1hAllowlist}function Ta8(q){G8.promptCache1hAllowlist=q}function ka8(){return G8.afkModeHeaderLatched}function Va8(q){G8.afkModeHeaderLatched=q}function Na8(){return G8.fastModeHeaderLatched}function ya8(q){G8.fastModeHeaderLatched=q}function Ea8(){return G8.cacheEditingHeaderLatched}function bO5(q){G8.cacheEditingHeaderLatched=q}function La8(){return G8.thinkingClearLatched}function ha8(q){G8.thinkingClearLatched=q}function Ra8(){G8.afkModeHeaderLatched=null,G8.fastModeHeaderLatched=null,G8.cacheEditingHeaderLatched=null,G8.thinkingClearLatched=null}function WI6(){return G8.promptId}function DI6(q){G8.promptId=q}var G8,Yo8,$o8,Do8=!1,Zo8,Go8,To8=0,ko8=null,m98=0,C98=!1,lx6,iJ7=150,dJ7=10,Ko8=1e4,cJ7;var T8=L(()=>{xJ7();sr8();Jl();G8=lJ7();Yo8=L_(),$o8=Yo8.subscribe;Zo8=L_(),Go8=Zo8.subscribe;cJ7=[]});function q_8(q){let K;for(let _ in q)if(_.startsWith("_PROTO_")){if(K===void 0)K={...q};delete K[_]}return K??q}function eJ7(q){if(We!==null)return;if(We=q,fI6.length>0){let K=[...fI6];fI6.length=0,queueMicrotask(()=>{for(let _ of K)if(_.async)We.logEventAsync(_.eventName,_.metadata);

J4(this,yP6,!0,"f"),J4(this,l96,!0,"f"),J4(this,vl,void 0,"f");try{while(!0){let K;try{if(x1(this,EP,"f").params.max_iterations&&x1(this,lI6,"f")>=x1(this,EP,"f").params.max_iterations)break;J4(this,l96,!1,"f"),J4(this,vl,void 0,"f"),J4(this,lI6,(q=x1(this,lI6,"f"),q++,q),"f"),J4(this,yE,void 0,"f");let{max_iterations:_,compactionControl:z,...Y}=x1(this,EP,"f").params;if(Y.stream)K=this.client.beta.messages.stream({...Y},x1(this,cI6,"f")),J4(this,yE,K.finalMessage(),"f"),x1(this,yE,"f").catch(()=>{}),yield K;else J4(this,yE,this.client.beta.messages.create({...Y,stream:!1},x1(this,cI6,"f")),"f"),yield x1(this,yE,"f");if(!await x1(this,dI6,"m",nM7).call(this)){if(!x1(this,l96,"f")){let{role:A,content:w}=await x1(this,yE,"f");x1(this,EP,"f").params.messages.push({role:A,content:w})}let O=await x1(this,dI6,"m",As8).call(this,x1(this,EP,"f").params.messages.at(-1));if(O)x1(this,EP,"f").params.messages.push(O);else if(!x1(this,l96,"f"))break}}finally{if(K)K.abort()}}if(!x1(this,yE,"f"))throw new mq("ToolRunner concluded without a message from the server");x1(this,ke,"f").resolve(await x1(this,yE,"f"))}catch(K){throw J4(this,yP6,!1,"f"),x1(this,ke,"f").promise.catch(()=>{}),x1(this,ke,"f").reject(K),J4(this,ke,iM7(),"f"),K}}setMessagesParams(q){if(typeof q==="function")x1(this,EP,"f").params=q(x1(this,EP,"f").params);else x1(this,EP,"f").params=q;J4(this,l96,!0,"f"),J4(this,vl,void 0,"f")}async generateToolResponse(){let q=await x1(this,yE,"f")??this.params.messages.at(-1);if(!q)return null;return x1(this,dI6,"m",As8).call(this,q)}done(){return x1(this,ke,"f").promise}async runUntilDone(){if(!x1(this,yP6,"f"))for await(let q of this);return this.done()}get params(){return x1(this,EP,"f").params}pushMessages(...q){this.setMessagesParams((K)=>({...K,messages:[...K.messages,...q]}))}then(q,K){return this.runUntilDone().then(q,K)}};As8=async function(K){if(x1(this,vl,"f")!==void 0)return x1(this,vl,"f");return J4(this,vl,JA5(x1(this,EP,"f").params,K),"f"),x1(this,vl,"f")}});var EP6;var js8=L(()=>{FW();Ba8();

return K.split(`
`).filter((_)=>_.startsWith("worktree ")).map((_)=>_.slice(9).normalize("NFC"))}catch{return[]}}var gR5;var wA8=L(()=>{gR5=BR5(pR5)});import{open as Ch7,readdir as zm$,realpath as Ym$,stat as $m$}from"fs/promises";import{join as bh7}from"path";function xh7(q){if(typeof q!=="string")return null;return FR5.test(q)?q:null}function Ih7(q){if(!q.includes("\\"))return q;try{return JSON.parse(`"${q}"`)}catch{return q}}function P66(q,K){let _=[`"${K}":"`,`"${K}": "`];for(let z of _){let Y=q.indexOf(z);if(Y<0)continue;let $=Y+z.length,O=$;while(O<q.length){if(q[O]==="\\"){O+=2;continue}if(q[O]==='"')return Ih7(q.slice($,O));O++}}return}function OT(q,K){let _=[`"${K}":"`,`"${K}": "`],z;for(let Y of _){let $=0;while(!0){let O=q.indexOf(Y,$);if(O<0)break;let A=O+Y.length,w=A;while(w<q.length){if(q[w]==="\\"){w+=2;continue}if(q[w]==='"'){z=Ih7(q.slice(A,w));break}w++}$=w+1}}return z}async function uh7(q,K,_){try{let z=await Ch7(q,"r");try{let Y=await z.read(_,0,X66,0);if(Y.bytesRead===0)return{head:"",tail:""};let $=_.toString("utf8",0,Y.bytesRead),O=Math.max(0,K-X66),A=$;if(O>0){let w=await z.read(_,0,X66,O);A=_.toString("utf8",0,w.bytesRead)}return{head:$,tail:A}}finally{await z.close()}}catch{return{head:"",tail:""}}}function UR5(q){return Math.abs(Q_6(q)).toString(36)}function XX(q){let K=q.replace(/[^a-zA-Z0-9]/g,"-");if(K.length<=_51)return K;let _=typeof Bun<"u"?Bun.hash(q).toString(36):UR5(q);return`${K.slice(0,_51)}-${_}`}function JA8(){return bh7(q7(),"projects")}function mh7(q){return bh7(JA8(),XX(q))}function cR5(){return dR5??=Buffer.from('"compact_boundary"')}function ph7(q){try{let K=JSON.parse(q);if(K.type!=="system"||K.subtype!=="compact_boundary")return null;return{hasPreservedSegment:Boolean(K.compactMetadata?.preservedSegment)}}catch{return null}}function i_6(q,K,_,z){let Y=z-_;if(Y<=0)return;if(q.len+Y>q.buf.length){let $=Buffer.allocUnsafe(Math.min(Math.max(q.buf.length*2,q.len+Y),q.cap));

Object.setPrototypeOf(this,q.prototype)}},LE3="AgreementAvailability",hE3="AccessDeniedException",RE3="AutomatedEvaluationConfig",SE3="AutomatedEvaluationCustomMetrics",CE3="AutomatedEvaluationCustomMetricConfig",bE3="AutomatedEvaluationCustomMetricSource",xE3="AutomatedReasoningCheckDifferenceScenarioList",IE3="AutomatedReasoningCheckFinding",uE3="AutomatedReasoningCheckFindingList",mE3="AutomatedReasoningCheckImpossibleFinding",pE3="AutomatedReasoningCheckInvalidFinding",BE3="AutomatedReasoningCheckInputTextReference",gE3="AutomatedReasoningCheckInputTextReferenceList",FE3="AutomatedReasoningCheckLogicWarning",UE3="AutomatedReasoningCheckNoTranslationsFinding",QE3="AutomatedReasoningCheckRule",dE3="AutomatedReasoningCheckRuleList",cE3="AutomatedReasoningCheckScenario",lE3="AutomatedReasoningCheckSatisfiableFinding",nE3="AutomatedReasoningCheckTranslation",iE3="AutomatedReasoningCheckTranslationAmbiguousFinding",rE3="AutomatedReasoningCheckTooComplexFinding",oE3="AutomatedReasoningCheckTranslationList",aE3="AutomatedReasoningCheckTranslationOption",sE3="AutomatedReasoningCheckTranslationOptionList",tE3="AutomatedReasoningCheckValidFinding",eE3="AutomatedReasoningLogicStatement",qL3="AutomatedReasoningLogicStatementContent",KL3="AutomatedReasoningLogicStatementList",_L3="AutomatedReasoningNaturalLanguageStatementContent",zL3="AutomatedReasoningPolicyAnnotation",YL3="AutomatedReasoningPolicyAnnotationFeedbackNaturalLanguage",$L3="AutomatedReasoningPolicyAnnotationIngestContent",OL3="AutomatedReasoningPolicyAnnotationList",AL3="AutomatedReasoningPolicyAddRuleAnnotation",wL3="AutomatedReasoningPolicyAddRuleFromNaturalLanguageAnnotation",jL3="AutomatedReasoningPolicyAddRuleMutation",HL3="AutomatedReasoningPolicyAnnotationRuleNaturalLanguage",JL3="AutomatedReasoningPolicyAddTypeAnnotation",ML3="AutomatedReasoningPolicyAddTypeMutation",XL3="AutomatedReasoningPolicyAddTypeValue",PL3="AutomatedReasoningPolicyAddVariableAnnotation",WL3="AutomatedReasoningPolicyAddVariableMutation",DL3="AutomatedReasoningPolicyBuildDocumentBlob",fL3="AutomatedReasoningPolicyBuildDocumentDescription",ZL3="AutomatedReasoningPolicyBuildDocumentName",GL3="AutomatedReasoningPolicyBuildLog",vL3="AutomatedReasoningPolicyBuildLogEntry",TL3="AutomatedReasoningPolicyBuildLogEntryList",kL3="AutomatedReasoningPolicyBuildResultAssets",VL3="AutomatedReasoningPolicyBuildStep",NL3="AutomatedReasoningPolicyBuildStepContext",yL3="AutomatedReasoningPolicyBuildStepList",EL3="AutomatedReasoningPolicyBuildStepMessage",LL3="AutomatedReasoningPolicyBuildStepMessageList",hL3="AutomatedReasoningPolicyBuildWorkflowDocument",RL3="AutomatedReasoningPolicyBuildWorkflowDocumentList",SL3="AutomatedReasoningPolicyBuildWorkflowRepairContent",CL3="AutomatedReasoningPolicyBuildWorkflowSource",bL3="AutomatedReasoningPolicyBuildWorkflowSummary",xL3="AutomatedReasoningPolicyBuildWorkflowSummaries",IL3="AutomatedReasoningPolicyDescription",uL3="AutomatedReasoningPolicyDefinitionElement",mL3="AutomatedReasoningPolicyDefinitionQualityReport",pL3="AutomatedReasoningPolicyDefinitionRule",BL3="AutomatedReasoningPolicyDeleteRuleAnnotation",gL3="AutomatedReasoningPolicyDefinitionRuleAlternateExpression",FL3="AutomatedReasoningPolicyDefinitionRuleExpression",UL3="AutomatedReasoningPolicyDefinitionRuleList",QL3="AutomatedReasoningPolicyDeleteRuleMutation",dL3="AutomatedReasoningPolicyDisjointRuleSet",cL3="AutomatedReasoningPolicyDisjointRuleSetList",lL3="AutomatedReasoningPolicyDefinitionType",nL3="AutomatedReasoningPolicyDeleteTypeAnnotation",iL3="AutomatedReasoningPolicyDefinitionTypeDescription",rL3="AutomatedReasoningPolicyDefinitionTypeList",oL3="AutomatedReasoningPolicyDeleteTypeMutation",aL3="AutomatedReasoningPolicyDefinitionTypeName",sL3="AutomatedReasoningPolicyDefinitionTypeNameList",tL3="AutomatedReasoningPolicyDefinitionTypeValue",eL3="AutomatedReasoningPolicyDefinitionTypeValueDescription",qh3="AutomatedReasoningPolicyDefinitionTypeValueList",Kh3="AutomatedReasoningPolicyDefinitionTypeValuePair",_h3="AutomatedReasoningPolicyDefinitionTypeValuePairList",zh3="AutomatedReasoningPolicyDeleteTypeValue",Yh3="AutomatedReasoningPolicyDefinitionVariable",$h3="AutomatedReasoningPolicyDeleteVariableAnnotation",Oh3="AutomatedReasoningPolicyDefinitionVariableDescription",Ah3="AutomatedReasoningPolicyDefinitionVariableList",wh3="AutomatedReasoningPolicyDeleteVariableMutation",jh3="AutomatedReasoningPolicyDefinitionVariableName",Hh3="AutomatedReasoningPolicyDefinitionVariableNameList",Jh3="AutomatedReasoningPolicyDefinition",Mh3="AutomatedReasoningPolicyGeneratedTestCase",Xh3="AutomatedReasoningPolicyGeneratedTestCaseList",Ph3="AutomatedReasoningPolicyGeneratedTestCases",Wh3="AutomatedReasoningPolicyIngestContentAnnotation",Dh3="AutomatedReasoningPolicyMutation",fh3="AutomatedReasoningPolicyName",Zh3="AutomatedReasoningPolicyPlanning",Gh3="AutomatedReasoningPolicyScenario",vh3="AutomatedReasoningPolicyScenarioAlternateExpression",Th3="AutomatedReasoningPolicyScenarioExpression",kh3="AutomatedReasoningPolicySummary",Vh3="AutomatedReasoningPolicySummaries",Nh3="AutomatedReasoningPolicyTestCase",yh3="AutomatedReasoningPolicyTestCaseList",Eh3="AutomatedReasoningPolicyTestGuardContent",Lh3="AutomatedReasoningPolicyTestList",hh3="AutomatedReasoningPolicyTestQueryContent",Rh3="AutomatedReasoningPolicyTestResult",Sh3="AutomatedReasoningPolicyTypeValueAnnotation",Ch3="AutomatedReasoningPolicyTypeValueAnnotationList",bh3="AutomatedReasoningPolicyUpdateFromRuleFeedbackAnnotation",xh3="AutomatedReasoningPolicyUpdateFromScenarioFeedbackAnnotation",Ih3="AutomatedReasoningPolicyUpdateRuleAnnotation",uh3="AutomatedReasoningPolicyUpdateRuleMutation",mh3="AutomatedReasoningPolicyUpdateTypeAnnotation",ph3="AutomatedReasoningPolicyUpdateTypeMutation",Bh3="AutomatedReasoningPolicyUpdateTypeValue",gh3="AutomatedReasoningPolicyUpdateVariableAnnotation",Fh3="AutomatedReasoningPolicyUpdateVariableMutation",Uh3="AutomatedReasoningPolicyWorkflowTypeContent",Qh3="ByteContentBlob",dh3="ByteContentDoc",ch3="BatchDeleteEvaluationJob",lh3="BatchDeleteEvaluationJobError",nh3="BatchDeleteEvaluationJobErrors",ih3="BatchDeleteEvaluationJobItem",rh3="BatchDeleteEvaluationJobItems",oh3="BatchDeleteEvaluationJobRequest",ah3="BatchDeleteEvaluationJobResponse",sh3="BedrockEvaluatorModel",th3="BedrockEvaluatorModels",eh3="CreateAutomatedReasoningPolicy",qR3="CancelAutomatedReasoningPolicyBuildWorkflow",KR3="CancelAutomatedReasoningPolicyBuildWorkflowRequest",_R3="CancelAutomatedReasoningPolicyBuildWorkflowResponse",zR3="CreateAutomatedReasoningPolicyRequest",YR3="CreateAutomatedReasoningPolicyResponse",$R3="CreateAutomatedReasoningPolicyTestCase",OR3="CreateAutomatedReasoningPolicyTestCaseRequest",AR3="CreateAutomatedReasoningPolicyTestCaseResponse",wR3="CreateAutomatedReasoningPolicyVersion",jR3="CreateAutomatedReasoningPolicyVersionRequest",HR3="CreateAutomatedReasoningPolicyVersionResponse",JR3="CustomizationConfig",MR3="CreateCustomModel",XR3="CreateCustomModelDeployment",PR3="CreateCustomModelDeploymentRequest",WR3="CreateCustomModelDeploymentResponse",DR3="CreateCustomModelRequest",fR3="CreateCustomModelResponse",ZR3="ConflictException",GR3="CreateEvaluationJob",vR3="CreateEvaluationJobRequest",TR3="CreateEvaluationJobResponse",kR3="CreateFoundationModelAgreement",VR3="CreateFoundationModelAgreementRequest",NR3="CreateFoundationModelAgreementResponse",yR3="CreateGuardrail",ER3="CreateGuardrailRequest",LR3="CreateGuardrailResponse",hR3="CreateGuardrailVersion",RR3="CreateGuardrailVersionRequest",SR3="CreateGuardrailVersionResponse",CR3="CreateInferenceProfile",bR3="CreateInferenceProfileRequest",xR3="CreateInferenceProfileResponse",IR3="CustomMetricBedrockEvaluatorModel",uR3="CustomMetricBedrockEvaluatorModels",mR3="CreateModelCopyJob",pR3="CreateModelCopyJobRequest",BR3="CreateModelCopyJobResponse",gR3="CreateModelCustomizationJobRequest",FR3="CreateModelCustomizationJobResponse",UR3="CreateModelCustomizationJob",QR3="CustomMetricDefinition",dR3="CustomModelDeploymentSummary",cR3="CustomModelDeploymentSummaryList",lR3="CustomMetricEvaluatorModelConfig",nR3="CreateModelImportJob",iR3="CreateModelImportJobRequest",rR3="CreateModelImportJobResponse",oR3="CreateModelInvocationJobRequest",aR3="CreateModelInvocationJobResponse",sR3="CreateModelInvocationJob",tR3="CreateMarketplaceModelEndpoint",eR3="CreateMarketplaceModelEndpointRequest",qS3="CreateMarketplaceModelEndpointResponse",KS3="CustomModelSummary",_S3="CustomModelSummaryList",zS3="CustomModelUnits",YS3="CreateProvisionedModelThroughput",$S3="CreateProvisionedModelThroughputRequest",OS3="CreateProvisionedModelThroughputResponse",AS3="CreatePromptRouter",wS3="CreatePromptRouterRequest",jS3="CreatePromptRouterResponse",HS3="CloudWatchConfig",JS3="DeleteAutomatedReasoningPolicy",MS3="DeleteAutomatedReasoningPolicyBuildWorkflow",XS3="DeleteAutomatedReasoningPolicyBuildWorkflowRequest",PS3="DeleteAutomatedReasoningPolicyBuildWorkflowResponse",WS3="DeleteAutomatedReasoningPolicyRequest",DS3="DeleteAutomatedReasoningPolicyResponse",fS3="DeleteAutomatedReasoningPolicyTestCase",ZS3="DeleteAutomatedReasoningPolicyTestCaseRequest",GS3="DeleteAutomatedReasoningPolicyTestCaseResponse",vS3="DistillationConfig",TS3="DeleteCustomModel",kS3="DeleteCustomModelDeployment",VS3="DeleteCustomModelDeploymentRequest",NS3="DeleteCustomModelDeploymentResponse",yS3="DeleteCustomModelRequest",ES3="DeleteCustomModelResponse",LS3="DeleteFoundationModelAgreement",hS3="DeleteFoundationModelAgreementRequest",RS3="DeleteFoundationModelAgreementResponse",SS3="DeleteGuardrail",CS3="DeleteGuardrailRequest",bS3="DeleteGuardrailResponse",xS3="DeleteImportedModel",IS3="DeleteImportedModelRequest",uS3="DeleteImportedModelResponse",mS3="DeleteInferenceProfile",pS3="DeleteInferenceProfileRequest",BS3="DeleteInferenceProfileResponse",gS3="DeleteModelInvocationLoggingConfiguration",FS3="DeleteModelInvocationLoggingConfigurationRequest",US3="DeleteModelInvocationLoggingConfigurationResponse",QS3="DeleteMarketplaceModelEndpoint",dS3="DeleteMarketplaceModelEndpointRequest",cS3="DeleteMarketplaceModelEndpointResponse",lS3="DeregisterMarketplaceModelEndpointRequest",nS3="DeregisterMarketplaceModelEndpointResponse",iS3="DeregisterMarketplaceModelEndpoint",rS3="DataProcessingDetails",oS3="DeleteProvisionedModelThroughput",aS3="DeleteProvisionedModelThroughputRequest",sS3="DeleteProvisionedModelThroughputResponse",tS3="DimensionalPriceRate",eS3="DeletePromptRouterRequest",qC3="DeletePromptRouterResponse",KC3="DeletePromptRouter",_C3="ExportAutomatedReasoningPolicyVersion",zC3="ExportAutomatedReasoningPolicyVersionRequest",YC3="ExportAutomatedReasoningPolicyVersionResponse",$C3="EvaluationBedrockModel",OC3="EndpointConfig",AC3="EvaluationConfig",wC3="EvaluationDataset",jC3="EvaluationDatasetLocation",HC3="EvaluationDatasetMetricConfig",JC3="EvaluationDatasetMetricConfigs",MC3="EvaluationDatasetName",XC3="EvaluationInferenceConfig",PC3="EvaluationInferenceConfigSummary",WC3="EvaluationJobDescription",DC3="EvaluationJobIdentifier",fC3="EvaluationJobIdentifiers",ZC3="EvaluationModelConfigs",GC3="EvaluationModelConfigSummary",vC3="EvaluationModelConfig",TC3="EvaluatorModelConfig",kC3="EvaluationMetricDescription",VC3="EvaluationModelInferenceParams",NC3="EvaluationMetricName",yC3="EvaluationMetricNames",EC3="EvaluationOutputDataConfig",LC3="EvaluationPrecomputedInferenceSource",hC3="EvaluationPrecomputedRetrieveAndGenerateSourceConfig",RC3="EvaluationPrecomputedRetrieveSourceConfig",SC3="EvaluationPrecomputedRagSourceConfig",CC3="EvaluationRagConfigSummary",bC3="EvaluationSummary",xC3="ExternalSourcesGenerationConfiguration",IC3="ExternalSourcesRetrieveAndGenerateConfiguration",uC3="EvaluationSummaries",mC3="ExternalSource",pC3="ExternalSources",BC3="FilterAttribute",gC3="FieldForReranking",FC3="FieldsForReranking",UC3="FoundationModelDetails",QC3="FoundationModelLifecycle",dC3="FoundationModelSummary",cC3="FoundationModelSummaryList",lC3="GuardrailAutomatedReasoningPolicy",nC3="GetAutomatedReasoningPolicyAnnotations",iC3="GetAutomatedReasoningPolicyAnnotationsRequest",rC3="GetAutomatedReasoningPolicyAnnotationsResponse",oC3="GetAutomatedReasoningPolicyBuildWorkflow",aC3="GetAutomatedReasoningPolicyBuildWorkflowRequest",sC3="GetAutomatedReasoningPolicyBuildWorkflowResultAssets",tC3="GetAutomatedReasoningPolicyBuildWorkflowResultAssetsRequest",eC3="GetAutomatedReasoningPolicyBuildWorkflowResultAssetsResponse",qb3="GetAutomatedReasoningPolicyBuildWorkflowResponse",Kb3="GuardrailAutomatedReasoningPolicyConfig",_b3="GetAutomatedReasoningPolicyNextScenario",zb3="GetAutomatedReasoningPolicyNextScenarioRequest",Yb3="GetAutomatedReasoningPolicyNextScenarioResponse",$b3="GetAutomatedReasoningPolicyRequest",Ob3="GetAutomatedReasoningPolicyResponse",Ab3="GetAutomatedReasoningPolicyTestCase",wb3="GetAutomatedReasoningPolicyTestCaseRequest",jb3="GetAutomatedReasoningPolicyTestCaseResponse",Hb3="GetAutomatedReasoningPolicyTestResult",Jb3="GetAutomatedReasoningPolicyTestResultRequest",Mb3="GetAutomatedReasoningPolicyTestResultResponse",Xb3="GetAutomatedReasoningPolicy",Pb3="GuardrailBlockedMessaging",Wb3="GenerationConfiguration",Db3="GuardrailContentFilter",fb3="GuardrailContentFilterAction",Zb3="GuardrailContentFilterConfig",Gb3="GuardrailContentFiltersConfig",vb3="GuardrailContentFiltersTier",Tb3="GuardrailContentFiltersTierConfig",kb3="GuardrailContentFiltersTierName",Vb3="GuardrailContentFilters",Nb3="GuardrailContextualGroundingAction",yb3="GuardrailContextualGroundingFilter",Eb3="GuardrailContextualGroundingFilterConfig",Lb3="GuardrailContextualGroundingFiltersConfig",hb3="GuardrailContextualGroundingFilters",Rb3="GuardrailContextualGroundingPolicy",Sb3="GuardrailContextualGroundingPolicyConfig",Cb3="GetCustomModel",bb3="GetCustomModelDeployment",xb3="GetCustomModelDeploymentRequest",Ib3="GetCustomModelDeploymentResponse",ub3="GetCustomModelRequest",mb3="GetCustomModelResponse",pb3="GuardrailContentPolicy",Bb3="GuardrailContentPolicyConfig",gb3="GuardrailCrossRegionConfig",Fb3="GuardrailCrossRegionDetails",Ub3="GuardrailConfiguration",Qb3="GuardrailDescription",db3="GetEvaluationJob",cb3="GetEvaluationJobRequest",lb3="GetEvaluationJobResponse",nb3="GetFoundationModel",ib3="GetFoundationModelAvailability",rb3="GetFoundationModelAvailabilityRequest",ob3="GetFoundationModelAvailabilityResponse",ab3="GetFoundationModelRequest",sb3="GetFoundationModelResponse",tb3="GuardrailFailureRecommendation",eb3="GuardrailFailureRecommendations",qx3="GetGuardrail",Kx3="GetGuardrailRequest",_x3="GetGuardrailResponse",zx3="GetImportedModel",Yx3="GetImportedModelRequest",$x3="GetImportedModelResponse",Ox3="GetInferenceProfile",Ax3="GetInferenceProfileRequest",wx3="GetInferenceProfileResponse",jx3="GuardrailModality",Hx3="GetModelCopyJob",Jx3="GetModelCopyJobRequest",Mx3="GetModelCopyJobResponse",Xx3="GetModelCustomizationJobRequest",Px3="GetModelCustomizationJobResponse",Wx3="GetModelCustomizationJob",Dx3="GetModelImportJob",fx3="GetModelImportJobRequest",Zx3="GetModelImportJobResponse",Gx3="GetModelInvocationJobRequest",vx3="GetModelInvocationJobResponse",Tx3="GetModelInvocationJob",kx3="GetModelInvocationLoggingConfiguration",Vx3="GetModelInvocationLoggingConfigurationRequest",Nx3="GetModelInvocationLoggingConfigurationResponse",yx3="GetMarketplaceModelEndpoint",Ex3="GetMarketplaceModelEndpointRequest",Lx3="GetMarketplaceModelEndpointResponse",hx3="GuardrailManagedWords",Rx3="GuardrailManagedWordsConfig",Sx3="GuardrailManagedWordLists",Cx3="GuardrailManagedWordListsConfig",bx3="GuardrailModalities",xx3="GuardrailName",Ix3="GuardrailPiiEntity",ux3="GuardrailPiiEntityConfig",mx3="GuardrailPiiEntitiesConfig",px3="GuardrailPiiEntities",Bx3="GetProvisionedModelThroughput",gx3="GetProvisionedModelThroughputRequest",Fx3="GetProvisionedModelThroughputResponse",Ux3="GetPromptRouter",Qx3="GetPromptRouterRequest",dx3="GetPromptRouterResponse",cx3="GuardrailRegex",lx3="GuardrailRegexConfig",nx3="GuardrailRegexesConfig",ix3="GuardrailRegexes",rx3="GuardrailSummary",ox3="GuardrailSensitiveInformationPolicy",ax3="GuardrailSensitiveInformationPolicyConfig",sx3="GuardrailStatusReason",tx3="GuardrailStatusReasons",ex3="GuardrailSummaries",qI3="GuardrailTopic",KI3="GuardrailTopicAction",_I3="GuardrailTopicConfig",zI3="GuardrailTopicsConfig",YI3="GuardrailTopicDefinition",$I3="GuardrailTopicExample",OI3="GuardrailTopicExamples",AI3="GuardrailTopicName",wI3="GuardrailTopicPolicy",jI3="GuardrailTopicPolicyConfig",HI3="GuardrailTopicsTier",JI3="GuardrailTopicsTierConfig",MI3="GuardrailTopicsTierName",XI3="GuardrailTopics",PI3="GetUseCaseForModelAccess",WI3="GetUseCaseForModelAccessRequest",DI3="GetUseCaseForModelAccessResponse",fI3="GuardrailWord",ZI3="GuardrailWordAction",GI3="GuardrailWordConfig",vI3="GuardrailWordsConfig",TI3="GuardrailWordPolicy",kI3="GuardrailWordPolicyConfig",VI3="GuardrailWords",NI3="HumanEvaluationConfig",yI3="HumanEvaluationCustomMetric",EI3="HumanEvaluationCustomMetrics",LI3="HumanTaskInstructions",hI3="HumanWorkflowConfig",RI3="Identifier",SI3="ImplicitFilterConfiguration",CI3="InvocationLogsConfig",bI3="InvocationLogSource",xI3="ImportedModelSummary",II3="ImportedModelSummaryList",uI3="InferenceProfileDescription",mI3="InferenceProfileModel",pI3="InferenceProfileModelSource",BI3="InferenceProfileModels",gI3="InferenceProfileSummary",FI3="InferenceProfileSummaries",UI3="InternalServerException",QI3="KnowledgeBaseConfig",dI3="KnowledgeBaseRetrieveAndGenerateConfiguration",cI3="KnowledgeBaseRetrievalConfiguration",lI3="KnowledgeBaseVectorSearchConfiguration",nI3="KbInferenceConfig",iI3="ListAutomatedReasoningPolicies",rI3="ListAutomatedReasoningPolicyBuildWorkflows",oI3="ListAutomatedReasoningPolicyBuildWorkflowsRequest",aI3="ListAutomatedReasoningPolicyBuildWorkflowsResponse",sI3="ListAutomatedReasoningPoliciesRequest",tI3="ListAutomatedReasoningPoliciesResponse",eI3="ListAutomatedReasoningPolicyTestCases",qu3="ListAutomatedReasoningPolicyTestCasesRequest",Ku3="ListAutomatedReasoningPolicyTestCasesResponse",_u3="ListAutomatedReasoningPolicyTestResults",zu3="ListAutomatedReasoningPolicyTestResultsRequest",Yu3="ListAutomatedReasoningPolicyTestResultsResponse",$u3="LoggingConfig",Ou3="ListCustomModels",Au3="ListCustomModelDeployments",wu3="ListCustomModelDeploymentsRequest",ju3="ListCustomModelDeploymentsResponse",Hu3="ListCustomModelsRequest",Ju3="ListCustomModelsResponse",Mu3="ListEvaluationJobs",Xu3="ListEvaluationJobsRequest",Pu3="ListEvaluationJobsResponse",Wu3="ListFoundationModels",Du3="ListFoundationModelAgreementOffers",fu3="ListFoundationModelAgreementOffersRequest",Zu3="ListFoundationModelAgreementOffersResponse",Gu3="ListFoundationModelsRequest",vu3="ListFoundationModelsResponse",Tu3="ListGuardrails",ku3="ListGuardrailsRequest",Vu3="ListGuardrailsResponse",Nu3="ListImportedModels",yu3="ListImportedModelsRequest",Eu3="ListImportedModelsResponse",Lu3="ListInferenceProfiles",hu3="ListInferenceProfilesRequest",Ru3="ListInferenceProfilesResponse",Su3="ListModelCopyJobs",Cu3="ListModelCopyJobsRequest",bu3="ListModelCopyJobsResponse",xu3="ListModelCustomizationJobsRequest",Iu3="ListModelCustomizationJobsResponse",uu3="ListModelCustomizationJobs",mu3="ListModelImportJobs",pu3="ListModelImportJobsRequest",Bu3="ListModelImportJobsResponse",gu3="ListModelInvocationJobsRequest",Fu3="ListModelInvocationJobsResponse",Uu3="ListModelInvocationJobs",Qu3="ListMarketplaceModelEndpoints",du3="ListMarketplaceModelEndpointsRequest",cu3="ListMarketplaceModelEndpointsResponse",lu3="ListProvisionedModelThroughputs",nu3="ListProvisionedModelThroughputsRequest",iu3="ListProvisionedModelThroughputsResponse",ru3="ListPromptRouters",ou3="ListPromptRoutersRequest",au3="ListPromptRoutersResponse",su3="LegalTerm",tu3="ListTagsForResource",eu3="ListTagsForResourceRequest",qm3="ListTagsForResourceResponse",Km3="Message",_m3="MetadataAttributeSchema",zm3="MetadataAttributeSchemaList",Ym3="MetadataConfigurationForReranking",$m3="ModelCopyJobSummary",Om3="ModelCustomizationJobSummary",Am3="ModelCopyJobSummaries",wm3="ModelCustomizationJobSummaries",jm3="ModelDataSource",Hm3="ModelInvocationJobInputDataConfig",Jm3="ModelInvocationJobOutputDataConfig",Mm3="ModelImportJobSummary",Xm3="ModelInvocationJobS3InputDataConfig",Pm3="ModelInvocationJobS3OutputDataConfig",Wm3="ModelInvocationJobSummary",Dm3="ModelImportJobSummaries",fm3="ModelInvocationJobSummaries",Zm3="MarketplaceModelEndpoint",Gm3="MarketplaceModelEndpointSummary",vm3="MarketplaceModelEndpointSummaries",Tm3="MetricName",km3="Offer",Vm3="OrchestrationConfiguration",Nm3="OutputDataConfig",ym3="Offers",Em3="PerformanceConfiguration",Lm3="PutModelInvocationLoggingConfiguration",hm3="PutModelInvocationLoggingConfigurationRequest",Rm3="PutModelInvocationLoggingConfigurationResponse",Sm3="ProvisionedModelSummary",Cm3="ProvisionedModelSummaries",bm3="PromptRouterDescription",xm3="PromptRouterSummary",Im3="PromptRouterSummaries",um3="PromptRouterTargetModel",mm3="PromptRouterTargetModels",pm3="PricingTerm",Bm3="PromptTemplate",gm3="PutUseCaseForModelAccess",Fm3="PutUseCaseForModelAccessRequest",Um3="PutUseCaseForModelAccessResponse",Qm3="QueryTransformationConfiguration",dm3="RetrieveAndGenerateConfiguration",cm3="RAGConfig",lm3="RetrieveConfig",nm3="RagConfigs",im3="RateCard",rm3="RoutingCriteria",om3="RetrievalFilter",am3="RetrievalFilterList",sm3="ResourceInUseException",tm3="RequestMetadataBaseFilters",em3="RequestMetadataFilters",qp3="RequestMetadataFiltersList",Kp3="RequestMetadataMap",_p3="RegisterMarketplaceModelEndpoint",zp3="RegisterMarketplaceModelEndpointRequest",Yp3="RegisterMarketplaceModelEndpointResponse",$p3="RerankingMetadataSelectiveModeConfiguration",Op3="ResourceNotFoundException",Ap3="RatingScale",wp3="RatingScaleItem",jp3="RatingScaleItemValue",Hp3="StartAutomatedReasoningPolicyBuildWorkflow",Jp3="StartAutomatedReasoningPolicyBuildWorkflowRequest",Mp3="StartAutomatedReasoningPolicyBuildWorkflowResponse",Xp3="StartAutomatedReasoningPolicyTestWorkflow",Pp3="StartAutomatedReasoningPolicyTestWorkflowRequest",Wp3="StartAutomatedReasoningPolicyTestWorkflowResponse",Dp3="S3Config",fp3="StatusDetails",Zp3="S3DataSource",Gp3="StopEvaluationJob",vp3="StopEvaluationJobRequest",Tp3="StopEvaluationJobResponse",kp3="StopModelCustomizationJob",Vp3="StopModelCustomizationJobRequest",Np3="StopModelCustomizationJobResponse",yp3="SageMakerEndpoint",Ep3="StopModelInvocationJob",Lp3="StopModelInvocationJobRequest",hp3="StopModelInvocationJobResponse",Rp3="S3ObjectDoc",Sp3="ServiceQuotaExceededException",Cp3="SupportTerm",bp3="ServiceUnavailableException",xp3="Tag",Ip3="TermDetails",up3="TrainingDataConfig",mp3="TrainingDetails",pp3="ThrottlingException",Bp3="TextInferenceConfig",gp3="TagList",Fp3="TrainingMetrics",Up3="TeacherModelConfig",Qp3="TooManyTagsException",dp3="TextPromptTemplate",cp3="TagResource",lp3="TagResourceRequest",np3="TagResourceResponse",ip3="UpdateAutomatedReasoningPolicy",rp3="UpdateAutomatedReasoningPolicyAnnotations",op3="UpdateAutomatedReasoningPolicyAnnotationsRequest",ap3="UpdateAutomatedReasoningPolicyAnnotationsResponse",sp3="UpdateAutomatedReasoningPolicyRequest",tp3="UpdateAutomatedReasoningPolicyResponse",ep3="UpdateAutomatedReasoningPolicyTestCase",qB3="UpdateAutomatedReasoningPolicyTestCaseRequest",KB3="UpdateAutomatedReasoningPolicyTestCaseResponse",_B3="UpdateGuardrail",zB3="UpdateGuardrailRequest",YB3="UpdateGuardrailResponse",$B3="UpdateMarketplaceModelEndpoint",OB3="UpdateMarketplaceModelEndpointRequest",AB3="UpdateMarketplaceModelEndpointResponse",wB3="UpdateProvisionedModelThroughput",jB3="UpdateProvisionedModelThroughputRequest",HB3="UpdateProvisionedModelThroughputResponse",JB3="UntagResource",MB3="UntagResourceRequest",XB3="UntagResourceResponse",PB3="Validator",WB3="VpcConfig",DB3="ValidationDetails",fB3="ValidationDataConfig",ZB3="ValidationException",GB3="ValidatorMetric",vB3="ValidationMetrics",TB3="VectorSearchBedrockRerankingConfiguration",kB3="VectorSearchBedrockRerankingModelConfiguration",VB3="VectorSearchRerankingConfiguration",NB3="ValidityTerm",yB3="Validators",EB3="annotation",LB3="agreementAvailability",J5q="andAll",hB3="agreementDuration",M5q="alternateExpression",RB3="acceptEula",Mj1="additionalModelRequestFields",X5q="addRule",SB3="addRuleFromNaturalLanguage",CB3="automatedReasoningPolicy",bB3="automatedReasoningPolicyBuildWorkflowSummaries",P5q="automatedReasoningPolicyConfig",xB3="automatedReasoningPolicySummaries",IB3="authorizationStatus",W5q="annotationSetHash",Xj1="applicationType",IKq="applicationTypeEquals",uB3="aggregatedTestFindingsResult",mB3="addTypeValue",D5q="addType",uKq="assetType",f5q="addVariable",mZ6="action",Pj1="annotations",pB3="arn",BB3="automated",gB3="byteContent",mKq="byCustomizationType",Z5q="bedrockEvaluatorModels",Wj1="blockedInputMessaging",pKq="byInferenceType",FB3="bedrockKnowledgeBaseIdentifiers",UB3="buildLog",QB3="bedrockModel",fJ8="baseModelArn",BKq="baseModelArnEquals",dB3="baseModelIdentifier",cB3="bedrockModelIdentifiers",lB3="baseModelName",nB3="bucketName",Dj1="blockedOutputsMessaging",gKq="byOutputModality",FKq="byProvider",iB3="bedrockRerankingConfiguration",rB3="buildSteps",oB3="buildWorkflowAssets",yG="buildWorkflowId",fj1="buildWorkflowType",W86="client",KD="createdAt",UKq="createdAfter",QKq="createdBefore",Zj1="customizationConfig",Gj1="commitmentDuration",G5q="customerEncryptionKeyId",v5q="commitmentExpirationTime",aB3="copyFrom",sB3="claimsFalseScenario",tB3="contextualGroundingPolicy",T5q="contextualGroundingPolicyConfig",k5q="customMetrics",eB3="customModelArn",qg3="customMetricConfig",Kg3="customMetricDefinition",vj1="customModelDeploymentArn",V5q="customModelDeploymentIdentifier",_g3="customModelDeploymentName",zg3="customMetricsEvaluatorModelIdentifiers",Yg3="customModelKmsKeyId",N5q="customModelName",$g3="customModelTags",Og3="customModelUnits",Ag3="customModelUnitsPerModelCopy",wg3="customModelUnitsVersion",jg3="contentPolicy",y5q="contentPolicyConfig",E5q="contradictingRules",L5q="crossRegionConfig",h5q="crossRegionDetails",fH="clientRequestToken",Hg3="conflictingRules",R5q="customizationsSupported",MU6="confidenceThreshold",UV="creationTimeAfter",QV="creationTimeBefore",S5q="claimsTrueScenario",Jg3="contentType",qZ="creationTime",XU6="customizationType",Mg3="cloudWatchConfig",C5q="claims",Xg3="confidence",Pg3="code",Wg3="context",Dg3="content",j$="description",fg3="distillationConfig",b5q="documentContentType",x5q="documentDescription",ZJ8="definitionHash",Zg3="datasetLocation",I5q="desiredModelArn",u5q="datasetMetricConfigs",Gg3="desiredModelId",m5q="desiredModelUnits",p5q="documentName",vg3="dataProcessingDetails",Tg3="desiredProvisionedModelName",B5q="deleteRule",kg3="disjointRuleSets",Vg3="differenceScenarios",g5q="deleteType",Ng3="deleteTypeValue",F5q="deleteVariable",yg3="data",Eg3="dataset",Tj1="definition",Lg3="dimension",hg3="document",Rg3="documents",Ug="error",pZ6="endpointArn",GJ8="expectedAggregatedFindingsResult",Sg3="entitlementAvailability",U5q="evaluationConfig",kj1="endpointConfig",Cg3="embeddingDataDeliveryEnabled",bg3="endpointIdentifier",xg3="evaluationJobs",Ig3="errorMessage",Q5q="evaluatorModelConfig",ug3="evaluatorModelIdentifiers",mg3="endpointName",pg3="expectedResult",Bg3="executionRole",gg3="endpointStatus",Fg3="externalSourcesConfiguration",Ug3="endpointStatusMessage",BZ6="endTime",Qg3="evaluationTaskTypes",dg3="entries",d5q="enabled",Vj1="equals",cg3="errors",vJ8="expression",c5q="examples",l5q="feedback",n5q="filtersConfig",i5q="formData",lg3="flowDefinitionArn",Nj1="fallbackModel",r5q="foundationModelArn",dKq="foundationModelArnEquals",D86="failureMessage",ng3="failureMessages",ig3="fieldName",rg3="failureRecommendations",og3="fieldsToExclude",ag3="fieldsToInclude",sg3="floatValue",o5q="filters",tg3="filter",cKq="force",eg3="guardrails",yj1="guardrailArn",TJ8="guardContent",a5q="generationConfiguration",s5q="guardrailConfiguration",PU6="guardrailId",IZ6="guardrailIdentifier",qF3="guardrailProfileArn",KF3="guardrailProfileIdentifier",_F3="guardrailProfileId",zF3="greaterThan",t5q="generatedTestCases",YF3="greaterThanOrEquals",HU6="guardrailVersion",$F3="human",Qg="httpError",OF3="httpHeader",Ej1="hyperParameters",S7="httpQuery",AF3="humanWorkflowConfig",Mq="http",kJ8="id",PS="inputAction",e5q="inferenceConfig",wF3="inferenceConfigSummary",jF3="ingestContent",Lj1="inputDataConfig",HF3="imageDataDeliveryEnabled",WS="inputEnabled",JF3="implicitFilterConfiguration",MF3="initialInstanceCount",XF3="invocationJobSummaries",PF3="invocationLogsConfig",WF3="invocationLogSource",VJ8="inputModalities",q3q="importedModelArn",DF3="importedModelKmsKeyArn",fF3="importedModelKmsKeyId",hj1="importedModelName",ZF3="importedModelTags",lKq="isOwned",GF3="inferenceParams",Rj1="inferenceProfileArn",K3q="inferenceProfileIdentifier",_3q="inferenceProfileId",Sj1="inferenceProfileName",vF3="inferenceProfileSummaries",z3q="instructSupported",TF3="inferenceSourceIdentifier",Y3q="inputStrength",kF3="instanceType",$3q="inferenceTypesSupported",VF3="idempotencyToken",NF3="identifier",yF3="impossible",O3q="instructions",EF3="in",LF3="invalid",_D="jobArn",A3q="jobDescription",w3q="jobExpirationTime",on="jobIdentifier",hF3="jobIdentifiers",cV="jobName",RF3="jobStatus",SF3="jobSummaries",Cj1="jobTags",j3q="jobType",bj1="key",CF3="knowledgeBaseConfiguration",bF3="knowledgeBaseConfig",H3q="knowledgeBaseId",xF3="knowledgeBaseRetrievalConfiguration",IF3="kmsEncryptionKey",J3q="kbInferenceConfig",M3q="kmsKeyArn",xj1="kmsKeyId",uF3="keyPrefix",mF3="logic",X3q="loggingConfig",pF3="listContains",BF3="largeDataDeliveryS3Config",gF3="logGroupName",DS="lastModifiedTime",FF3="legalTerm",UF3="lessThanOrEquals",QF3="lessThan",WU6="lastUpdatedAt",dF3="lastUpdatedAnnotationSetHash",cF3="lastUpdatedDefinitionHash",NJ8="logicWarning",lF3="latency",lV="message",zD="modelArn",HJ8="modelArnEquals",nF3="metadataAttributes",P3q="modelArchitecture",iF3="modelConfiguration",rF3="modelCopyJobSummaries",oF3="modelCustomizationJobSummaries",aF3="modelConfigSummary",sF3="metadataConfiguration",tF3="modelDetails",W3q="modelDeploymentName",Ij1="modelDataSource",eF3="modelDeploymentSummaries",f86="modelIdentifier",qU3="modelImportJobSummaries",YL="modelId",KU3="modelIdentifiers",uj1="modelKmsKeyArn",_U3="modelKmsKeyId",D3q="modelLifecycle",yJ8="marketplaceModelEndpoint",zU3="marketplaceModelEndpoints",OY6="modelName",YU3="metricNames",dY="maxResults",$U3="maxResponseLengthForInference",OU3="modelSource",AU3="modelSourceConfig",wU3="modelSourceEquals",DU6="modelSourceIdentifier",JJ8="modelStatus",mj1="modelSummaries",jU3="messageType",HU3="maxTokens",JU3="modelTags",pj1="modelUnits",MU3="managedWordLists",XU3="managedWordListsConfig",PU3="messages",gZ6="models",WU3="mutation",MA="name",NG="nameContains",Bj1="notEquals",DU3="notIn",f3q="naturalLanguage",Z3q="newName",fU3="numberOfResults",ZU3="numberOfRerankedResults",o5="nextToken",GU3="noTranslations",vU3="newValue",TU3="options",fS="outputAction",kU3="ownerAccountId",G3q="orAll",VU3="orchestrationConfiguration",Z86="outputDataConfig",ZS="outputEnabled",NU3="offerId",EJ8="outputModalities",yU3="outputModelArn",EU3="outputModelKmsKeyArn",LU3="outputModelName",hU3="outputModelNameContains",v3q="outputStrength",RU3="overrideSearchType",T3q="offerToken",nKq="offerType",SU3="offers",k3q="premises",X_="policyArn",CU3="performanceConfig",fU6="policyDefinition",bU3="policyDefinitionRule",xU3="policyDefinitionType",IU3="policyDefinitionVariable",uU3="priorElement",mU3="piiEntitiesConfig",pU3="piiEntities",V3q="policyId",BU3="precomputedInferenceSource",gU3="precomputedInferenceSourceIdentifiers",gj1="provisionedModelArn",Fj1="provisionedModelId",Uj1="provisionedModelName",FU3="provisionedModelSummaries",N3q="providerName",ZU6="promptRouterArn",UU3="policyRepairAssets",Qj1="promptRouterName",QU3="promptRouterSummaries",dU3="precomputedRagSourceConfig",cU3="precomputedRagSourceIdentifiers",y3q="promptTemplate",lU3="policyVersionArn",E3q="pattern",nU3="planning",L3q="policies",iU3="price",LJ8="queryContent",rU3="qualityReport",oU3="queryTransformationConfiguration",h3q="rule",Ku="roleArn",aU3="retrieveAndGenerateConfig",sU3="retrieveAndGenerateSourceConfig",dj1="resourceARN",tU3="regionAvailability",eU3="ruleCount",qQ3="ragConfigSummary",KQ3="rateCard",_Q3="ragConfigs",zQ3="regexesConfig",YQ3="rerankingConfiguration",$Q3="retrievalConfiguration",OQ3="retrieveConfig",cj1="routingCriteria",R3q="ruleId",AQ3="ragIdentifiers",lj1="ruleIds",wQ3="ratingMethod",jQ3="requestMetadataFilters",HQ3="resourceName",JQ3="refundPolicyDescription",MQ3="responseQualityDifference",XQ3="ratingScale",PQ3="retrieveSourceConfig",S3q="ragSourceIdentifier",C3q="responseStreamingSupported",WQ3="regexes",b3q="rules",qO="status",iKq="sourceAccountEquals",x3q="sourceAccountId",eW="sortBy",I3q="s3BucketOwner",DQ3="s3Config",fQ3="sourceContent",ZQ3="stringContains",u3q="statusDetails",GQ3="s3DataSource",vQ3="scenarioExpression",TQ3="s3EncryptionKeyId",dV="statusEquals",kQ3="securityGroupIds",VQ3="subnetIds",NQ3="s3InputDataConfig",yQ3="s3InputFormat",EQ3="sensitiveInformationPolicy",m3q="sensitiveInformationPolicyConfig",LQ3="s3Location",p3q="statusMessage",nj1="sourceModelArn",rKq="sourceModelArnEquals",hQ3="selectiveModeConfiguration",B3q="sourceModelName",RQ3="sageMaker",SQ3="selectionMode",qD="sortOrder",CQ3="s3OutputDataConfig",bQ3="supportingRules",xQ3="statusReasons",IQ3="stopSequences",uQ3="sourceType",oKq="submitTimeAfter",aKq="submitTimeBefore",g3q="submitTime",mQ3="supportTerm",an="s3Uri",pQ3="stringValue",BQ3="startsWith",gQ3="satisfiable",FQ3="scenario",F3q="server",U3q="smithy.ts.sdk.synthetic.com.amazonaws.bedrock",UQ3="sources",QQ3="statements",hJ8="translation",dQ3="translationAmbiguous",cQ3="typeCount",AY6="testCaseId",lQ3="testCaseIds",Q3q="testCase",nQ3="testCases",d3q="tierConfig",iQ3="topicsConfig",rQ3="tooComplex",oQ3="termDetails",ij1="trainingDataConfig",aQ3="textDataDeliveryEnabled",rj1="timeoutDurationInHours",sQ3="trainingDetails",tQ3="typeEquals",eQ3="testFindings",qd3="textInferenceConfig",Kd3="tagKeys",_d3="trainingLoss",c3q="trainingMetrics",l3q="targetModelArn",zd3="teacherModelConfig",Yd3="teacherModelIdentifier",n3q="targetModelKmsKeyArn",oj1="targetModelName",$d3="targetModelNameContains",aj1="targetModelTags",Od3="typeName",RJ8="tierName",Ad3="topicPolicy",i3q="topicPolicyConfig",wd3="textPromptTemplate",jd3="topP",Hd3="testResult",Jd3="testRunResult",Md3="testRunStatus",Xd3="testResults",Pd3="taskType",_u="tags",sj1="text",Wd3="temperature",r3q="threshold",o3q="tier",Dd3="topics",fd3="translations",sw="type",Zd3="types",Gd3="unit",$M="updatedAt",vd3="usageBasedPricingTerm",Td3="untranslatedClaims",kd3="updateFromRulesFeedback",Vd3="updateFromScenarioFeedback",Nd3="untranslatedPremises",yd3="usePromptResponse",a3q="updateRule",Ed3="unusedTypes",Ld3="unusedTypeValues",hd3="updateTypeValue",s3q="updateType",Rd3="unusedVariables",t3q="updateVariable",Sd3="url",Cd3="uri",tj1="values",bd3="variableCount",wY6="vpcConfig",xd3="validationDetails",ej1="validationDataConfig",Id3="videoDataDeliveryEnabled",ud3="validationLoss",e3q="validationMetrics",md3="valueName",pd3="vectorSearchConfiguration",Bd3="validityTerm",jY6="value",gd3="validators",Fd3="valid",q9q="variable",K9q="variables",dg="version",Ud3="vpc",Qd3="words",dd3="workflowContent",cd3="wordsConfig",ld3="wordPolicy",_9q="wordPolicyConfig",nd3="x-amz-client-token",y6="com.amazonaws.bedrock",id3=[0,y6,qL3,8,0],z9q=[0,y6,_L3,8,0],Y9q=[0,y6,YL3,8,0],rd3=[0,y6,$L3,8,0],od3=[0,y6,HL3,8,0],ad3=[0,y6,DL3,8,21],$9q=[0,y6,fL3,8,0],O9q=[0,y6,ZL3,8,0],sd3=[0,y6,gL3,8,0],qH1=[0,y6,FL3,8,0],KH1=[0,y6,iL3,8,0],Fg=[0,y6,aL3,8,0],_H1=[0,y6,eL3,8,0],zH1=[0,y6,Oh3,8,0],$Y6=[0,y6,jh3,8,0],FZ6=[0,y6,IL3,8,0],G86=[0,y6,fh3,8,0],td3=[0,y6,vh3,8,0],A9q=[0,y6,Th3,8,0],SJ8=[0,y6,Eh3,8,0],CJ8=[0,y6,hh3,8,0],ed3=[0,y6,Qh3,8,21],qc3=[0,y6,MC3,8,0],w9q=[0,y6,WC3,8,0],GU6=[0,y6,DC3,8,0],Kc3=[0,y6,kC3,8,0],j9q=[0,y6,NC3,8,0],_c3=[0,y6,VC3,8,0],uZ6=[0,y6,Pb3,8,0],MJ8=[0,y6,fb3,8,0],H9q=[0,y6,kb3,8,0],J9q=[0,y6,Nb3,8,0],vU6=[0,y6,Qb3,8,0],zc3=[0,y6,tb3,8,0],Yc3=[0,y6,jx3,8,0],bJ8=[0,y6,xx3,8,0],$c3=[0,y6,sx3,8,0],XJ8=[0,y6,KI3,8,0],M9q=[0,y6,YI3,8,0],Oc3=[0,y6,$I3,8,0],X9q=[0,y6,AI3,8,0],P9q=[0,y6,MI3,8,0],P86=[0,y6,ZI3,8,0],Ac3=[0,y6,LI3,8,0],wc3=[0,y6,RI3,8,0],YH1=[0,y6,uI3,8,0],W9q=[0,y6,Km3,8,0],jc3=[0,y6,Tm3,8,0],$H1=[0,y6,bm3,8,0],Hc3=[0,y6,dp3,8,0],Jc3=[-3,y6,hE3,{[Ug]:W86,[Qg]:403},[lV],[0]];

$F={inputTokens:3,outputTokens:15,promptCacheWriteTokens:3.75,promptCacheReadTokens:0.3,webSearchRequests:0.01},mHq={inputTokens:15,outputTokens:75,promptCacheWriteTokens:18.75,promptCacheReadTokens:1.5,webSearchRequests:0.01},RX8={inputTokens:5,outputTokens:25,promptCacheWriteTokens:6.25,promptCacheReadTokens:0.5,webSearchRequests:0.01},ay9={inputTokens:30,outputTokens:150,promptCacheWriteTokens:37.5,promptCacheReadTokens:3,webSearchRequests:0.01},aW1={inputTokens:0.8,outputTokens:4,promptCacheWriteTokens:1,promptCacheReadTokens:0.08,webSearchRequests:0.01},sW1={inputTokens:1,outputTokens:5,promptCacheWriteTokens:1.25,promptCacheReadTokens:0.1,webSearchRequests:0.01},sy9=RX8;SX8={[ET(RX1.firstParty)]:aW1,[ET(SX1.firstParty)]:sW1,[ET(hX1.firstParty)]:$F,[ET(LX1.firstParty)]:$F,[ET(CX1.firstParty)]:$F,[ET(bX1.firstParty)]:$F,[ET(mX1.firstParty)]:$F,[ET(xX1.firstParty)]:mHq,[ET(IX1.firstParty)]:mHq,[ET(uX1.firstParty)]:RX8,[ET(oZ6.firstParty)]:RX8}});function $i(q){return EY6.includes(q)}function X06(q){return KE9.includes(q)}var EY6,KE9;var P06=L(()=>{EY6=["sonnet","opus","haiku","best","sonnet[1m]","opus[1m]","opusplan"];KE9=["sonnet","opus","haiku"]});function _E9(q,K){if(q.includes(K))return!0;if($i(q))return Y5(q).toLowerCase().includes(K);return!1}function BHq(q,K){if(!q.startsWith(K))return!1;return q.length===K.length||q[K.length]==="-"}function zE9(q,K){let _=$i(q)?Y5(q).toLowerCase():q;if(BHq(_,K))return!0;if(!K.startsWith("claude-")&&BHq(_,`claude-${K}`))return!0;return!1}function gHq(q,K){for(let _ of K){if(X06(_))continue;let z=_.indexOf(q);if(z===-1)continue;let Y=z+q.length;if(Y===_.length||_[Y]==="-")return!0}return!1}function I86(q){let K=k7()||{},{availableModels:_}=K;if(!_)return!0;if(_.length===0)return!1;let Y=cM8(q).trim().toLowerCase(),$=_.map((O)=>O.trim().toLowerCase());if($.includes(Y)){if(!X06(Y)||!gHq(Y,$))return!0}for(let O of $)if(X06(O)&&!gHq(O,$)&&_E9(Y,O))return!0;if($i(Y)){let O=Y5(Y).toLowerCase();

if(f!==null&&f!==void 0){IHq(f),Y.fastMode=!1;continue}let G=Bo_(J);if(G!==null&&G<mo_){await C7(G,_.signal,{abortError:Nm1});continue}let Z=Math.max(G??uo_,po_),v=fw6(J)?"overloaded":"rate_limit";if(SHq(Date.now()+Z,v),gK())Y.fastMode=!1;continue}if(H&&Ro_(J)){CHq(),Y.fastMode=!1;continue}if(fw6(J)&&!Eo_(_.querySource))throw d("tengu_api_529_background_dropped",{query_source:_.querySource}),new Pm(J,Y);if(fw6(J)&&(process.env.FALLBACK_FOR_ALL_PRIMARY_MODELS||!i7()&&LY6(_.model))){if(O++,O>=Vo_){if(_.fallbackModel)throw d("tengu_api_opus_fallback_triggered",{original_model:_.model,fallback_model:_.fallbackModel,provider:L86()}),new Zw6(_.model,_.fallbackModel);if(!process.env.IS_SANDBOX&&!Oy8())throw d("tengu_api_custom_529_overloaded_error",{}),new Pm(Error(Rm1),Y)}}let M=Oy8()&&wP4(J);if(j>z&&!M)throw new Pm(J,Y);if(!(So_(J)||bo_(J))&&(!(J instanceof nq)||!xo_(J)))throw new Pm(J,Y);if(J instanceof nq){let f=HP4(J);if(f){let{inputTokens:G,contextLimit:Z}=f,v=1000,k=Math.max(0,Z-G-1000);if(k<ym1)throw j6(Error(`availableContext ${k} is less than FLOOR_OUTPUT_TOKENS ${ym1}`)),J;let V=(Y.thinkingConfig.type==="enabled"?Y.thinkingConfig.budgetTokens:0)+1,y=Math.max(ym1,k,V);Y.maxTokensOverride=y,d("tengu_max_tokens_context_overflow_adjustment",{inputTokens:G,contextLimit:Z,adjustedMaxTokens:y,attempt:j});continue}}let P=jP4(J),W;if(M&&J instanceof nq&&J.status===429)w++,W=go_(J)??Math.min(qb(w,P,AP4),Em1);else if(M)w++,W=Math.min(qb(w,P,AP4),Em1);else W=qb(j,P);let D=M?w:j;if(d("tengu_api_retry",{attempt:D,delayMs:W,error:J.message,status:J.status,provider:L86()}),M){if(W>60000)d("tengu_api_persistent_retry_wait",{status:J.status,delayMs:W,attempt:D,provider:L86()});let f=W;while(f>0){if(_.signal?.aborted)throw new c_;if(J instanceof nq)yield hm1(J,f,D,z);let G=Math.min(f,Lo_);await C7(G,_.signal,{abortError:Nm1}),f-=G}if(j>=z)j=z}else{if(J instanceof nq)yield hm1(J,W,j,z);

await C7(W,_.signal,{abortError:Nm1})}}}throw new Pm(A,Y)}function jP4(q){return(q.headers?.["retry-after"]||q.headers?.get?.("retry-after"))??null}function qb(q,K,_=32000){if(K){let $=parseInt(K,10);if(!isNaN($))return $*1000}let z=Math.min(No_*Math.pow(2,q-1),_),Y=Math.random()*0.25*z;return z+Y}function HP4(q){if(q.status!==400||!q.message)return;if(!q.message.includes("input length and `max_tokens` exceed context limit"))return;let K=/input length and `max_tokens` exceed context limit: (\d+) \+ (\d+) > (\d+)/,_=q.message.match(K);if(!_||_.length!==4)return;if(!_[1]||!_[2]||!_[3]){j6(Error("Unable to parse max_tokens from max_tokens exceed context limit error message"));return}let z=parseInt(_[1],10),Y=parseInt(_[2],10),$=parseInt(_[3],10);if(isNaN(z)||isNaN(Y)||isNaN($))return;return{inputTokens:z,maxTokens:Y,contextLimit:$}}function Ro_(q){if(!(q instanceof nq))return!1;return q.status===400&&(q.message?.includes("Fast mode is not enabled")??!1)}function fw6(q){if(!(q instanceof nq))return!1;return q.status===529||(q.message?.includes('"type":"overloaded_error"')??!1)}function Lm1(q){return q instanceof nq&&q.status===403&&(q.message?.includes("OAuth token has been revoked")??!1)}function JP4(q){if(c6(process.env.CLAUDE_CODE_USE_BEDROCK)){if(ZHq(q)||q instanceof nq&&q.status===403)return!0}return!1}function So_(q){if(JP4(q))return Wl6(),!0;return!1}function Co_(q){if(!(q instanceof Error))return!1;let K=q.message;return K.includes("Could not load the default credentials")||K.includes("Could not refresh access token")||K.includes("invalid_grant")}function MP4(q){if(c6(process.env.CLAUDE_CODE_USE_VERTEX)){if(Co_(q))return!0;if(q instanceof nq&&q.status===401)return!0}return!1}function bo_(q){if(MP4(q))return Dl6(),!0;return!1}function xo_(q){if(OP4(q))return!1;if(Oy8()&&wP4(q))return!0;if(c6(process.env.CLAUDE_CODE_REMOTE)&&(q.status===401||q.status===403))return!0;if(q.message?.includes('"type":"overloaded_error"'))return!0;if(HP4(q))return!0;let K=q.headers?.get("x-should-retry");

if("waitForReady"in q){if(typeof q.waitForReady!=="boolean")throw Error("Invalid method config: invalid waitForReady");_.waitForReady=q.waitForReady}if("timeout"in q)if(typeof q.timeout==="object"){if(!("seconds"in q.timeout)||typeof q.timeout.seconds!=="number")throw Error("Invalid method config: invalid timeout.seconds");if(!("nanos"in q.timeout)||typeof q.timeout.nanos!=="number")throw Error("Invalid method config: invalid timeout.nanos");_.timeout=q.timeout}else if(typeof q.timeout==="string"&&HC8.test(q.timeout)){let z=q.timeout.substring(0,q.timeout.length-1).split(".");_.timeout={seconds:z[0]|0,nanos:((K=z[1])!==null&&K!==void 0?K:0)|0}}else throw Error("Invalid method config: invalid timeout");if("maxRequestBytes"in q){if(typeof q.maxRequestBytes!=="number")throw Error("Invalid method config: invalid maxRequestBytes");_.maxRequestBytes=q.maxRequestBytes}if("maxResponseBytes"in q){if(typeof q.maxResponseBytes!=="number")throw Error("Invalid method config: invalid maxRequestBytes");_.maxResponseBytes=q.maxResponseBytes}if("retryPolicy"in q)if("hedgingPolicy"in q)throw Error("Invalid method config: retryPolicy and hedgingPolicy cannot both be specified");else _.retryPolicy=fyz(q.retryPolicy);else if("hedgingPolicy"in q)_.hedgingPolicy=Zyz(q.hedgingPolicy);return _}function ln4(q){if(!("maxTokens"in q)||typeof q.maxTokens!=="number"||q.maxTokens<=0||q.maxTokens>1000)throw Error("Invalid retryThrottling: maxTokens must be a number in (0, 1000]");if(!("tokenRatio"in q)||typeof q.tokenRatio!=="number"||q.tokenRatio<=0)throw Error("Invalid retryThrottling: tokenRatio must be a number greater than 0");return{maxTokens:+q.maxTokens.toFixed(3),tokenRatio:+q.tokenRatio.toFixed(3)}}function vyz(q){if(!(typeof q==="object"&&q!==null))throw Error(`Invalid loadBalancingConfig: unexpected type ${typeof q}`);let K=Object.keys(q);if(K.length>1)throw Error(`Invalid loadBalancingConfig: unexpected multiple keys ${K}`);if(K.length===0)throw Error("Invalid loadBalancingConfig: load balancing policy name required");

return(K=(q=this.child)===null||q===void 0?void 0:q.getPeer())!==null&&K!==void 0?K:this.channel.getTarget()}start(q,K){this.trace("start called"),this.metadata=q.clone(),this.listener=K,this.getConfig()}sendMessageWithContext(q,K){if(this.trace("write() called with message of length "+K.length),this.child)this.sendMessageOnChild(q,K);else this.pendingMessage={context:q,message:K}}startRead(){if(this.trace("startRead called"),this.child)this.child.startRead();else this.readPending=!0}halfClose(){if(this.trace("halfClose called"),this.child&&!this.writeFilterPending)this.child.halfClose();else this.pendingHalfClose=!0}setCredentials(q){this.credentials=q}addStatusWatcher(q){this.statusWatchers.push(q)}getCallNumber(){return this.callNumber}getAuthContext(){if(this.child)return this.child.getAuthContext();else return null}}Vs4.ResolvingCall=ks4});var Cs4=B((Rs4)=>{Object.defineProperty(Rs4,"__esModule",{value:!0});Rs4.RetryingCall=Rs4.MessageBufferTracker=Rs4.RetryThrottler=void 0;var zb8=e_(),zbz=LE6(),Ybz=nD(),$bz=Cw(),Obz="retrying_call";class Es4{constructor(q,K,_){if(this.maxTokens=q,this.tokenRatio=K,_)this.tokens=_.tokens*(q/_.maxTokens);else this.tokens=q}addCallSucceeded(){this.tokens=Math.min(this.tokens+this.tokenRatio,this.maxTokens)}addCallFailed(){this.tokens=Math.max(this.tokens-1,0)}canRetryCall(){return this.tokens>this.maxTokens/2}}Rs4.RetryThrottler=Es4;class Ls4{constructor(q,K){this.totalLimit=q,this.limitPerCall=K,this.totalAllocated=0,this.allocatedPerCall=new Map}allocate(q,K){var _;let z=(_=this.allocatedPerCall.get(K))!==null&&_!==void 0?_:0;if(this.limitPerCall-z<q||this.totalLimit-this.totalAllocated<q)return!1;return this.allocatedPerCall.set(K,z+q),this.totalAllocated+=q,!0}free(q,K){var _;if(this.totalAllocated<q)throw Error(`Invalid buffer allocation state: call ${K} freed ${q} > total allocated ${this.totalAllocated}`);this.totalAllocated-=q;let z=(_=this.allocatedPerCall.get(K))!==null&&_!==void 0?_:0;

this.updateState(P)},requestReresolution:()=>{throw Error("Resolving load balancer should never call requestReresolution")},addChannelzChild:(P)=>{if(this.channelzEnabled)this.channelzInfoTracker.childrenTracker.refChild(P)},removeChannelzChild:(P)=>{if(this.channelzEnabled)this.channelzInfoTracker.childrenTracker.unrefChild(P)}};this.resolvingLoadBalancer=new Jbz.ResolvingLoadBalancer(this.target,M,this.options,(P,W)=>{var D;if(P.retryThrottling)Ob8.set(this.getTarget(),new fn1.RetryThrottler(P.retryThrottling.maxTokens,P.retryThrottling.tokenRatio,Ob8.get(this.getTarget())));else Ob8.delete(this.getTarget());if(this.channelzEnabled)this.channelzInfoTracker.trace.addTrace("CT_INFO","Address resolution succeeded");(D=this.configSelector)===null||D===void 0||D.unref(),this.configSelector=W,this.currentResolutionError=null,process.nextTick(()=>{let f=this.configSelectionQueue;if(this.configSelectionQueue=[],f.length>0)this.callRefTimerUnref();for(let G of f)G.getConfig()})},(P)=>{if(this.channelzEnabled)this.channelzInfoTracker.trace.addTrace("CT_WARNING","Address resolution failed with code "+P.code+' and details "'+P.details+'"');if(this.configSelectionQueue.length>0)this.trace("Name resolution failed with calls queued for config selection");if(this.configSelector===null)this.currentResolutionError=Object.assign(Object.assign({},(0,vbz.restrictControlPlaneStatusCode)(P.code,P.details)),{metadata:P.metadata});let W=this.configSelectionQueue;if(this.configSelectionQueue=[],W.length>0)this.callRefTimerUnref();for(let D of W)D.reportResolverError(P)}),this.filterStackFactory=new Pbz.FilterStackFactory([new Wbz.CompressionFilterFactory(this,this.options)]),this.trace("Channel constructed with options "+JSON.stringify(_,void 0,2));let X=Error();if((0,Yb8.isTracerEnabled)("channel_stacktrace"))(0,Yb8.trace)(MK6.LogVerbosity.DEBUG,"channel_stacktrace","("+this.channelzRef.id+`) Channel constructed 
`+((w=X.stack)===null||w===void 0?void 0:w.substring(X.stack.indexOf(`
`)+1)));

function B4K({mode:q,reducedMotion:K,hasActiveTools:_,responseLengthRef:z,message:Y,messageColor:$,shimmerColor:O,overrideColor:A,loadingStartTimeRef:w,totalPausedMsRef:j,pauseStartTimeRef:H,spinnerSuffix:J,verbose:M,columns:X,hasRunningTeammates:P,teammateTokens:W,foregroundedTeammate:D,leaderIsIdle:f=!1,thinkingStatus:G,effortSuffix:Z}){let[v,k]=pO(K?null:50),V=Date.now(),y=H.current!==null?H.current-w.current-j.current:V-w.current-j.current,E=V-y,R=x68.useRef(E);if(!P||E<R.current)R.current=E;let b=z.current,{isStalled:I,stalledIntensity:m}=cr1(k,b,_||f,K),p=K?0:Math.floor(k/120),C=q==="requesting"?50:200,g=x68.useMemo(()=>J1(Y),[Y]),F=g+20,U=Math.floor(k/C),c=K?-100:I?-100:q==="requesting"?U%F-10:g+10-U%F,K6=K?0:q==="tool-use"?(Math.sin(k/1000*Math.PI)+1)/2:0,o=x68.useRef(b);if(K)o.current=b;else{let f8=b-o.current;if(f8>0){let k6;if(f8<70)k6=3;else if(f8<200)k6=Math.max(8,Math.ceil(f8*0.15));else k6=50;o.current=Math.min(o.current+k6,b)}}let q6=o.current,t=Math.round(q6/4),n=P?Math.max(y,V-R.current):y,z6=I5(n),M6=J1(z6),J6=D&&!D.isIdle?D.progress?.tokenCount??0:t+W,G6=pK(J6),H6=P?`${G6} tokens`:`${o6.arrowDown} ${G6} tokens`,e=J1(H6),a=G==="thinking"?`thinking${Z}`:typeof G==="number"?`thought for ${Math.max(1,Math.round(G/1000))}s`:null,_6=a?J1(a):0,l=g+2,i=nFz,A6=G!==null,O6=M||P||n>iFz,X6=X-l-5,v6=A6&&X6>_6;

if(!Z){for(let v6 of Object.values(W))if(gH(v6)&&v6.status==="running"){if(v6.progress?.tokenCount)n+=v6.progress.tokenCount}}let z6=z.current!==null?z.current-K.current-_.current:Date.now()-K.current-_.current,M6=Math.round($.current/4),J6="claude",G6="claudeShimmer",H6=O??J6,e=A??G6,a=null;if(M&&q6&&!V)return fq.createElement(u,{flexDirection:"column",width:"100%",alignItems:"flex-start"},fq.createElement(u,{flexDirection:"row",flexWrap:"wrap",marginTop:1,width:"100%"},fq.createElement(T,{dimColor:!0},QE," Idle",!t&&" · teammates running")),Z&&fq.createElement(yI8,{selectedIndex:v,isInSelectionMode:k==="selecting-agent",allIdle:t,leaderTokenCount:M6,leaderIdleText:"Idle"}));if(V?.isIdle){let v6=t?`${QE} Worked for ${I5(Date.now()-V.startTime)}`:`${QE} Idle`;return fq.createElement(u,{flexDirection:"column",width:"100%",alignItems:"flex-start"},fq.createElement(u,{flexDirection:"row",flexWrap:"wrap",marginTop:1,width:"100%"},fq.createElement(T,{dimColor:!0},v6)),Z&&q6&&fq.createElement(yI8,{selectedIndex:v,isInSelectionMode:k==="selecting-agent",allIdle:t,leaderVerb:M?void 0:g,leaderIdleText:M?"Idle":void 0,leaderTokenCount:M6}))}let _6=!1,l=X.spinnerTipsEnabled!==!1,i=l&&z6>1800000,A6=l&&z6>30000&&!w8().btwUseCount,O6=_6?void 0:i&&!p?"Use /clear to start fresh when switching topics and free up context":A6&&!p?"Use /btw to ask a quick side question without interrupting Claude's current work":Y,X6=null;

return(await Promise.all(Y.map((v)=>uh6(_s(v,".claude","skills"),"projectSettings")))).flat().map((v)=>v.skill)}let[A,w,j,H,J]=await Promise.all([c6(process.env.CLAUDE_CODE_DISABLE_POLICY_SKILLS)?Promise.resolve([]):uh6(_,"policySettings"),WJ("userSettings")&&!$?uh6(K,"userSettings"):Promise.resolve([]),O?Promise.all(z.map((Z)=>uh6(Z,"projectSettings"))):Promise.resolve([]),O?Promise.all(Y.map((Z)=>uh6(_s(Z,".claude","skills"),"projectSettings"))):Promise.resolve([]),$?Promise.resolve([]):fez(q)]),M=[...A,...w,...j.flat(),...H.flat(),...J],X=await Promise.all(M.map(({skill:Z,filePath:v})=>Z.type==="prompt"?Hez(v):Promise.resolve(null))),P=new Map,W=[];for(let Z=0;Z<M.length;Z++){let v=M[Z];if(v===void 0||v.skill.type!=="prompt")continue;let{skill:k}=v,V=X[Z];if(V===null||V===void 0){W.push(k);continue}let y=P.get(V);if(y!==void 0){N(`Skipping duplicate skill '${k.name}' from ${k.source} (same file already loaded from ${y})`);continue}P.set(V,k.source),W.push(k)}let D=M.length-W.length;if(D>0)N(`Deduplicated ${D} skills (same file)`);let f=[],G=[];for(let Z of W)if(Z.type==="prompt"&&Z.paths&&Z.paths.length>0&&!Jp8.has(Z.name))G.push(Z);else f.push(Z);for(let Z of G)mh6.set(Z.name,Z);if(G.length>0)N(`[skills] ${G.length} conditional skills stored (activated when matching files are touched)`);return N(`Loaded ${W.length} unique skills (${f.length} unconditional, ${G.length} conditional, managed: ${A.length}, user: ${w.length}, project: ${j.flat().length}, additional: ${H.flat().length}, legacy commands: ${J.length})`),f});rt1=new Set,zs=new Map,mh6=new Map,Jp8=new Set,et1=L_();xAK({createSkillCommand:at1,parseSkillFrontmatterFields:ot1})});function qe1(q){let K=kw();if(K.lastSessionId!==q)return;let _;if(K.lastModelUsage)_=Object.fromEntries(Object.entries(K.lastModelUsage).map(([z,Y])=>[z,{...Y,contextWindow:QT(z,gW()),maxOutputTokens:x16(z).default}]));

return _=!0,z.abortController?.abort(),z.unregisterCleanup?.(),{...z,status:"killed",endTime:Date.now(),evictAfter:z.retain?void 0:Date.now()+ML8,abortController:void 0,unregisterCleanup:void 0,selectedAgent:void 0}}),_)Sw(q)}function uDK(q,K){for(let[_,z]of Object.entries(q))if(z.type==="local_agent"&&z.status==="running")p46(_,K)}function mDK(q,K){w3(q,K,(_)=>{if(_.notified)return _;return{..._,notified:!0}})}function Wt6(q,K,_){w3(q,_,(z)=>{if(z.status!=="running")return z;let Y=z.progress?.summary;return{...z,progress:Y?{...K,summary:Y}:K}})}function o9K(q,K,_){let z=null;if(w3(q,_,(Y)=>{if(Y.status!=="running")return Y;return z={tokenCount:Y.progress?.tokenCount??0,toolUseCount:Y.progress?.toolUseCount??0,startTime:Y.startTime,toolUseId:Y.toolUseId},{...Y,progress:{...Y.progress,toolUseCount:Y.progress?.toolUseCount??0,tokenCount:Y.progress?.tokenCount??0,summary:K}}}),z&&RB()){let{tokenCount:Y,toolUseCount:$,startTime:O,toolUseId:A}=z;mR8({taskId:q,toolUseId:A,description:K,startTime:O,totalTokens:Y,toolUses:$,summary:K})}}function QR8(q,K){let _=q.agentId;w3(_,K,(z)=>{if(z.status!=="running")return z;return z.unregisterCleanup?.(),{...z,status:"completed",result:q,endTime:Date.now(),evictAfter:z.retain?void 0:Date.now()+ML8,abortController:void 0,unregisterCleanup:void 0,selectedAgent:void 0}}),Sw(_)}function dR8(q,K,_){w3(q,_,(z)=>{if(z.status!=="running")return z;return z.unregisterCleanup?.(),{...z,status:"failed",error:K,endTime:Date.now(),evictAfter:z.retain?void 0:Date.now()+ML8,abortController:void 0,unregisterCleanup:void 0,selectedAgent:void 0}}),Sw(q)}function qg8({agentId:q,description:K,prompt:_,selectedAgent:z,setAppState:Y,parentAbortController:$,toolUseId:O}){BH6(q,fW(sA(q)));

let _=[];_.push("=".repeat(80)),_.push(`QUERY PROFILING REPORT - Query #${sfK}`),_.push("=".repeat(80)),_.push("");let z=K[0]?.startTime??0,Y=z,$=0,O=0;for(let j of K){let H=j.startTime-z,J=j.startTime-Y;if(_.push(Kz8(H,J,j.name,j77.get(j.name),10,9,lzY(J,j.name))),j.name==="query_api_request_sent")$=H;if(j.name==="query_first_chunk_received")O=H;Y=j.startTime}let A=K[K.length-1],w=A?A.startTime-z:0;if(_.push(""),_.push("-".repeat(80)),O>0){let j=$,H=O-$,J=(j/O*100).toFixed(1),M=(H/O*100).toFixed(1);_.push(`Total TTFT: ${YI(O)}ms`),_.push(`  - Pre-request overhead: ${YI(j)}ms (${J}%)`),_.push(`  - Network latency: ${YI(H)}ms (${M}%)`)}else _.push(`Total time: ${YI(w)}ms`);return _.push(izY(K,z)),_.push("=".repeat(80)),_.join(`
`)}function izY(q,K){let _=[{name:"Context loading",start:"query_context_loading_start",end:"query_context_loading_end"},{name:"Microcompact",start:"query_microcompact_start",end:"query_microcompact_end"},{name:"Autocompact",start:"query_autocompact_start",end:"query_autocompact_end"},{name:"Query setup",start:"query_setup_start",end:"query_setup_end"},{name:"Tool schemas",start:"query_tool_schema_build_start",end:"query_tool_schema_build_end"},{name:"Message normalization",start:"query_message_normalization_start",end:"query_message_normalization_end"},{name:"Client creation",start:"query_client_creation_start",end:"query_client_creation_end"},{name:"Network TTFB",start:"query_api_request_sent",end:"query_first_chunk_received"},{name:"Tool execution",start:"query_tool_execution_start",end:"query_tool_execution_end"}],z=new Map(q.map((O)=>[O.name,O.startTime-K])),Y=[];Y.push(""),Y.push("PHASE BREAKDOWN:");for(let O of _){let A=z.get(O.start),w=z.get(O.end);if(A!==void 0&&w!==void 0){let j=w-A,H="█".repeat(Math.min(Math.ceil(j/10),50));Y.push(`  ${O.name.padEnd(22)} ${YI(j).padStart(10)}ms ${H}`)}}let $=z.get("query_api_request_sent");if($!==void 0)Y.push(""),Y.push(`  ${"Total pre-API overhead".padEnd(22)} ${YI($).padStart(10)}ms`);return Y.join(`
`)}function Eg8(){if(!b78)return;

let F=[...e2(k)],U=V,c=A.startsWith("agent:")||A.startsWith("repl_main_thread");F=await Wb4(F,v.contentReplacementState,c?(T6)=>void aH6(T6,v.agentId).catch(j6):void 0,new Set(v.options.tools.filter((T6)=>!Number.isFinite(T6.maxResultSizeChars)).map((T6)=>T6.name)));let K6=0;g3("query_microcompact_start"),F=(await H.microcompact(F,v,A)).messages;let q6=void 0;g3("query_microcompact_end");let t=tK(IZK(_,Y));g3("query_autocompact_start");let{compactionResult:n,consecutiveFailures:z6,consecutiveRapidRefills:M6,rapidRefillBreakerTripped:J6}=await H.autocompact(F,v,{systemPrompt:_,userContext:z,systemContext:Y,toolUseContext:v,forkContextMessages:F},A,U,K6);if(g3("query_autocompact_end"),J6)return d("tengu_auto_compact_rapid_refill_breaker",{consecutiveRapidRefills:U?.consecutiveRapidRefills??0,turnsSincePreviousCompact:U?.turnCounter??-1,queryChainId:g,queryDepth:C.depth}),yield U9({content:SZK,error:"invalid_request"}),{reason:"rapid_refill_breaker"};if(n){let{preCompactTokenCount:T6,postCompactTokenCount:s,truePostCompactTokenCount:$6,compactionUsage:h6}=n;if(d("tengu_auto_compact_succeeded",{originalMessageCount:k.length,compactedMessageCount:n.summaryMessages.length+n.attachments.length+n.hookResults.length,preCompactTokenCount:T6,postCompactTokenCount:s,truePostCompactTokenCount:$6,compactionInputTokens:h6?.input_tokens,compactionOutputTokens:h6?.output_tokens,compactionCacheReadTokens:h6?.cache_read_input_tokens??0,compactionCacheCreationTokens:h6?.cache_creation_input_tokens??0,compactionTotalTokens:h6?h6.input_tokens+(h6.cache_creation_input_tokens??0)+(h6.cache_read_input_tokens??0)+h6.output_tokens:0,queryChainId:g,queryDepth:C.depth}),q.taskBudget){let V6=qy8(F);X=Math.max(0,(X??q.taskBudget.total)-V6)}U={compacted:!0,turnId:H.uuid(),turnCounter:0,consecutiveFailures:0,consecutiveRapidRefills:M6};let P6=xa(n);for(let V6 of P6)yield V6;F=P6}else if(z6!==void 0)U={...U??{compacted:!1,turnId:"",turnCounter:0},consecutiveFailures:z6};v={...v,messages:F,turnStartIndex:LYY(F)};let G6=[],H6=[],e=[],a=!1;

if(q&&q.trim()!=="")K+=`

Additional Instructions:
${q}`;return K+=oZK,K}function gYY(q){let K=q;K=K.replace(/<analysis>[\s\S]*?<\/analysis>/,"");let _=K.match(/<summary>([\s\S]*?)<\/summary>/);if(_){let z=_[1]||"";K=K.replace(/<summary>[\s\S]*?<\/summary>/,`Summary:
${z.trim()}`)}return K=K.replace(/\n\n+/g,`

`),K.trim()}function B78(q,K,_,z,Y){let O=`This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

${gYY(q)}`;if(_)O+=`

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: ${_}`;if(z)O+=`

Recent messages are preserved verbatim.`;if(K)return`${O}
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.`;return O}var BYY,oZK;

let M=L8("tengu_compact_cache_prefix",!0),X=sZK(Y),P=n8({content:X}),W=q,D=_,f,G,Z=0;for(;;){if(f=await O0K({messages:W,summaryRequest:P,appState:j,context:K,preCompactTokenCount:w,cacheSafeParams:D,stripNonEssential:A}),G=KS6(f),!G?.startsWith(Dp))break;Z++;let n=Z<=K0K?_0K(W,f):null;if(!n)throw d("tengu_compact_failed",{reason:"prompt_too_long",preCompactTokenCount:w,promptCacheSharingEnabled:M,ptlAttempts:Z}),Error(z0K);d("tengu_compact_ptl_retry",{attempt:Z,droppedMessages:W.length-n.length,remainingMessages:n.length}),W=n,D={...D,forkContextMessages:n}}if(!G)throw N(`Compact failed: no summary text in response. Response: ${g6(f)}`,{level:"error"}),d("tengu_compact_failed",{reason:"no_summary",preCompactTokenCount:w,promptCacheSharingEnabled:M}),Error("Failed to generate conversation summary - response did not contain valid text content");else if(Ba(G))throw d("tengu_compact_failed",{reason:"api_error",preCompactTokenCount:w,promptCacheSharingEnabled:M}),Error(G);let v=Am1(K.readFileState);K.readFileState.clear(),K.loadedNestedMemoryPaths?.clear(),lo6(K.memorySelector);let[k,V]=await Promise.all([A0K(v,K,q0K),H0K(K)]),y=[...k,...V],E=dg8(K.agentId);if(E)y.push(E);let R=await j0K(K);if(R)y.push(R);let b=w0K(K.agentId);if(b)y.push(b);for(let n of cg8(K.options.tools,K.options.mainLoopModel,[],{callSite:"compact_full"}))y.push(P4(n));for(let n of lg8(K,[]))y.push(P4(n));for(let n of ng8(K.options.mcpClients,K.options.tools,K.options.mainLoopModel,[]))y.push(P4(n));K.onCompactProgress?.({type:"hooks_start",hookType:"session_start"});let I=await Kf("compact",{model:K.options.mainLoopModel}),m=F78($?"auto":"manual",w??0,q.at(-1)?.uuid),p=Rd(q);if(p.size>0)m.compactMetadata.preCompactDiscoveredTools=[...p].sort();let C=kY(),F=[n8({content:B78(G,z,C,void 0,!1),isCompactSummary:!0,isVisibleInTranscriptOnly:!0})],U=cN([f]),c=LV6([m,...F,...y,...I]),K6=tC(f),o=O?.querySource??K.options.querySource??"unknown";

d("tengu_compact",{preCompactTokenCount:w,stripNonEssential:A,postCompactTokenCount:U,truePostCompactTokenCount:c,autoCompactThreshold:O?.autoCompactThreshold??-1,willRetriggerNextTurn:O!==void 0&&c>=O.autoCompactThreshold,isAutoCompact:$,querySource:o,queryChainId:K.queryTracking?.chainId??"",queryDepth:K.queryTracking?.depth??-1,isRecompactionInChain:O?.isRecompactionInChain??!1,turnsSincePreviousCompact:O?.turnsSincePreviousCompact??-1,previousCompactTurnId:O?.previousCompactTurnId??"",compactionInputTokens:K6?.input_tokens,compactionOutputTokens:K6?.output_tokens,compactionCacheReadTokens:K6?.cache_read_input_tokens??0,compactionCacheCreationTokens:K6?.cache_creation_input_tokens??0,compactionTotalTokens:K6?K6.input_tokens+(K6.cache_creation_input_tokens??0)+(K6.cache_read_input_tokens??0)+K6.output_tokens:0,promptCacheSharingEnabled:M,...(()=>{try{return YP4(zP4(q))}catch(n){return j6(n),{}}})()}),Je(),ig8(),K.onCompactProgress?.({type:"hooks_start",hookType:"post_compact"});let q6=await rg8({trigger:$?"auto":"manual",compactSummary:G},K.abortController.signal),t=[J,q6.userDisplayMessage].filter(Boolean).join(`
`);return{boundaryMarker:m,summaryMessages:F,attachments:y,hookResults:I,userDisplayMessage:t||void 0,preCompactTokenCount:w,postCompactTokenCount:U,truePostCompactTokenCount:c,compactionUsage:K6}}catch(w){if(!$)$0K(w,K);throw w}finally{K.setStreamMode?.("requesting"),K.setResponseLength?.(()=>0),K.onCompactProgress?.({type:"compact_end"}),K.setSDKStatus?.(null)}}async function Y0K(q,K,_,z,Y,$="from"){try{let O=$==="up_to"?q.slice(0,K):q.slice(K),A=$==="up_to"?q.slice(K).filter((q6)=>q6.type!=="progress"&&!pJ(q6)&&!(q6.type==="user"&&q6.isCompactSummary)):q.slice(0,K).filter((q6)=>q6.type!=="progress");if(O.length===0)throw Error($==="up_to"?"Nothing to summarize before the selected message.":"Nothing to summarize after the selected message.");let w=SZ(q);_.onCompactProgress?.({type:"hooks_start",hookType:"pre_compact"}),_.setSDKStatus?.("compacting");

let j=await _S6({trigger:"manual",customInstructions:null},_.abortController.signal),H;if(j.newCustomInstructions&&Y)H=`${j.newCustomInstructions}

User context: ${Y}`;else if(j.newCustomInstructions)H=j.newCustomInstructions;else if(Y)H=`User context: ${Y}`;_.setStreamMode?.("requesting"),_.setResponseLength?.(()=>0),_.onCompactProgress?.({type:"compact_start"});let J=aZK(H,$),M=n8({content:J}),X={preCompactTokenCount:w,direction:$,messagesSummarized:O.length},P=$==="up_to"?O:q,W=$==="up_to"?{...z,forkContextMessages:O}:z,D,f,G=0;for(;;){if(D=await O0K({messages:P,summaryRequest:M,appState:_.getAppState(),context:_,preCompactTokenCount:w,cacheSafeParams:W}),f=KS6(D),!f?.startsWith(Dp))break;G++;let q6=G<=K0K?_0K(P,D):null;if(!q6)throw d("tengu_partial_compact_failed",{reason:"prompt_too_long",...X,ptlAttempts:G}),Error(z0K);d("tengu_compact_ptl_retry",{attempt:G,droppedMessages:P.length-q6.length,remainingMessages:q6.length,path:"partial"}),P=q6,W={...W,forkContextMessages:q6}}if(!f)throw d("tengu_partial_compact_failed",{reason:"no_summary",...X}),Error("Failed to generate conversation summary - response did not contain valid text content");else if(Ba(f))throw d("tengu_partial_compact_failed",{reason:"api_error",...X}),Error(f);let Z=Am1(_.readFileState);_.readFileState.clear(),_.loadedNestedMemoryPaths?.clear(),lo6(_.memorySelector);let[v,k]=await Promise.all([A0K(Z,_,q0K,A),H0K(_)]),V=[...v,...k],y=dg8(_.agentId);if(y)V.push(y);let E=await j0K(_);if(E)V.push(E);let R=w0K(_.agentId);if(R)V.push(R);for(let q6 of cg8(_.options.tools,_.options.mainLoopModel,A,{callSite:"compact_partial"}))V.push(P4(q6));for(let q6 of lg8(_,A))V.push(P4(q6));for(let q6 of ng8(_.options.mcpClients,_.options.tools,_.options.mainLoopModel,A))V.push(P4(q6));_.onCompactProgress?.({type:"hooks_start",hookType:"session_start"});let b=await Kf("compact",{model:_.options.mainLoopModel}),I=cN([D]),m=tC(D);

d("tengu_partial_compact",{preCompactTokenCount:w,postCompactTokenCount:I,messagesKept:A.length,messagesSummarized:O.length,direction:$,hasUserFeedback:!!Y,trigger:"message_selector",compactionInputTokens:m?.input_tokens,compactionOutputTokens:m?.output_tokens,compactionCacheReadTokens:m?.cache_read_input_tokens??0,compactionCacheCreationTokens:m?.cache_creation_input_tokens??0});let p=$==="up_to"?q.slice(0,K).findLast((q6)=>q6.type!=="progress")?.uuid:A.at(-1)?.uuid,C=F78("manual",w??0,p,Y,O.length),g=Rd(q);if(g.size>0)C.compactMetadata.preCompactDiscoveredTools=[...g].sort();let F=kY(),c=[n8({content:B78(f,!1,F,void 0,!1),isCompactSummary:!0,...A.length>0?{summarizeMetadata:{messagesSummarized:O.length,userContext:Y,direction:$}}:{isVisibleInTranscriptOnly:!0}})];Je(),ig8(),_.onCompactProgress?.({type:"hooks_start",hookType:"post_compact"});let K6=await rg8({trigger:"manual",compactSummary:f},_.abortController.signal),o=$==="up_to"?c.at(-1)?.uuid??C.uuid:C.uuid;return{boundaryMarker:b77(C,o,A),summaryMessages:c,messagesToKeep:A,attachments:V,hookResults:b,userDisplayMessage:K6.userDisplayMessage,preCompactTokenCount:w,postCompactTokenCount:I,compactionUsage:m}}catch(O){throw $0K(O,_),O}finally{_.setStreamMode?.("requesting"),_.setResponseLength?.(()=>0),_.onCompactProgress?.({type:"compact_end"}),_.setSDKStatus?.(null)}}function $0K(q,K){if(!Ee(q,ba)&&!Ee(q,qS6))K.addNotification?.({key:"error-compacting-conversation",text:"Error compacting conversation",priority:"immediate",color:"error"})}function rYY(){return async()=>({behavior:"deny",message:"Tool use is not allowed during compaction",decisionReason:{type:"other",reason:"compaction agent should only produce text summary"}})}async function O0K({messages:q,summaryRequest:K,appState:_,context:z,preCompactTokenCount:Y,cacheSafeParams:$,stripNonEssential:O=!1}){let A=!O&&L8("tengu_compact_cache_prefix",!0),w=pfK()?setInterval((j)=>{mfK(),j?.("compacting")},30000,z.setSDKStatus):void 0;

function P0K(){return`IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, claude.md entries, or any past session summaries), update the session notes file.

The file {{notesPath}} has already been read for you. Here are its current contents:
<current_notes_content>
{{currentNotes}}
</current_notes_content>

Your ONLY task is to use the Edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed) - make all Edit tool calls in parallel in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Do not include information that's already in the CLAUDE.md files included in the context
- Keep each section under ~${og8} tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation
- IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction

Use the Edit tool with file_path: {{notesPath}}

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_.`}async function m77(){let q=f0K(q7(),"session-memory","config","template.md");

$.push(O),Y+=O.length+1}return $.push(`
[... section truncated for length ...]`),{lines:$,wasTruncated:!0}}var og8=2000,M0K=12000,X0K=`
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
`;var p77=L(()=>{UN();d8();E8();h8()});function z$Y(q){g77={...g77,...q}}function Y$Y(){return{...g77}}async function $$Y(){if(T0K)return;T0K=!0;let q=await jC("tengu_sm_compact_config",{}),K={minTokens:q.minTokens&&q.minTokens>0?q.minTokens:ag8.minTokens,minTextBlockMessages:q.minTextBlockMessages&&q.minTextBlockMessages>0?q.minTextBlockMessages:ag8.minTextBlockMessages,maxTokens:q.maxTokens&&q.maxTokens>0?q.maxTokens:ag8.maxTokens};z$Y(K)}function k0K(q){if(q.type==="assistant")return q.message.content.some((_)=>_.type==="text");if(q.type==="user"){let K=q.message.content;if(typeof K==="string")return K.length>0;if(Array.isArray(K))return K.some((_)=>_.type==="text")}return!1}function O$Y(q){if(q.type!=="user")return[];

M+=`

Some session memory sections were truncated for length. The full session memory can be viewed at: ${f}`}let X=[n8({content:M,isCompactSummary:!0,isVisibleInTranscriptOnly:!0})],P=dg8($),W=P?[P]:[],D=Vo6(X);return{boundaryMarker:b77(w,X.at(-1).uuid,_),summaryMessages:X,attachments:W,hookResults:z,messagesToKeep:_,preCompactTokenCount:A,postCompactTokenCount:D,truePostCompactTokenCount:D}}async function tg8(q,K,_,z){if(!sg8())return null;await $$Y(),await VX4();let Y=vX4(),$=await _y8();if(!$)return d("tengu_sm_compact_no_session_memory",{}),null;if(await Z0K($))return d("tengu_sm_compact_empty_template",{}),null;try{let O;if(Y){if(O=q.findIndex((P)=>P.uuid===Y),O===-1)return d("tengu_sm_compact_summarized_id_not_found",{}),null}else O=q.length-1,d("tengu_sm_compact_resumed_session",{});let A=w$Y(q,O),w=q.slice(A).filter((P)=>!pJ(P)),j=await Kf("compact",{model:D5()}),H=kY(),J=j$Y(q,$,w,j,H,K,z),M=xa(J),X=Vo6(M);if(_!==void 0&&X>=_)return d("tengu_sm_compact_threshold_exceeded",{postCompactTokenCount:X,autoCompactThreshold:_}),null;return{...J,postCompactTokenCount:X,truePostCompactTokenCount:X}}catch(O){return d("tengu_sm_compact_error",{}),null}}var ag8,g77,T0K=!1;var eg8=L(()=>{_8();d8();E8();a1();dq();Nz();sK6();t4();CZ();eC();l1();k8();p77();SV6();Ia();aC();C77();ag8={minTokens:1e4,minTextBlockMessages:5,maxTokens:40000},g77={...ag8}});function y0K(q){let K=q.trim().toLowerCase(),_;if(K.endsWith("m"))_=parseFloat(K)*1e6;else if(K.endsWith("k"))_=parseFloat(K)*1000;else{let z=parseInt(K,10);_=z>=100&&z<=1000?z*1000:z}if(!Number.isFinite(_)||_<F77||_>N0K)return;return Math.round(_)}function I56(q,K){let _=QT(q,gW());if(process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW){let z=pU("CLAUDE_CODE_AUTO_COMPACT_WINDOW",process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW,F77,N0K);if(z.status!=="invalid"){let Y=Math.max(F77,z.effective);return{window:Math.min(_,Y),configured:Y,source:"env"}}}if(K!==void 0)return{window:Math.min(_,K),configured:K,source:"settings"};

return{window:_,configured:_,source:"model"}}function Sd(q,K){let _=Math.min(U78(q),H$Y),z=f0()?K:void 0,{window:Y}=I56(q,z);return Y-_}function P$Y(){return Date.now()-DR()>=X$Y}function a68(q,K){let _=Sd(q,K),z=_-Q77,Y=process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;if(Y){let $=parseFloat(Y);if(!isNaN($)&&$>0&&$<=100){let O=Math.floor(_*($/100));return Math.min(O,z)}}return z}function oH6(q,K,_){let z=f0(),Y=z?_:void 0,$=a68(K,Y),O=z?$:Sd(K,Y),A=Math.max(0,Math.round((O-q)/O*100)),w=O-J$Y,j=O-M$Y,H=q>=w,J=q>=j,M=z&&q>=$,P=Sd(K,Y)-d77,W=process.env.CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE,D=W?parseInt(W,10):NaN,f=!isNaN(D)&&D>0?D:P,G=q>=f;return{percentLeft:A,isAboveWarningThreshold:H,isAboveErrorThreshold:J,isAboveAutoCompactThreshold:M,isAtBlockingLimit:G}}function f0(){if(c6(process.env.DISABLE_COMPACT))return!1;if(c6(process.env.DISABLE_AUTO_COMPACT))return!1;return w8().autoCompactEnabled}async function W$Y(q,K,_,z,Y=0){if(z==="session_memory"||z==="compact")return!1;if(!f0())return!1;let $=SZ(q)-Y,O=a68(K,_),A=Sd(K,_);N(`autocompact: tokens=${$} threshold=${O} effectiveWindow=${A}${Y>0?` snipFreed=${Y}`:""}`);let{isAboveAutoCompactThreshold:w}=oH6($,K,_);return w}async function LZK(q,K,_,z,Y,$){if(c6(process.env.DISABLE_COMPACT))return{wasCompacted:!1};if(Y?.consecutiveFailures!==void 0&&Y.consecutiveFailures>=V0K)return{wasCompacted:!1};let O=K.options.mainLoopModel,A=K.getAppState().autoCompactWindow;if(!await W$Y(q,O,A,z,$))return{wasCompacted:!1};let H=Y?.compacted===!0&&Y.turnCounter<U77?(Y?.consecutiveRapidRefills??0)+1:0;if(H>=E0K)return N(`autocompact: rapid-refill breaker tripped — ${H} consecutive refills within <${U77} turns each (last was ${Y?.turnCounter} turns)`,{level:"warn"}),{wasCompacted:!1,rapidRefillBreakerTripped:!0};let J={isRecompactionInChain:Y?.compacted===!0,turnsSincePreviousCompact:Y?.turnCounter??-1,previousCompactTurnId:Y?.turnId,autoCompactThreshold:a68(O,A),querySource:z},M=await tg8(q,K.agentId,J.autoCompactThreshold,!1);

eH6=$1(()=>{let q=L8("tengu_amber_wren",{}),K=typeof q?.maxSizeBytes==="number"&&Number.isFinite(q.maxSizeBytes)&&q.maxSizeBytes>0?q.maxSizeBytes:$51,z=qOY()??(typeof q?.maxTokens==="number"&&Number.isFinite(q.maxTokens)&&q.maxTokens>0?q.maxTokens:e$Y),Y=typeof q?.includeMaxSizeInPrompt==="boolean"?q.includeMaxSizeInPrompt:void 0,$=typeof q?.targetedRangeNudge==="boolean"?q.targetedRangeNudge:void 0;return{maxSizeBytes:K,maxTokens:z,includeMaxSizeInPrompt:Y,targetedRangeNudge:$}})});function wF8(q){let K=`${Gh6()}/`,_=".output";if(q.startsWith(K)&&q.endsWith(".output")){let z=q.slice(K.length,-7);if(z.length>0&&z.length<=20&&/^[a-zA-Z0-9_-]+$/.test(z))return z}return null}function i0K({file_path:q,offset:K,limit:_,pages:z},{verbose:Y}){if(!q)return null;if(wF8(q))return"";let $=Y?q:m5(q);if(z)return RK.createElement(RK.Fragment,null,RK.createElement(O0,{filePath:q},$),` · pages ${z}`);if(Y&&(K||_)){let O=K??1,A=_?`lines ${O}-${O+_-1}`:`from line ${O}`;return RK.createElement(RK.Fragment,null,RK.createElement(O0,{filePath:q},$),` · ${A}`)}return RK.createElement(O0,{filePath:q},$)}function r0K({file_path:q}){let K=q?wF8(q):null;if(!K)return null;return RK.createElement(T,{dimColor:!0}," ",K)}function o0K(q){switch(q.type){case"image":{let{originalSize:K}=q.file,_=B4(K);return RK.createElement(_1,{height:1},RK.createElement(T,null,"Read image (",_,")"))}case"notebook":{let{cells:K}=q.file;if(!K||K.length<1)return RK.createElement(T,{color:"error"},"No cells found in notebook");return RK.createElement(_1,{height:1},RK.createElement(T,null,"Read ",RK.createElement(T,{bold:!0},K.length)," cells"))}case"pdf":{let{originalSize:K}=q.file,_=B4(K);return RK.createElement(_1,{height:1},RK.createElement(T,null,"Read PDF (",_,")"))}case"parts":return RK.createElement(_1,{height:1},RK.createElement(T,null,"Read ",RK.createElement(T,{bold:!0},q.file.count)," ",q.file.count===1?"page":"pages"," (",B4(q.file.originalSize),")"));case"text":{let{numLines:K}=q.file;

let $=iV6(z),O=$.split("/")[1]||"png",A;try{let j=await nL(z,Y,O);A=jF8(j.buffer,j.mediaType,Y,j.dimensions)}catch(j){if(j instanceof lU)throw j;j6(j),A=jF8(z,O,Y)}if(Math.ceil(A.file.base64.length*0.125)>K)try{let j=await FD4(z,K,$);return{type:"image",file:{base64:j.base64,type:j.mediaType,originalSize:Y}}}catch(j){j6(j);try{let H=await Promise.resolve().then(() => w6(Kp1(),1)),M=await(H.default||H)(z).resize(400,400,{fit:"inside",withoutEnlargement:!0}).jpeg({quality:20}).toBuffer();return jF8(M,"jpeg",Y)}catch(H){return j6(H),jF8(z,O,Y)}}return A}var $OY,AOY,HF8,KGK,HOY,JOY,uz,POY=`

<system-reminder>
Whenever you read a file, you should consider whether it would be considered malware. You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer questions about the code behavior.
</system-reminder>
`,WOY,_GK;var hd=L(()=>{u7();l51();l1();k8();nA();UN();Ys();aq();F7();d8();E8();yK();Zp8();I7();e7();zb();h8();v78();a1();dq();Bp8();i_();Oq7();Q08();Nz();Vo();oR6();M18();r8();Aq7();ZY();t0K();$OY=new Set(["/dev/zero","/dev/random","/dev/urandom","/dev/full","/dev/stdin","/dev/tty","/dev/console","/dev/stdout","/dev/stderr","/dev/fd/0","/dev/fd/1","/dev/fd/2"]);AOY=String.fromCharCode(8239);HF8=class HF8 extends Error{tokenCount;maxTokens;constructor(q,K){super(`File content (${q} tokens) exceeds maximum allowed tokens (${K}). Use offset and limit parameters to read specific portions of the file, or search for specific content instead of reading the whole file.`);this.tokenCount=q;this.maxTokens=K;this.name="MaxFileReadTokenExceededError"}};KGK=new Set(["png","jpg","jpeg","gif","webp"]);

return[n8({content:Nv(z.join(" ")),isMeta:!0})]}case"async_hook_response":{let _=q.response,z=[];if(_.systemMessage)z.push(n8({content:_.systemMessage,isMeta:!0}));if(_.hookSpecificOutput&&"additionalContext"in _.hookSpecificOutput&&_.hookSpecificOutput.additionalContext)z.push(n8({content:_.hookSpecificOutput.additionalContext,isMeta:!0}));return V9(z)}case"token_usage":return[n8({content:Nv(`Token usage: ${q.used}/${q.total}; ${q.remaining} remaining`),isMeta:!0})];case"budget_usd":return[n8({content:Nv(`USD budget: $${q.used}/$${q.total}; $${q.remaining} remaining`),isMeta:!0})];case"output_token_usage":{let _=q.budget!==null?`${pK(q.turn)} / ${pK(q.budget)}`:pK(q.turn);return[n8({content:Nv(`Output tokens — turn: ${_} · session: ${pK(q.session)}`),isMeta:!0})]}case"hook_blocking_error":return[n8({content:Nv(`${q.hookName} hook blocking error from command: "${q.blockingError.command}": ${q.blockingError.blockingError}`),isMeta:!0})];case"hook_success":if(q.hookEvent!=="SessionStart"&&q.hookEvent!=="UserPromptSubmit")return[];if(q.content==="")return[];return[n8({content:Nv(`${q.hookName} hook success: ${q.content}`),isMeta:!0})];case"hook_additional_context":{if(q.content.length===0)return[];return[n8({content:Nv(`${q.hookName} hook additional context: ${q.content.join(`
`)}`),isMeta:!0})]}case"hook_stopped_continuation":return[n8({content:Nv(`${q.hookName} hook stopped continuation: ${q.message}`),isMeta:!0})];case"compaction_reminder":return V9([n8({content:"Auto-compact is enabled. When the context window is nearly full, older messages will be automatically summarized so you can continue working seamlessly. There is no need to stop or rush — you have unlimited context through automatic compaction.",isMeta:!0})]);case"context_efficiency":return[];case"date_change":return V9([n8({content:`The date has changed. Today's date is now ${q.newDate}. DO NOT mention this to the user explicitly because they are already aware.`,isMeta:!0})]);

if(K[30]!==R||K[31]!==_)q6=()=>_(`Auto-compact window unchanged: ${R}`),K[30]=R,K[31]=_,K[32]=q6;else q6=K[32];let t,n;if(K[33]===Symbol.for("react.memo_cache_sentinel"))t=LA.createElement(T,null,"This command configures when auto-compaction happens. The actual threshold is the minimum of this setting and your model's context window."),n=!J&&LA.createElement(T,{color:"warning"},"Auto-compact is currently disabled (see /config)"),K[33]=t,K[34]=n;else t=K[33],n=K[34];let z6;if(K[35]!==K6||K[36]!==X)z6=X?LA.createElement(T,{color:"warning"},"CLAUDE_CODE_AUTO_COMPACT_WINDOW is set and takes precedence. Unset it to change this setting here."):LA.createElement(u,null,LA.createElement(T,null,"Select auto-compact window: "),LA.createElement(T,{bold:!0,color:"suggestion"},K6)),K[35]=K6,K[36]=X,K[37]=z6;else z6=K[37];let M6;if(K[38]===Symbol.for("react.memo_cache_sentinel"))M6=LA.createElement(u,{flexDirection:"column",marginTop:1},LA.createElement(T,{bold:!0},"Long context that holds up"),LA.createElement(T,null,"Both Opus 4.6 and Sonnet 4.6 achieve state-of-the-art scores on long-context retrieval benchmarks at 1M tokens — Opus 4.6 scores 78.3% on MRCR v2, the highest among frontier models at that length. Opus 4.6 includes 1M context at standard pricing; Sonnet 4.6 1M is available with overages."),LA.createElement(T,{dimColor:!0},"Learn more: ",SJY)),K[38]=M6;else M6=K[38];let J6;if(K[39]!==z6)J6=LA.createElement(u,{flexDirection:"column",gap:1},t,n,z6,M6),K[39]=z6,K[40]=J6;else J6=K[40];let G6;if(K[41]!==o||K[42]!==q6||K[43]!==J6)G6=LA.createElement(h1,{title:"Auto-compact",subtitle:o,onCancel:q6,inputGuide:bJY},J6),K[41]=o,K[42]=q6,K[43]=J6,K[44]=G6;else G6=K[44];return G6}function bJY(){return LA.createElement(T,{dimColor:!0},"↑/↓ to change · Enter to apply · Esc to cancel")}function xJY(q){return q.autoCompactWindow}var LA,rK7,SJY="https://claude.com/blog/1m-context-ga",lK7=1e5,nK7=1e5,iK7=1e6,mS6=0,IJY=async(q,K,_)=>{let z=_?.trim()||"";if(z){let Y=Bq8(z,K);

return{...r6,model:q8}}return{...S6,model:q8}})}function f8(P6){S8((V6)=>({...V6,verbose:P6})),J({...w8(),verbose:P6}),n((V6)=>({...V6,verbose:P6})),M6((V6)=>{if("verbose"in V6){let{verbose:S6,...q8}=V6;return q8}return{...V6,verbose:P6}})}let k6=[{id:"autoCompactEnabled",label:"Auto-compact",value:H.autoCompactEnabled,type:"boolean",onChange(P6){S8((V6)=>({...V6,autoCompactEnabled:P6})),J({...w8(),autoCompactEnabled:P6}),d("tengu_auto_compact_setting_changed",{enabled:P6})}},{id:"spinnerTipsEnabled",label:"Show tips",value:X?.spinnerTipsEnabled??!0,type:"boolean",onChange(P6){P7("localSettings",{spinnerTipsEnabled:P6}),P((V6)=>({...V6,spinnerTipsEnabled:P6})),d("tengu_tips_setting_changed",{enabled:P6})}},{id:"prefersReducedMotion",label:"Reduce motion",value:X?.prefersReducedMotion??!1,type:"boolean",onChange(P6){P7("localSettings",{prefersReducedMotion:P6}),P((V6)=>({...V6,prefersReducedMotion:P6})),n((V6)=>({...V6,settings:{...V6.settings,prefersReducedMotion:P6}})),d("tengu_reduce_motion_setting_changed",{enabled:P6})}},{id:"thinkingEnabled",label:"Thinking mode",value:c??!0,type:"boolean",onChange(P6){n((V6)=>({...V6,thinkingEnabled:P6})),P7("userSettings",{alwaysThinkingEnabled:P6?void 0:!1}),d("tengu_thinking_toggled",{enabled:P6})}},...gK()&&AM()?[{id:"fastMode",label:`Fast mode (${wu} only)`,value:!!K6,type:"boolean",onChange(P6){if(yY6(),P7("userSettings",{fastMode:P6?!0:void 0}),P6)n((V6)=>({...V6,mainLoopModel:$Q6(),mainLoopModelForSession:null,fastMode:!0})),M6((V6)=>({...V6,model:$Q6(),"Fast mode":"ON"}));

return N(`Migrated stats cache from v${z.version} to v${vJ6}`),await lq8(Y),Y}if(!Array.isArray(z.dailyActivity)||!Array.isArray(z.dailyModelTokens)||typeof z.totalSessions!=="number"||typeof z.totalMessages!=="number")return N("Stats cache has invalid structure, returning empty cache"),w57();return z}catch(_){return N(`Failed to load stats cache: ${F6(_)}`),w57()}}async function lq8(q){let K=M8(),_=FNK(),z=`${_}.${kMY(8).toString("hex")}.tmp`;try{let Y=q7();await K.mkdir(Y);let $=g6(q,null,2),O=await VMY(z,"w",384);try{await O.writeFile($,{encoding:"utf-8"}),await O.sync()}finally{await O.close()}await K.rename(z,_),N(`Stats cache saved successfully (lastComputedDate: ${q.lastComputedDate})`)}catch(Y){j6(Y);try{await K.unlink(z)}catch{}}}function j57(q,K,_){let z=new Map;for(let M of q.dailyActivity)z.set(M.date,{...M});for(let M of K.dailyActivity){let X=z.get(M.date);if(X)X.messageCount+=M.messageCount,X.sessionCount+=M.sessionCount,X.toolCallCount+=M.toolCallCount;else z.set(M.date,{...M})}let Y=new Map;for(let M of q.dailyModelTokens)Y.set(M.date,{...M.tokensByModel});for(let M of K.dailyModelTokens){let X=Y.get(M.date);if(X)for(let[P,W]of Object.entries(M.tokensByModel))X[P]=(X[P]||0)+W;else Y.set(M.date,{...M.tokensByModel})}let $={...q.modelUsage};for(let[M,X]of Object.entries(K.modelUsage))if($[M])$[M]={inputTokens:$[M].inputTokens+X.inputTokens,outputTokens:$[M].outputTokens+X.outputTokens,cacheReadInputTokens:$[M].cacheReadInputTokens+X.cacheReadInputTokens,cacheCreationInputTokens:$[M].cacheCreationInputTokens+X.cacheCreationInputTokens,webSearchRequests:$[M].webSearchRequests+X.webSearchRequests,costUSD:$[M].costUSD+X.costUSD,contextWindow:Math.max($[M].contextWindow,X.contextWindow),maxOutputTokens:Math.max($[M].maxOutputTokens,X.maxOutputTokens)};else $[M]={...X};let O={...q.hourCounts};for(let[M,X]of Object.entries(K.hourCounts)){let P=parseInt(M,10);O[P]=(O[P]||0)+X}let A=q.totalSessions+K.sessionStats.length,w=q.totalMessages+K.sessionStats.reduce((M,X)=>M+X.messageCount,0),j=q.longestSession;

if(K)for(let[Z,v]of Object.entries(K.modelUsage))if(Y[Z])Y[Z]={inputTokens:Y[Z].inputTokens+v.inputTokens,outputTokens:Y[Z].outputTokens+v.outputTokens,cacheReadInputTokens:Y[Z].cacheReadInputTokens+v.cacheReadInputTokens,cacheCreationInputTokens:Y[Z].cacheCreationInputTokens+v.cacheCreationInputTokens,webSearchRequests:Y[Z].webSearchRequests+v.webSearchRequests,costUSD:Y[Z].costUSD+v.costUSD,contextWindow:Math.max(Y[Z].contextWindow,v.contextWindow),maxOutputTokens:Math.max(Y[Z].maxOutputTokens,v.maxOutputTokens)};else Y[Z]={...v};let $=new Map;for(let[Z,v]of Object.entries(q.hourCounts))$.set(parseInt(Z,10),v);if(K)for(let[Z,v]of Object.entries(K.hourCounts)){let k=parseInt(Z,10);$.set(k,($.get(k)||0)+v)}let O=Array.from(_.values()).sort((Z,v)=>Z.date.localeCompare(v.date)),A=KyK(O),w=Array.from(z.entries()).map(([Z,v])=>({date:Z,tokensByModel:v})).sort((Z,v)=>Z.date.localeCompare(v.date)),j=q.totalSessions+(K?.sessionStats.length||0),H=q.totalMessages+(K?.totalMessages||0),J=q.longestSession;if(K){for(let Z of K.sessionStats)if(!J||Z.duration>J.duration)J=Z}let M=q.firstSessionDate,X=null;if(K)for(let Z of K.sessionStats){if(!M||Z.timestamp<M)M=Z.timestamp;if(!X||Z.timestamp>X)X=Z.timestamp}if(!X&&O.length>0)X=O.at(-1).date;let P=O.length>0?O.reduce((Z,v)=>v.messageCount>Z.messageCount?v:Z).date:null,W=$.size>0?Array.from($.entries()).reduce((Z,[v,k])=>k>Z[1]?[v,k]:Z)[0]:null,D=M&&X?Math.ceil((new Date(X).getTime()-new Date(M).getTime())/86400000)+1:0,f=q.totalSpeculationTimeSavedMs+(K?.totalSpeculationTimeSavedMs||0);return{totalSessions:j,totalMessages:H,totalDays:D,activeDays:_.size,streaks:A,dailyActivity:O,dailyModelTokens:w,longestSession:J,modelUsage:Y,firstSessionDate:M,lastSessionDate:X,peakActivityDay:P,peakActivityHour:W,totalSpeculationTimeSavedMs:f}}async function zXY(){let q=await qyK();if(q.length===0)return _yK();let K=await gNK(async()=>{let Y=await UNK(),$=QNK(),O=Y;if(!Y.lastComputedDate){N("Stats cache empty, processing all historical data");let A=await rU8(q,{toDate:$});

if(A.project_path)_.projects[A.project_path]=(_.projects[A.project_path]||0)+1;let w=K.get(A.session_id);if(w){for(let[j,H]of gC6(w.goal_categories))if(H>0)_.goal_categories[j]=(_.goal_categories[j]||0)+H;_.outcomes[w.outcome]=(_.outcomes[w.outcome]||0)+1;for(let[j,H]of gC6(w.user_satisfaction_counts))if(H>0)_.satisfaction[j]=(_.satisfaction[j]||0)+H;_.helpfulness[w.claude_helpfulness]=(_.helpfulness[w.claude_helpfulness]||0)+1,_.session_types[w.session_type]=(_.session_types[w.session_type]||0)+1;for(let[j,H]of gC6(w.friction_counts))if(H>0)_.friction[j]=(_.friction[j]||0)+H;if(w.primary_success!=="none")_.success[w.primary_success]=(_.success[w.primary_success]||0)+1}if(_.session_summaries.length<50)_.session_summaries.push({id:A.session_id.slice(0,8),date:A.start_time.split("T")[0]||"",summary:A.summary||A.first_prompt.slice(0,100),goal:w?.underlying_goal})}if(z.sort(),_.date_range.start=z[0]?.split("T")[0]||"",_.date_range.end=z[z.length-1]?.split("T")[0]||"",_.user_response_times=Y,Y.length>0){let A=[...Y].sort((w,j)=>w-j);_.median_response_time=A[Math.floor(A.length/2)]||0,_.avg_response_time=Y.reduce((w,j)=>w+j,0)/Y.length}let O=new Set(z.map((A)=>A.split("T")[0]));return _.days_active=O.size,_.messages_per_day=_.days_active>0?Math.round(_.total_messages/_.days_active*10)/10:0,_.message_hours=$,_.multi_clauding=gUK(q),_}async function mUK(q,K){try{let _=await Lc8({systemPrompt:tK([]),userPrompt:q.prompt+`

DATA:
`+K,signal:new AbortController().signal,options:{model:EbY(),querySource:"insights",agents:[],isNonInteractiveSession:!0,hasAppendSystemPrompt:!1,mcpTools:[],maxOutputTokensOverride:q.maxTokens}}),z=Z3(_.message.content);if(z){let Y=z.match(/\{[\s\S]*\}/);

let j=w.project_areas?.areas?.map((G)=>`- ${G.name}: ${G.description}`).join(`
`)||"",H=w.what_works?.impressive_workflows?.map((G)=>`- ${G.title}: ${G.description}`).join(`
`)||"",J=w.friction_analysis?.categories?.map((G)=>`- ${G.category}: ${G.description}`).join(`
`)||"",M=w.suggestions?.features_to_try?.map((G)=>`- ${G.feature}: ${G.one_liner}`).join(`
`)||"",X=w.suggestions?.usage_patterns?.map((G)=>`- ${G.title}: ${G.suggestion}`).join(`
`)||"",P=w.on_the_horizon?.opportunities?.map((G)=>`- ${G.title}: ${G.whats_possible}`).join(`
`)||"",D={name:"at_a_glance",prompt:`You're writing an "At a Glance" summary for a Claude Code usage insights report for Claude Code users. The goal is to help them understand their usage and improve how they can use Claude better, especially as models improve.

Use this 4-part structure:

1. **What's working** - What is the user's unique style of interacting with Claude and what are some impactful things they've done? You can include one or two details, but keep it high level since things might not be fresh in the user's memory. Don't be fluffy or overly complimentary. Also, don't focus on the tool calls they use.

2. **What's hindering you** - Split into (a) Claude's fault (misunderstandings, wrong approaches, bugs) and (b) user-side friction (not providing enough context, environment issues -- ideally more general than just one project). Be honest but constructive.

3. **Quick wins to try** - Specific Claude Code features they could try from the examples below, or a workflow technique if you think it's really compelling. (Avoid stuff like "Ask Claude to confirm before taking actions" or "Type out more context up front" which are less compelling.)

4. **Ambitious workflows for better models** - As we move to much more capable models over the next 3-6 months, what should they prepare for? What workflows that seem impossible now will become possible? Draw from the appropriate section below.

Keep each section to 2-3 not-too-long sentences. Don't overwhelm the user. Don't mention specific numerical stats or underlined_categories from the session data below. Use a coaching tone.

RESPOND WITH ONLY A VALID JSON OBJECT:
{
  "whats_working": "(refer to instructions above)",
  "whats_hindering": "(refer to instructions above)",
  "quick_wins": "(refer to instructions above)",
  "ambitious_workflows": "(refer to instructions above)"
}

SESSION DATA:
${O}

## Project Areas (what user works on)
${j}

## Big Wins (impressive accomplishments)
${H}

## Friction Categories (where things go wrong)
${J}

## Features to Try
${M}

## Usage Patterns to Adopt
${X}

## On the Horizon (ambitious workflows for better models)
${P}`,maxTokens:8192},f=await mUK(D,"");

if(F<0||F>=R)break;if(I<0)I=F;let U=F+J+36;if(U+j<=R&&q.compare(w,0,j,U,U+j)===0)if(m<0)m=F;else(p??=[m]).push(F);C=F+J}let g=p?mxY(q,W,p):m>=0?m:I;if(g>=0){let F=g+J,U=q.toString("latin1",F,F+36);P.set(U,M.length),M.push(W,R,b)}else X.push(W,R)}else X.push(W,R);W=R}let f=-1;for(let E=M.length-3;E>=0;E-=3){let R=q.indexOf(O,M[E]);if(R===-1||R>=M[E+1]){f=E;break}}if(f<0)return q;let G=new Set,Z=new Set,v=0,k=f;while(k!==void 0){if(G.has(k))break;G.add(k),Z.add(M[k]),v+=M[k+1]-M[k];let E=M[k+2];if(E<0)break;let R=q.toString("latin1",E,E+36);k=P.get(R)}if(D-v<D>>1)return q;let V=[],y=0;for(let E=0;E<M.length;E+=3){let R=M[E];while(y<X.length&&X[y]<R)V.push(q.subarray(X[y],X[y+1])),y+=2;if(Z.has(R))V.push(q.subarray(R,M[E+1]))}while(y<X.length)V.push(q.subarray(X[y],X[y+1])),y+=2;return Buffer.concat(V)}function BxY(q,K,_,z){let O=Buffer.from('{"type":"attribution-snapshot"'),A=Buffer.from('"compact_boundary"'),w=Buffer.allocUnsafe(1048576),j=Buffer.allocUnsafe(O.length),H=oz7(q,"r"),J=-1,M=0,X=-1,P=0,W=(D,f,G,Z)=>{if(G>=O.length&&D.compare(O,0,O.length,f,f+O.length)===0){J=Z,M=G;return}let v=D.toString("utf8",f,f+G);if(D.includes(A,f)&&D.indexOf(A,f)<f+G){let V=l8(v);if(V?.type==="system"&&V.subtype==="compact_boundary"){if(!V.compactMetadata?.preservedSegment)z(),J=-1,M=0}}let k=l8(v);if(k)_(k)};try{while(P<K){let D=FC6(H,w,0,1048576,P);if(D===0)break;let f=0;for(let G=0;G<D;G++)if(w[G]===10){if(X>=0){let Z=P+G-X,v=Math.min(O.length,Z);if(FC6(H,j,0,v,X),v===O.length&&j.compare(O,0,O.length,0,O.length)===0)J=X,M=Z;else{let k=Buffer.allocUnsafe(Z);FC6(H,k,0,Z,X),W(k,0,Z,X)}X=-1}else if(G>f)W(w,f,G-f,P+f);f=G+1}if(f<D&&X<0)X=P+f;P+=D}if(X>=0){let D=K-X,f=Buffer.allocUnsafe(D);FC6(H,f,0,D,X),W(f,0,D,X)}}finally{rz7(H)}return{lastAttributionOffset:J,lastAttributionLength:M}}function gxY(q,K,_){if(K<0||_<=0)return null;let z=oz7(q,"r");try{let Y=Buffer.allocUnsafe(_);

if(Dq()==="firstParty"&&OM()&&(L8("tengu_fgts",!1)||c6(process.env.CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING)))$.eager_input_streaming=!0;Y.set(z,$)}let O={name:$.name,description:$.description,input_schema:$.input_schema,...$.strict&&{strict:!0},...$.eager_input_streaming&&{eager_input_streaming:!0}};if(K.deferLoading)O.defer_loading=!0;if(K.cacheControl)O.cache_control=K.cacheControl;if(c6(process.env.CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS)){let A=new Set(["name","description","input_schema","cache_control"]),w=Object.keys(O).filter((j)=>!A.has(j));if(w.length>0)return EuY(w),{name:O.name,description:O.description,input_schema:O.input_schema,...O.cache_control&&{cache_control:O.cache_control}}}return O}function EuY(q){if(BdK)return;BdK=!0,N(`[betas] Stripped from tool schemas: [${q.join(", ")}] (CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1)`)}function LuY(q){let K=gdK.get(q);if(K===void 0)K=g6(q),gdK.set(q,K);return K}function UdK(q){let[K]=QY7(q),_=K?.text;d("tengu_sysprompt_block",{snippet:_?.slice(0,20),length:_?.length??0,hash:_?VuY("sha256").update(_).digest("hex"):""})}function QY7(q,K){let _=rG6();if(_&&K?.skipGlobalCacheForSystemPrompt){d("tengu_sysprompt_using_tool_based_cache",{promptBlockCount:q.length});let w,j,H=[];for(let X of q){if(!X)continue;if(X===tH6)continue;if(X.startsWith("x-anthropic-billing-header"))w=X;else if(C08.has(X))j=X;else H.push(X)}let J=[];if(w)J.push({text:w,cacheScope:null});if(j)J.push({text:j,cacheScope:"org"});let M=H.join(`

`);if(M)J.push({text:M,cacheScope:"org"});return J}if(_){let w=q.findIndex((j)=>j===tH6);if(w!==-1){let j,H,J=[],M=[];for(let D=0;D<q.length;D++){let f=q[D];if(!f||f===tH6)continue;if(f.startsWith("x-anthropic-billing-header"))j=f;else if(C08.has(f))H=f;else if(D<w)J.push(f);else M.push(f)}let X=[];if(j)X.push({text:j,cacheScope:null});if(H)X.push({text:H,cacheScope:null});let P=J.join(`

`);if(P)X.push({text:P,cacheScope:"global"});let W=M.join(`

`);if(W)X.push({text:W,cacheScope:null});

if(m6)y=[n8({content:`<available-deferred-tools>
${m6}
</available-deferred-tools>`,isMeta:!0}),...y]}K=tK([x08(R),b08({isNonInteractive:$.isNonInteractiveSession,hasAppendSystemPrompt:$.hasAppendSystemPrompt}),...K,...H?[Km4]:[]].filter(Boolean)),UdK(K);let b=$.enablePromptCaching??rdK($.model),I=nuY(K,b,{skipGlobalCacheForSystemPrompt:Z,querySource:$.querySource}),m=j.length>0,p=[...$.extraToolSchemas??[]];if(H)p.push({type:"advisor_20260301",name:"advisor",model:H});let C=[...k,...p],g=gK()&&AM()&&!YF()&&GJ($.model)&&!!$.fastMode,F=ka8()===!0;if(!F&&w&&u16()&&(xuY?.isAutoModeActive()??!1))F=!0,Va8(!0);let U=Na8()===!0;if(!U&&g)U=!0,ya8(!0);let c=Ea8()===!0,K6=La8()===!0;if(!K6&&w){let m6=x96();if(m6!==null&&Date.now()-m6>YM4)K6=!0,ha8(!0)}let o=Mk6($.model,$.effortValue),q6=pH()?{systemPrompt:K.join(`

`),querySource:$.querySource,tools:g6(C)}:void 0,t=EQ4($.model,q6,y,g),n=Date.now(),z6=Date.now(),M6=0,J6=[],G6=void 0,H6=void 0,e=void 0,a=void 0;function _6(){if(duY(G6),G6=void 0,a)a.body?.cancel().catch(()=>{}),a=void 0}let l=W?MM4():null,i=W?XM4():[],A6,O6=(m6)=>{let b6=[...j];if(!b6.includes(zi)&&of8(m6.model))b6.push(zi);let T6=Dq()==="bedrock"?[...ZV1(m6.model),...P?[P]:[]]:[],s=x46(T6),$6={...s.output_config??{}};if(uuY(o,$6,s,b6,$.model),muY($.taskBudget,$6,b6),$.outputFormat&&!("format"in $6)){if($6.format=$.outputFormat,o$6($.model)&&!b6.includes(C86))b6.push(C86)}let h6=m6?.maxTokensOverride||$.maxOutputTokensOverride||U78($.model),P6=_.type!=="disabled"&&!c6(process.env.CLAUDE_CODE_DISABLE_THINKING),V6=void 0;if(P6&&I_4($.model))if(!c6(process.env.CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING)&&pk8($.model))V6={type:"adaptive"};else{let b8=Gxq($.model);if(_.type==="enabled"&&_.budgetTokens!==void 0)b8=_.budgetTokens;b8=Math.min(h6-1,b8),V6={budget_tokens:b8,type:"enabled"}}let S6=ndK({hasThinking:P6,isRedactThinkingActive:b6.includes(LX8),clearAllThinking:K6}),q8=$.enablePromptCaching??rdK(m6.model),e6;if(gK()&&AM()&&!YF()&&GJ($.model)&&!!m6.fastMode)e6="fast";if(U&&!b6.includes(UW1))b6.push(UW1);

N(`[presence] pulse → ${K}`),O1.post(K,{client_id:uBY,connected_at:p$7},{headers:{...Yb6.getAuthHeaders(),"anthropic-version":"2023-06-01","anthropic-client-platform":"cli"},timeout:YnK,validateStatus:()=>!0}).then((_)=>{if(_.status>=400)N(`[presence] pulse got ${_.status}`)},()=>{})}var YnK=5000,uBY,Yb6=null,m$7=null,p$7=null,B$7=0;var $nK=L(()=>{VK();sr8();T8();_8();l1();uBY=cx6()});import{readFile as pBY,stat as BBY}from"fs/promises";async function AnK(q,K,_){let[z,Y]=await Promise.all([K.readMain(),K.readSubagents()]),$=new Set;for(let j of z??[]){let H=j.payload.uuid;if(typeof H==="string")$.add(H)}for(let j of Y??[]){let H=j.payload.uuid;if(typeof H==="string")$.add(H)}N(`[persistence-sync] Server has ${$.size} events since compaction`);let O=(j)=>{N(`[persistence-sync] Write failed: ${j}`)},A=await OnK(Df(N8()),$);for(let j of A)q("transcript",j,{...pJ(j)&&{isCompaction:!0}}).catch(O);let w=0;for(let j of _){let H=await OnK(fW(j),$);for(let J of H)q("transcript",J,{...pJ(J)&&{isCompaction:!0},agentId:j}).catch(O);w+=H.length}return N(`[persistence-sync] Uploaded ${A.length} main + ${w} subagent entries`),{uploadedMain:A.length,uploadedSubagents:w}}async function OnK(q,K){let _;try{_=(await BBY(q)).size}catch(A){if(K7(A))return[];throw A}if(_>fD6)return N(`[persistence-sync] Skipping ${q} — ${_} bytes exceeds ${fD6} threshold`),[];let z;try{z=await pBY(q,"utf8")}catch(A){if(K7(A))return[];throw A}let Y=z.split(`
`).filter(Boolean),$=[],O=-1;for(let A=0;A<Y.length;A++){let w=Y[A];if(!w)continue;let j;try{j=l8(w)}catch{continue}if(!FBY(j))continue;if($.push(j),pJ(j))O=$.length-1}return $.slice(O+1).filter((A)=>!K.has(A.uuid))}function FBY(q){return typeof q==="object"&&q!==null&&"type"in q&&gBY.has(q.type)&&"uuid"in q&&typeof q.uuid==="string"}var gBY;var wnK=L(()=>{T8();_8();E8();a1();t4();W66();r8();gBY=new Set(["user","assistant","attachment","system"])});function kl8(q){if(q===null||typeof q!=="object")return q;let K=q;if("requestId"in K&&!("request_id"in K))K.request_id=K.requestId,delete K.requestId;

d("tengu_post_compact_survey_event",{event_type:"responded",appearance_id:q,response:K,session_memory_compaction_enabled:_}),QO("feedback_survey",{event_type:"responded",appearance_id:q,response:K,survey_type:"post_compact"})}function DrY(q){let K=sg8();d("tengu_post_compact_survey_event",{event_type:"appeared",appearance_id:q,session_memory_compaction_enabled:K}),QO("feedback_survey",{event_type:"appeared",appearance_id:q,survey_type:"post_compact"})}var i36,jrY=3000,HrY="tengu_post_compact_survey",JrY=0.2;var S65=L(()=>{t6();l16();l1();k8();eg8();d8();a1();vm();Yi8();i36=w6(D6(),1)});function C65(q){let K=Y6(11),{onSelect:_,inputValue:z,setInputValue:Y}=q,$;if(K[0]!==_)$=(M)=>_(ZrY[M]),K[0]=_,K[1]=$;else $=K[1];let O;if(K[2]!==z||K[3]!==Y||K[4]!==$)O={inputValue:z,setInputValue:Y,isValidDigit:GrY,onDigit:$},K[2]=z,K[3]=Y,K[4]=$,K[5]=O;else O=K[5];xb6(O);let A;if(K[6]===Symbol.for("react.memo_cache_sentinel"))A=U0.default.createElement(u,null,U0.default.createElement(T,{color:"ansi:cyan"},C9," "),U0.default.createElement(T,{bold:!0},"Can Anthropic look at your session transcript to help us improve Claude Code?")),K[6]=A;else A=K[6];let w;if(K[7]===Symbol.for("react.memo_cache_sentinel"))w=U0.default.createElement(u,{marginLeft:2},U0.default.createElement(T,{dimColor:!0},"Learn more: https://code.claude.com/docs/en/data-usage#session-quality-surveys")),K[7]=w;else w=K[7];let j;if(K[8]===Symbol.for("react.memo_cache_sentinel"))j=U0.default.createElement(u,{width:10},U0.default.createElement(T,null,U0.default.createElement(T,{color:"ansi:cyan"},"1"),": Yes")),K[8]=j;else j=K[8];let H;if(K[9]===Symbol.for("react.memo_cache_sentinel"))H=U0.default.createElement(u,{width:10},U0.default.createElement(T,null,U0.default.createElement(T,{color:"ansi:cyan"},"2"),": No")),K[9]=H;else H=K[9];let J;

if(y7.push(...L7),F8==="fork")E_K(x8,cX(j8));else Su8(x8,cX(j8));if(S58(x8,O6),x8.fileHistorySnapshots)hu8(x8);let{agentDefinition:yq}=dM6(x8.agentSetting,G,o);g(yq),O6((fK)=>({...fK,agent:yq?.agentType})),O6((fK)=>({...fK,standaloneAgentContext:C58(x8.agentName,x8.agentColor)})),BF(x8.agentName),ab6(y7,x8.projectPath??z7()),KR(),E_(null),vP(j8);let p4=qe1(j8);Xp8(),wP6(),uf(cX(j8),x8.fullPath?VsY(x8.fullPath):null);let{renameRecordingForSession:XK}=await Promise.resolve().then(() => (R58(),q65));if(await XK(),await jx(),Iq8(),Xc(x8),xA.current=!0,KA(void 0),F8!=="fork")K65(),b58(x8.worktreeSession),Mc(),Ds1({abortController:new AbortController,getAppState:()=>x6.getState(),setAppState:O6});else{let fK=t2();if(fK)uy(fK)}if(p4)tx6(p4);if(dv.current&&F8!=="fork")dv.current=Qh8(y7,x8.contentReplacements??[]);A4(()=>y7),C5(null),D9(""),d("tengu_session_resumed",{entrypoint:F8,success:!0,resume_duration_ms:Math.round(performance.now()-f7)})}catch(y7){throw d("tengu_session_resumed",{entrypoint:F8,success:!1}),y7}},[KR,O6]),[w38]=z1.useState(()=>Mm(uU)),_R=z1.useRef(w38),Y96=z1.useRef(new Set),jE=z1.useRef(0),AX6=z1.useRef(new Set),$96=z1.useRef(new Set),cx=z1.useRef(bq6()),ab6=z1.useCallback((j8,x8)=>{let F8=yN6(j8,x8,uU);_R.current=NV6(_R.current,F8);for(let f7 of Ng1(j8))Y96.current.add(f7)},[]);z1.useEffect(()=>{if(z&&z.length>0){if(ab6(z,z7()),Ds1({abortController:new AbortController,getAppState:()=>x6.getState(),setAppState:O6}),L8("tengu_gleaming_fair",!1)){let j8=Number(process.env.CLAUDE_CODE_RESUME_THRESHOLD_MINUTES??70),x8=Number(process.env.CLAUDE_CODE_RESUME_TOKEN_THRESHOLD??1e5),F8=Date.now()-60000,f7=z.findLast((y7)=>(y7.type==="user"||y7.type==="assistant")&&Date.parse(y7.timestamp)<F8)?.timestamp;if(f7&&!w8().resumeReturnDismissed){let y7=(Date.now()-Date.parse(f7))/60000;if(y7>=j8)Promise.resolve().then(() => (CZ(),DX4)).then(({tokenCountWithEstimation:bq})=>{let L7=bq(z);if(L7>=x8)db6({sessionAgeMinutes:y7,estimatedTokens:L7})})}}}},[]);

}
\`\`\`

---

## Thinking

**Adaptive thinking is the recommended mode for Claude 4.6+ models.** Claude decides dynamically when and how much to think. The builder has a direct \`.thinking(ThinkingConfigAdaptive)\` overload — no manual union wrapping.

\`\`\`java
import com.anthropic.models.messages.ContentBlock;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;
import com.anthropic.models.messages.ThinkingConfigAdaptive;

MessageCreateParams params = MessageCreateParams.builder()
    .model(Model.CLAUDE_SONNET_4_6)
    .maxTokens(16000L)
    .thinking(ThinkingConfigAdaptive.builder().build())
    .addUserMessage("Solve this step by step: 27 * 453")
    .build();

for (ContentBlock block : client.messages().create(params).content()) {
    block.thinking().ifPresent(t -> System.out.println("[thinking] " + t.thinking()));
    block.text().ifPresent(t -> System.out.println(t.text()));
}
\`\`\`

> **Deprecated:** \`ThinkingConfigEnabled.builder().budgetTokens(N)\` (and the \`.enabledThinking(N)\` shortcut) still works on Claude 4.6 but is deprecated. Use adaptive thinking above.

\`ContentBlock\` narrowing: \`.thinking()\` / \`.text()\` return \`Optional<T>\` — use \`.ifPresent(...)\` or \`.stream().flatMap(...)\`. Alternative: \`isThinking()\` / \`asThinking()\` boolean+unwrap pairs (throws on wrong variant).

---

## Tool Use (Beta)

The Java SDK supports beta tool use with annotated classes. Tool classes implement \`Supplier<String>\` for automatic execution via \`BetaToolRunner\`.

### Tool Runner (automatic loop)

\`\`\`java
import com.anthropic.models.beta.messages.MessageCreateParams;
import com.anthropic.models.beta.messages.BetaMessage;
import com.anthropic.helpers.BetaToolRunner;
import com.fasterxml.jackson.annotation.JsonClassDescription;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import java.util.function.Supplier;

\`\`\`

---

## Structured Output

The class-based overload auto-derives the JSON schema from your POJO and gives you a typed \`.text()\` return — no manual schema, no manual parsing.

\`\`\`java
import com.anthropic.models.messages.StructuredMessageCreateParams;

record Book(String title, String author) {}
record BookList(List<Book> books) {}

StructuredMessageCreateParams<BookList> params = MessageCreateParams.builder()
    .model(Model.CLAUDE_SONNET_4_6)
    .maxTokens(16000L)
    .outputConfig(BookList.class)  // returns a typed builder
    .addUserMessage("List 3 classic novels")
    .build();

client.messages().create(params).content().stream()
    .flatMap(cb -> cb.text().stream())
    .forEach(typed -> {
        // typed.text() returns BookList, not String
        for (Book b : typed.text().books()) System.out.println(b.title());
    });
\`\`\`

Supports Jackson annotations: \`@JsonPropertyDescription\`, \`@JsonIgnore\`, \`@ArraySchema(minItems=...)\`. Manual schema path: \`OutputConfig.builder().format(JsonOutputFormat.builder().schema(...).build())\`.

---

## PDF / Document Input

\`DocumentBlockParam\` builder has source shortcuts. Wrap in \`ContentBlockParam.ofDocument()\` and pass via \`.addUserMessageOfBlockParams()\`.

\`\`\`java
import com.anthropic.models.messages.DocumentBlockParam;
import com.anthropic.models.messages.ContentBlockParam;
import com.anthropic.models.messages.TextBlockParam;

DocumentBlockParam doc = DocumentBlockParam.builder()
    .base64Source(base64String)  // or .urlSource("https://...") or .textSource("...")
    .title("My Document")        // optional
    .build();

.addUserMessageOfBlockParams(List.of(
    ContentBlockParam.ofDocument(doc),
    ContentBlockParam.ofText(TextBlockParam.builder().text("Summarize this").build())))
\`\`\`

---

## Server-Side Tools

Version-suffixed types;

}
}
\`\`\`

If you only want the first text block:

\`\`\`php
foreach ($message->content as $block) {
    if ($block->type === 'text') {
        echo $block->text;
        break;
    }
}
\`\`\`

---

## Streaming

> **Requires SDK v0.5.0+.** v0.4.0 and earlier used a single \`$params\` array; calling with named parameters throws \`Unknown named parameter $model\`. Upgrade: \`composer require "anthropic-ai/sdk:^0.7"\`

\`\`\`php
use Anthropic\\Messages\\RawContentBlockDeltaEvent;
use Anthropic\\Messages\\TextDelta;

$stream = $client->messages->createStream(
    model: '{{OPUS_ID}}',
    maxTokens: 64000,
    messages: [
        ['role' => 'user', 'content' => 'Write a haiku'],
    ],
);

foreach ($stream as $event) {
    if ($event instanceof RawContentBlockDeltaEvent && $event->delta instanceof TextDelta) {
        echo $event->delta->text;
    }
}
\`\`\`

---

## Tool Use

### Tool Runner (Beta)

**Beta:** The PHP SDK provides a tool runner via \`$client->beta->messages->toolRunner()\`. Define tools with \`BetaRunnableTool\` — a definition array plus a \`run\` closure:

\`\`\`php
use Anthropic\\Lib\\Tools\\BetaRunnableTool;

$weatherTool = new BetaRunnableTool(
    definition: [
        'name' => 'get_weather',
        'description' => 'Get the current weather for a location.',
        'input_schema' => [
            'type' => 'object',
            'properties' => [
                'location' => ['type' => 'string', 'description' => 'City and state'],
            ],
            'required' => ['location'],
        ],
    ],
    run: function (array $input): string {
        return "The weather in {$input['location']} is sunny and 72°F.";
    },
);

$runner = $client->beta->messages->toolRunner(
    maxTokens: 16000,
    messages: [['role' => 'user', 'content' => 'What is the weather in Paris?']],
    model: '{{OPUS_ID}}',
    tools: [$weatherTool],
);

foreach ($runner as $message) {
    foreach ($message->content as $block) {
        if ($block->type === 'text') {
            echo $block->text;

\`\`\`

For 1-hour TTL: \`'cacheControl' => ['type' => 'ephemeral', 'ttl' => '1h']\`. There's also a top-level \`cacheControl:\` on \`messages->create(...)\` that auto-places on the last cacheable block.

Verify hits via \`$message->usage->cacheCreationInputTokens\` / \`$message->usage->cacheReadInputTokens\`.

---

## Structured Outputs

### Using StructuredOutputModel (Recommended)

Define a PHP class implementing \`StructuredOutputModel\` and pass it as \`outputConfig\`:

\`\`\`php
use Anthropic\\Lib\\Contracts\\StructuredOutputModel;
use Anthropic\\Lib\\Concerns\\StructuredOutputModelTrait;
use Anthropic\\Lib\\Attributes\\Constrained;

class Person implements StructuredOutputModel
{
    use StructuredOutputModelTrait;

    #[Constrained(description: 'Full name')]
    public string $name;

    public int $age;

    public ?string $email = null;  // nullable = optional field
}

$message = $client->messages->create(
    model: '{{OPUS_ID}}',
    maxTokens: 16000,
    messages: [['role' => 'user', 'content' => 'Generate a profile for Alice, age 30']],
    outputConfig: ['format' => Person::class],
);

$person = $message->parsedOutput();  // Person instance
echo $person->name;

\`\`\`

Types are inferred from PHP type hints. Use \`#[Constrained(description: '...')]\` to add descriptions. Nullable properties (\`?string\`) become optional fields.

### Raw Schema

\`\`\`php
$message = $client->messages->create(
    model: '{{OPUS_ID}}',
    maxTokens: 16000,
    messages: [['role' => 'user', 'content' => 'Extract: John (john@co.com), Enterprise plan']],
    outputConfig: [
        'format' => [
            'type' => 'json_schema',
            'schema' => [
                'type' => 'object',
                'properties' => [
                    'name' => ['type' => 'string'],
                    'email' => ['type' => 'string'],
                    'plan' => ['type' => 'string'],
                ],
                'required' => ['name', 'email', 'plan'],
                'additionalProperties' => false,
            ],
        ],
    ],
);

// First text block contains valid JSON
foreach ($message->content as $block) {
    if ($block->type === 'text') {
        $data = json_decode($block->text, true);
        break;
    }
}
\`\`\`

---

## Beta Features & Server-Side Tools

**\`betas:\` is NOT a param on \`$client->messages->create()\`** — it only exists on the beta namespace. Use it for features that need an explicit opt-in header:

\`\`\`php
use Anthropic\\Beta\\Messages\\BetaRequestMCPServerURLDefinition;

$response = $client->beta->messages->create(
    model: '{{OPUS_ID}}',
    maxTokens: 16000,
    mcpServers: [
        BetaRequestMCPServerURLDefinition::with(
            name: 'my-server',
            url: 'https://example.com/mcp',
        ),
    ],
    betas: ['mcp-client-2025-11-20'],  // only valid on ->beta->messages
    messages: [['role' => 'user', 'content' => 'Use the MCP tools']],
);

Claude reads the full file when the task calls for it. |

Both patterns keep the fixed context small and load detail on demand.

---

## Long-Running Agents: Managing Context

| Pattern | When to use it | What to expect |
| --- | --- | --- |
| **Context editing** | Context grows stale over many turns (old tool results, completed thinking). | Tool results and thinking blocks are cleared based on configurable thresholds. Keeps the transcript lean without summarizing. |
| **Compaction** | Conversation likely to reach or exceed the context window limit. | Earlier context is summarized into a compaction block server-side. See \`SKILL.md\` §Compaction for the critical \`response.content\` handling. |
| **Memory** | State must persist across sessions (not just within one conversation). | Claude reads/writes files in a memory directory. Survives process restarts. |

**Choosing between them:** Context editing and compaction operate within a session — editing prunes stale turns, compaction summarizes when you're near the limit. Memory is for cross-session persistence. Many long-running agents use all three.

---

## Caching for Agents

**Read \`prompt-caching.md\` first.** It covers the prefix-match invariant, breakpoint placement, the silent-invalidator audit, and why changing tools or models mid-session breaks the cache. This section covers only the agent-specific workarounds for those constraints.

| Constraint (from \`prompt-caching.md\`) | Agent-specific workaround |
| --- | --- |
| Editing the system prompt mid-session invalidates the cache. | Append a \`<system-reminder>\` block in the \`messages\` array instead. The cached prefix stays intact. Claude Code uses this for time updates and mode transitions. |
| Switching models mid-session invalidates the cache. | Spawn a **subagent** with the cheaper model for the sub-task;

adding, removing, or reordering a tool invalidates the entire cache. Same for switching models (caches are model-scoped). If you need "modes", don't swap the tool set — give Claude a tool that records the mode transition, or pass the mode as message content. Serialize tools deterministically (sort by name).

**Fork operations must reuse the parent's exact prefix.** Side computations (summarization, compaction, sub-agents) often spin up a separate API call. If the fork rebuilds \`system\` / \`tools\` / \`model\` with any difference, it misses the parent's cache entirely. Copy the parent's \`system\`, \`tools\`, and \`model\` verbatim, then append fork-specific content at the end.

---

## Silent invalidators

When reviewing code, grep for these inside anything that feeds the prompt prefix:

| Pattern | Why it breaks caching |
|---|---|
| \`datetime.now()\` / \`Date.now()\` / \`time.time()\` in system prompt | Prefix changes every request |
| \`uuid4()\` / \`crypto.randomUUID()\` / request IDs early in content | Same — every request is unique |
| \`json.dumps(d)\` without \`sort_keys=True\` / iterating a \`set\` | Non-deterministic serialization → prefix bytes differ |
| f-string interpolating session/user ID into system prompt | Per-user prefix; no cross-user sharing |
| Conditional system sections (\`if flag: system += ...\`) | Every flag combination is a distinct prefix |
| \`tools=build_tools(user)\` where set varies per user | Tools render at position 0;

try {
  const response = await client.messages.create({...});
} catch (error) {
  if (error instanceof Anthropic.BadRequestError) {
    console.error("Bad request:", error.message);
  } else if (error instanceof Anthropic.AuthenticationError) {
    console.error("Invalid API key");
  } else if (error instanceof Anthropic.RateLimitError) {
    console.error("Rate limited - retry later");
  } else if (error instanceof Anthropic.APIError) {
    console.error(\`API error \${error.status}:\`, error.message);
  }
}
\`\`\`

All classes extend \`Anthropic.APIError\` with a typed \`status\` field. Check from most specific to least specific. See [shared/error-codes.md](../../shared/error-codes.md) for the full error code reference.

---

## Multi-Turn Conversations

The API is stateless — send the full conversation history each time. Use \`Anthropic.MessageParam[]\` to type the messages array:

\`\`\`typescript
const messages: Anthropic.MessageParam[] = [
  { role: "user", content: "My name is Alice." },
  { role: "assistant", content: "Hello Alice! Nice to meet you." },
  { role: "user", content: "What's my name?" },
];

const response = await client.messages.create({
  model: "{{OPUS_ID}}",
  max_tokens: 16000,
  messages: messages,
});
\`\`\`

**Rules:**

- Consecutive same-role messages are allowed — the API combines them into a single turn
- First message must be \`user\`
- Use SDK types (\`Anthropic.MessageParam\`, \`Anthropic.Message\`, \`Anthropic.Tool\`, etc.) for all API data structures — don't redefine equivalent interfaces

---

### Compaction (long conversations)

> **Beta, Opus 4.6 and Sonnet 4.6.** When conversations approach the 200K context window, compaction automatically summarizes earlier context server-side. The API returns a \`compaction\` block; you must pass it back on subsequent requests — append \`response.content\`, not just the text.

\`\`\`typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const messages: Anthropic.Beta.BetaMessageParam[] = [];

async function chat(userMessage: string): Promise<string> {
  messages.push({ role: "user", content: userMessage });

  const response = await client.beta.messages.create({
    betas: ["compact-2026-01-12"],
    model: "{{OPUS_ID}}",
    max_tokens: 16000,
    messages,
    context_management: {
      edits: [{ type: "compact_20260112" }],
    },
  });

  // Append full content — compaction blocks must be preserved
  messages.push({ role: "assistant", content: response.content });

  const textBlock = response.content.find(
    (b): b is Anthropic.Beta.BetaTextBlock => b.type === "text",
  );
  return textBlock?.text ?? "";
}

// Compaction triggers automatically when context grows large
console.log(await chat("Help me build a Python web scraper"));
console.log(await chat("Add support for JavaScript-rendered pages"));
console.log(await chat("Now add rate limiting and error handling"))