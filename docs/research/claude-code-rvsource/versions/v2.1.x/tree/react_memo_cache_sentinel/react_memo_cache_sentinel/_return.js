// Module: _return

/* original: jJ7 */ var error_handler=L(()=>{
  AJ7();
  g$5=/[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g,F$5=/\\(\\)?/g,U$5=OJ7(function(q){var K=[];if(q.charCodeAt(0)===46)K.push("");return q.replace(g$5,function(_,z,Y,$){K.push(Y?$.replace(F$5,"$1"):z||_)}),K}),wJ7=U$5});function Q$5(q,K){var _=-1,z=q==null?0:q.length,Y=Array(z);while(++_<z)Y[_]=K(q[_],_,q);return Y}var we;var Qx6=L(()=>{we=Q$5});function MJ7(q){if(typeof q=="string")return q;if(EO(q))return we(q,MJ7)+"";if(Ae(q))return JJ7?JJ7.call(q):"";var K=q+"";return K=="0"&&1/q==-d$5?"-0":K}var d$5=1/0,HJ7,JJ7,XJ7;var PJ7=L(()=>{T96();Qx6();lv();Ux6();HJ7=pW?pW.prototype:void 0,JJ7=HJ7?HJ7.toString:void 0;XJ7=MJ7});function c$5(q){return q==null?"":XJ7(q)}var zP6;var L98=L(()=>{PJ7();zP6=c$5});function l$5(q,K){if(EO(q))return q;return _P6(q,K)?[q]:wJ7(zP6(q))}var MR;var E96=L(()=>{lv();E98();error_handler();L98();MR=l$5});function i$5(q){if(typeof q=="string"||Ae(q))return q;var K=q+"";return K=="0"&&1/q==-n$5?"-0":K}var n$5=1/0,XR;var L96=L(()=>{Ux6();XR=i$5});function r$5(q,K){K=MR(K,q);var _=0,z=K.length;while(q!=null&&_<z)q=q[XR(K[_++])];return _&&_==z?q:void 0}var je;var dx6=L(()=>{E96();L96();je=r$5});function o$5(q,K,_){var z=q==null?void 0:je(q,K);return z===void 0?_:z}var WJ7;var DJ7=L(()=>{dx6();WJ7=o$5});function a$5(q,K){return q!=null&&K in Object(q)}var fJ7;var ZJ7=L(()=>{fJ7=a$5});function s$5(q,K,_){K=MR(K,q);var z=-1,Y=K.length,$=!1;while(++z<Y){var O=XR(K[z]);if(!($=q!=null&&_(q,O)))break;q=q[O]}if($||++z!=Y)return $;return Y=q==null?0:q.length,!!Y&&aX6(Y)&&$e(O,Y)&&(EO(q)||jl(q))}var GJ7;var vJ7=L(()=>{E96();ux6();lv();px6();H98();L96();GJ7=s$5});function t$5(q,K){return q!=null&&GJ7(q,K,fJ7)}var TJ7;var kJ7=L(()=>{ZJ7();vJ7();TJ7=t$5});function KO5(q,K){if(_P6(q)&&N98(K))return y98(XR(q),K);return function(_){var z=WJ7(_,q);return z===void 0&&z===K?TJ7(_,q):KP6(K,z,e$5|qO5)}}var e$5=1,qO5=2,VJ7;var NJ7=L(()=>{V98();DJ7();kJ7();E98();rr8();or8();L96();VJ7=KO5});function _O5(q){return q}var YP6;var h98=L(()=>{YP6=_O5});function zO5(q){return function(K){return K==null?void 0:K[q]}}var yJ7;var EJ7=L(()=>{yJ7=zO5});function YO5(q){return function(K){return je(K,q)}}var LJ7;var hJ7=L(()=>{dx6();LJ7=YO5});function $O5(q){return _P6(q)?yJ7(XR(q)):LJ7(q)}var RJ7;var SJ7=L(()=>{EJ7();hJ7();E98();L96();RJ7=$O5});function OO5(q){if(typeof q=="function")return q;if(q==null)return YP6;if(typeof q=="object")return EO(q)?VJ7(q[0],q[1]):YJ7(q);return RJ7(q)}var PR;var h96=L(()=>{$J7();NJ7();h98();lv();SJ7();PR=OO5});function AO5(q,K){var _,z=-1,Y=q.length;while(++z<Y){var $=K(q[z]);if($!==void 0)_=_===void 0?$:_+$}return _}var CJ7;var bJ7=L(()=>{CJ7=AO5});function wO5(q,K){return q&&q.length?CJ7(q,PR(K,2)):0}var $P6;var xJ7=L(()=>{h96();bJ7();$P6=wO5});import{randomUUID as cx6}from"crypto";var sr8=()=>{};function IJ7(){return tr8}function uJ7(q){tr8=q}function mJ7(q){return R98.has(q)?R98.get(q):void 0}function pJ7(q,K){R98.set(q,K)}function BJ7(q){return er8.get(q)}function gJ7(q,K){er8.set(q,K)}function BW(){tr8=null,R98.clear(),er8.clear()}function S98(){return qo8}function FJ7(q){qo8=q}function UJ7(){qo8=void 0}var tr8=null,R98,er8,qo8;var Jl=L(()=>{R98=new Map;er8=new Map});function L_(){let q=new Set;return{subscribe(K){return q.add(K),()=>{q.delete(K)}},emit(...K){for(let _ of q)_(...K)},clear(){q.clear()}}}var e98={};v8(e98,{waitForScrollIdle:()=>sx6,updateLastInteractionTime:()=>S96,switchSession:()=>uf,snapshotOutputTokensForTurn:()=>vO5,setUserMsgOptIn:()=>CB,setUseCoworkPlugins:()=>vE,setTracerProvider:()=>Q98,setThinkingClearLatched:()=>ha8,setTeleportedSessionInfo:()=>XI6,setSystemPromptSectionCacheEntry:()=>fa8,setStrictToolResultPairing:()=>EO5,setStatsStore:()=>Wo8,setSessionTrustAccepted:()=>HI6,setSessionSource:()=>Qo8,setSessionPersistenceDisabled:()=>Aa8,setSessionIngressToken:()=>m96,setSessionBypassPermissionsMode:()=>$a8,setSdkBetas:()=>Ro8,setSdkAgentProgressSummariesEnabled:()=>go8,setScheduledTasksEnabled:()=>AI6,setQuestionPreviewFormat:()=>c98,setPromptId:()=>DI6,setPromptCache1hAllowlist:()=>Ta8,setProjectRoot:()=>nx6,setOriginalCwd:()=>WR,setOauthTokenFromFd:()=>io8,setNeedsPlanModeExitAttachment:()=>ex,setNeedsAutoModeExitAttachment:()=>s0,setModelStrings:()=>ex6,setMeterProvider:()=>U98,setMeter:()=>So8,setMemoryToggledOff:()=>Uo8,setMainThreadAgentType:()=>Pl,setMainLoopModelOverride:()=>yP,setLspRecommendationShownThisSession:()=>Ma8,setLoggerProvider:()=>g98,setLastMainRequestId:()=>No8,setLastEmittedDate:()=>DP6,setLastClassifierRequests:()=>zI6,setLastApiCompletionTimestamp:()=>ax6,setLastAPIRequestMessages:()=>to8,setLastAPIRequest:()=>ao8,setKairosActive:()=>yO5,setIsRemoteMode:()=>Wa8,setIsInteractive:()=>po8,setInlinePlugins:()=>za8,setInitialMainLoopModel:()=>ho8,setInitJsonSchema:()=>Xa8,setHasUnknownModelCost:()=>p98,setHasExitedPlanMode:()=>TE,setHasDevChannels:()=>t98,setFlagSettingsPath:()=>do8,setFlagSettingsInline:()=>co8,setFastModeHeaderLatched:()=>ya8,setEventLogger:()=>F98,setDirectConnectServerUrl:()=>HO5,setCwdState:()=>b98,setCostStateForRestore:()=>tx6,setClientType:()=>Bo8,setChromeFlagOverride:()=>Ya8,setCachedClaudeMdContent:()=>eo8,setCacheEditingHeaderLatched:()=>bO5,setApiKeyFromFd:()=>oo8,setAllowedSettingSources:()=>_a8,setAllowedChannels:()=>Wl,setAfkModeHeaderLatched:()=>Va8,setAdditionalDirectoriesForClaudeMd:()=>PI6,resetTurnToolDuration:()=>Mo8,resetTurnHookDuration:()=>Jo8,resetTurnClassifierDuration:()=>Po8,resetTotalDurationStateAndCost_FOR_TESTS_ONLY:()=>JO5,resetStateForTests:()=>rJ7,resetSdkInitState:()=>aJ7,resetModelStringsForTestingOnly:()=>VO5,resetCostState:()=>wP6,removeSessionCronTasks:()=>jI6,registerHookCallbacks:()=>Xe,regenerateSessionId:()=>_o8,preferThirdPartyAuthentication:()=>YI6,onSessionSwitch:()=>$o8,onInteraction:()=>Go8,needsPlanModeExitAttachment:()=>wa8,needsAutoModeExitAttachment:()=>ja8,markScrollActivity:()=>Eo8,markPostCompaction:()=>Je,markFirstTeleportMessageLogged:()=>o98,isSessionPersistenceDisabled:()=>jV,incrementBudgetContinuationCount:()=>kO5,hasUnknownModelCost:()=>Vo8,hasShownLspRecommendationThisSession:()=>Ja8,hasExitedPlanModeInSession:()=>JI6,handlePlanModeTransition:()=>Xl,handleAutoModeTransition:()=>Ha8,getUserMsgOptIn:()=>GE,getUseCoworkPlugins:()=>OI6,getUsageForModel:()=>Lo8,getTurnToolDurationMs:()=>PO5,getTurnToolCount:()=>WO5,getTurnOutputTokens:()=>ZO5,getTurnHookDurationMs:()=>MO5,getTurnHookCount:()=>XO5,getTurnClassifierDurationMs:()=>DO5,getTurnClassifierCount:()=>fO5,getTracerProvider:()=>u96,getTotalWebSearchRequests:()=>vo8,getTotalToolDuration:()=>Ho8,getTotalOutputTokens:()=>Ml,getTotalLinesRemoved:()=>b96,getTotalLinesAdded:()=>C96,getTotalInputTokens:()=>OV,getTotalDuration:()=>OP6,getTotalCostUSD:()=>sJ,getTotalCacheReadInputTokens:()=>ix6,getTotalCacheCreationInputTokens:()=>rx6,getTotalAPIDurationWithoutRetries:()=>jo8,getTotalAPIDuration:()=>a0,getTokenCounter:()=>HP6,getThinkingClearLatched:()=>La8,getTeleportedSessionInfo:()=>r98,getSystemPromptSectionCache:()=>Da8,getStrictToolResultPairing:()=>Fo8,getStatsStore:()=>He,getSlowOperations:()=>tJ7,getSessionTrustAccepted:()=>PP6,getSessionSource:()=>LO5,getSessionProjectDir:()=>R96,getSessionIngressToken:()=>lo8,getSessionId:()=>N8,getSessionCronTasks:()=>wI6,getSessionCreatedTeams:()=>MI6,getSessionCounter:()=>Co8,getSessionBypassPermissionsMode:()=>p96,getSdkBetas:()=>gW,getSdkAgentProgressSummariesEnabled:()=>RB,getScheduledTasksEnabled:()=>XP6,getRegisteredHooks:()=>fR,getQuestionPreviewFormat:()=>d98,getPromptId:()=>WI6,getPromptCache1hAllowlist:()=>va8,getProjectRoot:()=>iz,getPrCounter:()=>qI6,getPlanSlugCache:()=>B96,getParentSessionId:()=>zo8,getOriginalCwd:()=>z7,getOauthTokenFromFd:()=>no8,getModelUsage:()=>AV,getModelStrings:()=>jP6,getMeterProvider:()=>mo8,getMeter:()=>NO5,getMemoryToggledOff:()=>SB,getMainThreadAgentType:()=>xB,getMainLoopModelOverride:()=>tx,getLoggerProvider:()=>_I6,getLocCounter:()=>B98,getLastMainRequestId:()=>ox6,getLastInteractionTime:()=>DR,getLastEmittedDate:()=>Ga8,getLastClassifierRequests:()=>oJ7,getLastApiCompletionTimestamp:()=>x96,getLastAPIRequestMessages:()=>hO5,getLastAPIRequest:()=>so8,getKairosActive:()=>wV,getIsScrollDraining:()=>I96,getIsRemoteMode:()=>_5,getIsNonInteractiveSession:()=>g7,getIsInteractive:()=>nv,getInvokedSkillsForAgent:()=>a98,getInvokedSkills:()=>CO5,getInlinePlugins:()=>bB,getInitialMainLoopModel:()=>AP6,getInitJsonSchema:()=>n98,getHasDevChannels:()=>s98,getFlagSettingsPath:()=>Me,getFlagSettingsInline:()=>MP6,getFastModeHeaderLatched:()=>Na8,getEventLogger:()=>uo8,getDirectConnectServerUrl:()=>Oo8,getCwdState:()=>sx,getCurrentTurnTokenBudget:()=>GO5,getCostCounter:()=>xo8,getCommitCounter:()=>bo8,getCodeEditToolDecisionCounter:()=>KI6,getClientType:()=>JP6,getChromeFlagOverride:()=>$I6,getCachedClaudeMdContent:()=>qa8,getCacheEditingHeaderLatched:()=>Ea8,getBudgetContinuationCount:()=>TO5,getApiKeyFromFd:()=>ro8,getAllowedSettingSources:()=>Ka8,getAllowedChannels:()=>wJ,getAgentColorMap:()=>l98,getAfkModeHeaderLatched:()=>ka8,getAdditionalDirectoriesForClaudeMd:()=>t0,getActiveTimeCounter:()=>Io8,flushInteractionTime:()=>fo8,consumePostCompaction:()=>yo8,clearSystemPromptSectionState:()=>Za8,clearRegisteredPluginHooks:()=>i98,clearRegisteredHooks:()=>SO5,clearInvokedSkillsForAgent:()=>Pe,clearInvokedSkills:()=>Pa8,clearBetaHeaderLatches:()=>Ra8,addToTurnHookDuration:()=>I98,addToTurnClassifierDuration:()=>Xo8,addToTotalLinesChanged:()=>u98,addToTotalDurationState:()=>Ao8,addToTotalCostState:()=>wo8,addToToolDuration:()=>x98,addToInMemoryErrorLog:()=>RO5,addSlowOperation:()=>sJ7,addSessionCronTask:()=>Oa8,addInvokedSkill:()=>WP6});import{realpathSync as QJ7}from"fs";import{cwd as jO5}from"process";function lJ7(){let q="";if(typeof process<"u"&&typeof process.cwd==="function"&&typeof QJ7==="function"){let _=jO5();try{q=QJ7(_).normalize("NFC")}catch{q=_.normalize("NFC")}}return{originalCwd:q,projectRoot:q,totalCostUSD:0,totalAPIDuration:0,totalAPIDurationWithoutRetries:0,totalToolDuration:0,turnHookDurationMs:0,turnToolDurationMs:0,turnClassifierDurationMs:0,turnToolCount:0,turnHookCount:0,turnClassifierCount:0,startTime:Date.now(),lastInteractionTime:Date.now(),totalLinesAdded:0,totalLinesRemoved:0,hasUnknownModelCost:!1,cwd:q,modelUsage:{},mainLoopModelOverride:void 0,initialMainLoopModel:null,modelStrings:null,isInteractive:!1,kairosActive:!1,strictToolResultPairing:!1,memoryToggledOff:!1,sdkAgentProgressSummariesEnabled:!1,userMsgOptIn:!1,clientType:"cli",sessionSource:void 0,questionPreviewFormat:void 0,sessionIngressToken:void 0,oauthTokenFromFd:void 0,apiKeyFromFd:void 0,flagSettingsPath:void 0,flagSettingsInline:null,allowedSettingSources:["userSettings","projectSettings","localSettings","flagSettings","policySettings"],meter:null,sessionCounter:null,locCounter:null,prCounter:null,commitCounter:null,costCounter:null,tokenCounter:null,codeEditToolDecisionCounter:null,activeTimeCounter:null,statsStore:null,sessionId:cx6(),parentSessionId:void 0,loggerProvider:null,eventLogger:null,meterProvider:null,tracerProvider:null,agentColorMap:new Map,agentColorIndex:0,lastAPIRequest:null,lastAPIRequestMessages:null,lastClassifierRequests:null,cachedClaudeMdContent:null,inMemoryErrorLog:[],inlinePlugins:[],chromeFlagOverride:void 0,useCoworkPlugins:!1,sessionBypassPermissionsMode:!1,scheduledTasksEnabled:!1,sessionCronTasks:[],sessionCreatedTeams:new Set,sessionTrustAccepted:!1,sessionPersistenceDisabled:!1,hasExitedPlanMode:!1,needsPlanModeExitAttachment:!1,needsAutoModeExitAttachment:!1,lspRecommendationShownThisSession:!1,initJsonSchema:null,registeredHooks:null,planSlugCache:new Map,teleportedSessionInfo:null,invokedSkills:new Map,slowOperations:[],sdkBetas:void 0,mainThreadAgentType:void 0,isRemoteMode:!1,...!1,directConnectServerUrl:void 0,systemPromptSectionCache:new Map,lastEmittedDate:null,additionalDirectoriesForClaudeMd:[],allowedChannels:[],hasDevChannels:!1,sessionProjectDir:null,promptCache1hAllowlist:null,afkModeHeaderLatched:null,fastModeHeaderLatched:null,cacheEditingHeaderLatched:null,thinkingClearLatched:null,promptId:null,lastMainRequestId:void 0,lastApiCompletionTimestamp:null,pendingPostCompaction:!1}}function N8(){return G8.sessionId}function _o8(q={}){if(q.setCurrentAsParent)G8.parentSessionId=G8.sessionId;return G8.planSlugCache.delete(G8.sessionId),G8.sessionId=cx6(),G8.sessionProjectDir=null,G8.sessionId}function zo8(){return G8.parentSessionId}function uf(q,K=null){if(G8.sessionId!==q)G8.planSlugCache.delete(G8.sessionId);G8.sessionId=q,G8.sessionProjectDir=K,Yo8.emit(q)}function R96(){return G8.sessionProjectDir}function z7(){return G8.originalCwd}function iz(){return G8.projectRoot}function WR(q){G8.originalCwd=q.normalize("NFC")}function nx6(q){G8.projectRoot=q.normalize("NFC")}function sx(){return G8.cwd}function b98(q){G8.cwd=q.normalize("NFC")}function Oo8(){return G8.directConnectServerUrl}function HO5(q){G8.directConnectServerUrl=q}function Ao8(q,K){G8.totalAPIDuration+=q,G8.totalAPIDurationWithoutRetries+=K}function JO5(){G8.totalAPIDuration=0,G8.totalAPIDurationWithoutRetries=0,G8.totalCostUSD=0}function wo8(q,K,_){G8.modelUsage[_]=K,G8.totalCostUSD+=q}function sJ(){return G8.totalCostUSD}function a0(){return G8.totalAPIDuration}function OP6(){return Date.now()-G8.startTime}function jo8(){return G8.totalAPIDurationWithoutRetries}function Ho8(){return G8.totalToolDuration}function x98(q){G8.totalToolDuration+=q,G8.turnToolDurationMs+=q,G8.turnToolCount++}function MO5(){return G8.turnHookDurationMs}function I98(q){G8.turnHookDurationMs+=q,G8.turnHookCount++}function Jo8(){G8.turnHookDurationMs=0,G8.turnHookCount=0}function XO5(){return G8.turnHookCount}function PO5(){return G8.turnToolDurationMs}function Mo8(){G8.turnToolDurationMs=0,G8.turnToolCount=0}function WO5(){return G8.turnToolCount}function DO5(){return G8.turnClassifierDurationMs}function Xo8(q){G8.turnClassifierDurationMs+=q,G8.turnClassifierCount++}function Po8(){G8.turnClassifierDurationMs=0,G8.turnClassifierCount=0}function fO5(){return G8.turnClassifierCount}function He(){return G8.statsStore}function Wo8(q){G8.statsStore=q}function S96(q){if(q)nJ7();else Do8=!0}function fo8(){if(Do8)nJ7()}function nJ7(){G8.lastInteractionTime=Date.now(),Do8=!1,Zo8.emit()}function u98(q,K){G8.totalLinesAdded+=q,G8.totalLinesRemoved+=K}function C96(){return G8.totalLinesAdded}function b96(){return G8.totalLinesRemoved}function OV(){return $P6(Object.values(G8.modelUsage),"inputTokens")}function Ml(){return $P6(Object.values(G8.modelUsage),"outputTokens")}function ix6(){return $P6(Object.values(G8.modelUsage),"cacheReadInputTokens")}function rx6(){return $P6(Object.values(G8.modelUsage),"cacheCreationInputTokens")}function vo8(){return $P6(Object.values(G8.modelUsage),"webSearchRequests")}function ZO5(){return Ml()-To8}function GO5(){return ko8}function vO5(q){To8=Ml(),ko8=q,m98=0}function TO5(){return m98}function kO5(){m98++}function p98(){G8.hasUnknownModelCost=!0}function Vo8(){return G8.hasUnknownModelCost}function ox6(){return G8.lastMainRequestId}function No8(q){G8.lastMainRequestId=q}function x96(){return G8.lastApiCompletionTimestamp}function ax6(q){G8.lastApiCompletionTimestamp=q}function Je(){G8.pendingPostCompaction=!0}function yo8(){let q=G8.pendingPostCompaction;return G8.pendingPostCompaction=!1,q}function DR(){return G8.lastInteractionTime}function Eo8(){if(C98=!0,lx6)clearTimeout(lx6);lx6=setTimeout(()=>{C98=!1,lx6=void 0},iJ7),lx6.unref?.()}function I96(){return C98}async function sx6(){while(C98)await new Promise((q)=>setTimeout(q,iJ7).unref?.())}function AV(){return G8.modelUsage}function Lo8(q){return G8.modelUsage[q]}function tx(){return G8.mainLoopModelOverride}function AP6(){return G8.initialMainLoopModel}function yP(q){G8.mainLoopModelOverride=q}function ho8(q){G8.initialMainLoopModel=q}function gW(){return G8.sdkBetas}function Ro8(q){G8.sdkBetas=q}function wP6(){G8.totalCostUSD=0,G8.totalAPIDuration=0,G8.totalAPIDurationWithoutRetries=0,G8.totalToolDuration=0,G8.startTime=Date.now(),G8.totalLinesAdded=0,G8.totalLinesRemoved=0,G8.hasUnknownModelCost=!1,G8.modelUsage={},G8.promptId=null}function tx6({totalCostUSD:q,totalAPIDuration:K,totalAPIDurationWithoutRetries:_,totalToolDuration:z,totalLinesAdded:Y,totalLinesRemoved:$,lastDuration:O,modelUsage:A}){if(G8.totalCostUSD=q,G8.totalAPIDuration=K,G8.totalAPIDurationWithoutRetries=_,G8.totalToolDuration=z,G8.totalLinesAdded=Y,G8.totalLinesRemoved=$,A)G8.modelUsage=A;if(O)G8.startTime=Date.now()-O}function rJ7(){throw Error("resetStateForTests can only be called in tests")}function jP6(){return G8.modelStrings}function ex6(q){G8.modelStrings=q}function VO5(){G8.modelStrings=null}function So8(q,K){G8.meter=q,G8.sessionCounter=K("claude_code.session.count",{description:"Count of CLI sessions started"}),G8.locCounter=K("claude_code.lines_of_code.count",{description:"Count of lines of code modified, with the 'type' attribute indicating whether lines were added or removed"}),G8.prCounter=K("claude_code.pull_request.count",{description:"Number of pull requests created"}),G8.commitCounter=K("claude_code.commit.count",{description:"Number of git commits created"}),G8.costCounter=K("claude_code.cost.usage",{description:"Cost of the Claude Code session",unit:"USD"}),G8.tokenCounter=K("claude_code.token.usage",{description:"Number of tokens used",unit:"tokens"}),G8.codeEditToolDecisionCounter=K("claude_code.code_edit_tool.decision",{description:"Count of code editing tool permission decisions (accept/reject) for Edit, Write, and NotebookEdit tools"}),G8.activeTimeCounter=K("claude_code.active_time.total",{description:"Total active time in seconds",unit:"s"})}function NO5(){return G8.meter}function Co8(){return G8.sessionCounter}function B98(){return G8.locCounter}function qI6(){return G8.prCounter}function bo8(){return G8.commitCounter}function xo8(){return G8.costCounter}function HP6(){return G8.tokenCounter}function KI6(){return G8.codeEditToolDecisionCounter}function Io8(){return G8.activeTimeCounter}function _I6(){return G8.loggerProvider}function g98(q){G8.loggerProvider=q}function uo8(){return G8.eventLogger}function F98(q){G8.eventLogger=q}function mo8(){return G8.meterProvider}function U98(q){G8.meterProvider=q}function u96(){return G8.tracerProvider}function Q98(q){G8.tracerProvider=q}function g7(){return!G8.isInteractive}function nv(){return G8.isInteractive}function po8(q){G8.isInteractive=q}function JP6(){return G8.clientType}function Bo8(q){G8.clientType=q}function RB(){return G8.sdkAgentProgressSummariesEnabled}function go8(q){G8.sdkAgentProgressSummariesEnabled=q}function wV(){return G8.kairosActive}function yO5(q){G8.kairosActive=q}function Fo8(){return G8.strictToolResultPairing}function EO5(q){G8.strictToolResultPairing=q}function SB(){return G8.memoryToggledOff}function Uo8(q){G8.memoryToggledOff=q}function GE(){return G8.userMsgOptIn}function CB(q){G8.userMsgOptIn=q}function LO5(){return G8.sessionSource}function Qo8(q){G8.sessionSource=q}function d98(){return G8.questionPreviewFormat}function c98(q){G8.questionPreviewFormat=q}function l98(){return G8.agentColorMap}function Me(){return G8.flagSettingsPath}function do8(q){G8.flagSettingsPath=q}function MP6(){return G8.flagSettingsInline}function co8(q){G8.flagSettingsInline=q}function lo8(){return G8.sessionIngressToken}function m96(q){G8.sessionIngressToken=q}function no8(){return G8.oauthTokenFromFd}function io8(q){G8.oauthTokenFromFd=q}function ro8(){return G8.apiKeyFromFd}function oo8(q){G8.apiKeyFromFd=q}function ao8(q){G8.lastAPIRequest=q}function so8(){return G8.lastAPIRequest}function to8(q){G8.lastAPIRequestMessages=q}function hO5(){return G8.lastAPIRequestMessages}function zI6(q){G8.lastClassifierRequests=q}function oJ7(){return G8.lastClassifierRequests}function eo8(q){G8.cachedClaudeMdContent=q}function qa8(){return G8.cachedClaudeMdContent}function RO5(q){if(G8.inMemoryErrorLog.length>=100)G8.inMemoryErrorLog.shift();G8.inMemoryErrorLog.push(q)}function Ka8(){return G8.allowedSettingSources}function _a8(q){G8.allowedSettingSources=q}function YI6(){return g7()&&G8.clientType!=="claude-vscode"}function za8(q){G8.inlinePlugins=q}function bB(){return G8.inlinePlugins}function Ya8(q){G8.chromeFlagOverride=q}function $I6(){return G8.chromeFlagOverride}function vE(q){G8.useCoworkPlugins=q,BW()}function OI6(){return G8.useCoworkPlugins}function $a8(q){G8.sessionBypassPermissionsMode=q}function p96(){return G8.sessionBypassPermissionsMode}function AI6(q){G8.scheduledTasksEnabled=q}function XP6(){return G8.scheduledTasksEnabled}function wI6(){return G8.sessionCronTasks}function Oa8(q){G8.sessionCronTasks.push(q)}function jI6(q){if(q.length===0)return 0;let K=new Set(q),_=G8.sessionCronTasks.filter((Y)=>!K.has(Y.id)),z=G8.sessionCronTasks.length-_.length;if(z===0)return 0;return G8.sessionCronTasks=_,z}function HI6(q){G8.sessionTrustAccepted=q}function PP6(){return G8.sessionTrustAccepted}function Aa8(q){G8.sessionPersistenceDisabled=q}function jV(){return G8.sessionPersistenceDisabled}function JI6(){return G8.hasExitedPlanMode}function TE(q){G8.hasExitedPlanMode=q}function wa8(){return G8.needsPlanModeExitAttachment}function ex(q){G8.needsPlanModeExitAttachment=q}function Xl(q,K){if(K==="plan"&&q!=="plan")G8.needsPlanModeExitAttachment=!1;if(q==="plan"&&K!=="plan")G8.needsPlanModeExitAttachment=!0}function ja8(){return G8.needsAutoModeExitAttachment}function s0(q){G8.needsAutoModeExitAttachment=q}function Ha8(q,K){if(q==="auto"&&K==="plan"||q==="plan"&&K==="auto")return;let _=q==="auto",z=K==="auto";if(z&&!_)G8.needsAutoModeExitAttachment=!1;if(_&&!z)G8.needsAutoModeExitAttachment=!0}function Ja8(){return G8.lspRecommendationShownThisSession}function Ma8(q){G8.lspRecommendationShownThisSession=q}function Xa8(q){G8.initJsonSchema=q}function n98(){return G8.initJsonSchema}function Xe(q){if(!G8.registeredHooks)G8.registeredHooks={};for(let[K,_]of Object.entries(q)){let z=K;if(!G8.registeredHooks[z])G8.registeredHooks[z]=[];G8.registeredHooks[z].push(..._)}}function fR(){return G8.registeredHooks}function SO5(){G8.registeredHooks=null}function i98(){if(!G8.registeredHooks)return;let q={};for(let[K,_]of Object.entries(G8.registeredHooks)){let z=_.filter((Y)=>!("pluginRoot"in Y));if(z.length>0)q[K]=z}G8.registeredHooks=Object.keys(q).length>0?q:null}function aJ7(){G8.initJsonSchema=null,G8.registeredHooks=null}function B96(){return G8.planSlugCache}function MI6(){return G8.sessionCreatedTeams}function XI6(q){G8.teleportedSessionInfo={isTeleported:!0,hasLoggedFirstMessage:!1,sessionId:q.sessionId}}function r98(){return G8.teleportedSessionInfo}function o98(){if(G8.teleportedSessionInfo)G8.teleportedSessionInfo.hasLoggedFirstMessage=!0}function WP6(q,K,_,z=null){let Y=`${z??""}:${q}`;G8.invokedSkills.set(Y,{skillName:q,skillPath:K,content:_,invokedAt:Date.now(),agentId:z})}function CO5(){return G8.invokedSkills}function a98(q){let K=q??null,_=new Map;for(let[z,Y]of G8.invokedSkills)if(Y.agentId===K)_.set(z,Y);return _}function Pa8(q){if(!q||q.size===0){G8.invokedSkills.clear();return}for(let[K,_]of G8.invokedSkills)if(_.agentId===null||!q.has(_.agentId))G8.invokedSkills.delete(K)}function Pe(q){for(let[K,_]of G8.invokedSkills)if(_.agentId===q)G8.invokedSkills.delete(K)}function sJ7(q,K){return}function tJ7(){if(G8.slowOperations.length===0)return cJ7;let q=Date.now();if(G8.slowOperations.some((K)=>q-K.timestamp>=Ko8)){if(G8.slowOperations=G8.slowOperations.filter((K)=>q-K.timestamp<Ko8),G8.slowOperations.length===0)return cJ7}return G8.slowOperations}function xB(){return G8.mainThreadAgentType}function Pl(q){G8.mainThreadAgentType=q}function _5(){return G8.isRemoteMode}function Wa8(q){G8.isRemoteMode=q}function Da8(){return G8.systemPromptSectionCache}function fa8(q,K){G8.systemPromptSectionCache.set(q,K)}function Za8(){G8.systemPromptSectionCache.clear()}function Ga8(){return G8.lastEmittedDate}function DP6(q){G8.lastEmittedDate=q}function t0(){return G8.additionalDirectoriesForClaudeMd}function PI6(q){G8.additionalDirectoriesForClaudeMd=q}function wJ(){return G8.allowedChannels}function Wl(q){G8.allowedChannels=q}function s98(){return G8.hasDevChannels}function t98(q){G8.hasDevChannels=q}function va8(){return G8.promptCache1hAllowlist}function Ta8(q){G8.promptCache1hAllowlist=q}function ka8(){return G8.afkModeHeaderLatched}function Va8(q){G8.afkModeHeaderLatched=q}function Na8(){return G8.fastModeHeaderLatched}function ya8(q){G8.fastModeHeaderLatched=q}function Ea8(){return G8.cacheEditingHeaderLatched}function bO5(q){G8.cacheEditingHeaderLatched=q}function La8(){return G8.thinkingClearLatched}function ha8(q){G8.thinkingClearLatched=q}function Ra8(){G8.afkModeHeaderLatched=null,G8.fastModeHeaderLatched=null,G8.cacheEditingHeaderLatched=null,G8.thinkingClearLatched=null}function WI6(){return G8.promptId}function DI6(q){G8.promptId=q}var G8,Yo8,$o8,Do8=!1,Zo8,Go8,To8=0,ko8=null,m98=0,C98=!1,lx6,iJ7=150,dJ7=10,Ko8=1e4,cJ7;var T8=L(()=>{xJ7();sr8();Jl();G8=lJ7();Yo8=L_(),$o8=Yo8.subscribe;Zo8=L_(),Go8=Zo8.subscribe;cJ7=[]});function q_8(q){let K;for(let _ in q)if(_.startsWith("_PROTO_")){if(K===void 0)K={...q};delete K[_]}return K??q}function eJ7(q){if(We!==null)return;if(We=q,fI6.length>0){let K=[...fI6];fI6.length=0,queueMicrotask(()=>{for(let _ of K)if(_.async)We.logEventAsync(_.eventName,_.metadata);else We.logEvent(_.eventName,_.metadata)})}}function d(q,K){if(We===null){fI6.push({eventName:q,metadata:K,async:!1});return}We.logEvent(q,K)}async function qM7(q,K){if(We===null){fI6.push({eventName:q,metadata:K,async:!0});return}await We.logEventAsync(q,K)}var fI6,We=null;var k8=L(()=>{fI6=[]});function fP6({writeFn:q,flushIntervalMs:K=1000,maxBufferSize:_=100,maxBufferBytes:z=1/0,immediateMode:Y=!1}){let $=[],O=0,A=null,w=null;function j(){if(A)clearTimeout(A),A=null}function H(){if(w)q(w.join("")),w=null;if($.length===0)return;q($.join("")),$=[],O=0,j()}function J(){if(!A)A=setTimeout(H,K)}function M(){if(w){w.push(...$),$=[],O=0,j();return}let X=$;$=[],O=0,j(),w=X,setImmediate(()=>{let P=w;if(w=null,P)q(P.join(""))})}return{write(X){if(Y){q(X);return}if($.push(X),O+=X.length,J(),$.length>=_||O>=z)M()},flush:H,dispose(){H()}}}function gq(q){return Sa8.add(q),()=>Sa8.delete(q)}async function KM7(){await Promise.all(Array.from(Sa8).map((q)=>q()))}var Sa8;var R9=L(()=>{Sa8=new Set});function xO5(q){let K=[],_=q.match(/^MCP server ["']([^"']+)["']/);if(_&&_[1])K.push("mcp"),K.push(_[1].toLowerCase());else{let $=q.match(/^([^:[]+):/);if($&&$[1])K.push($[1].trim().toLowerCase())}let z=q.match(/^\[([^\]]+)]/);if(z&&z[1])K.push(z[1].trim().toLowerCase());if(q.toLowerCase().includes("1p event:"))K.push("1p");let Y=q.match(/:\s*([^:]+?)(?:\s+(?:type|mode|status|event))?:/);if(Y&&Y[1]){let $=Y[1].trim().toLowerCase();if($.length<30&&!$.includes(" "))K.push($)}return Array.from(new Set(K))}function IO5(q,K){if(!K)return!0;if(q.length===0)return!1;if(K.isExclusive)return!q.some((_)=>K.exclude.includes(_));else return q.some((_)=>K.include.includes(_))}function zM7(q,K){if(!K)return!0;let _=xO5(q);return IO5(_,K)}var _M7;var YM7=L(()=>{c4();_M7=$1((q)=>{if(!q||q.trim()==="")return null;let K=q.split(",").map(($)=>$.trim()).filter(Boolean);if(K.length===0)return null;let _=K.some(($)=>$.startsWith("!")),z=K.some(($)=>!$.startsWith("!"));if(_&&z)return null;let Y=K.map(($)=>$.replace(/^!/,"").toLowerCase());return{include:_?[]:Y,exclude:_?Y:[],isExclusive:_}})});import{homedir as uO5}from"os";import{join as $M7}from"path";function ZP6(){return $M7(q7(),"teams")}function GP6(q){let K=process.env.NODE_OPTIONS;if(!K)return!1;return K.split(/\s+/).includes(q)}function c6(q){if(!q)return!1;if(typeof q==="boolean")return q;let K=q.toLowerCase().trim();return["1","true","yes","on"].includes(K)}function d_(q){if(q===void 0)return!1;if(typeof q==="boolean")return!q;if(!q)return!1;let K=q.toLowerCase().trim();return["0","false","no","off"].includes(K)}function f9(){return c6(process.env.CLAUDE_CODE_SIMPLE)||process.argv.includes("--bare")}function OM7(q){let K={};if(q)for(let _ of q){let[z,...Y]=_.split("=");if(!z||Y.length===0)throw Error(`Invalid environment variable format: ${_}, environment variables should be added as: -e KEY1=value1 -e KEY2=value2`);K[z]=Y.join("=")}return K}function De(){return process.env.AWS_REGION||process.env.AWS_DEFAULT_REGION||"us-east-1"}function K_8(){return process.env.CLOUD_ML_REGION||"us-east5"}function AM7(){return c6(process.env.CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR)}function iv(){return!1}function HV(){return!1}function wM7(){return{namespace:void 0,cluster:void 0}}function __8(q){if(q){let K=mO5.find(([_])=>q.startsWith(_));if(K)return process.env[K[1]]||K_8()}return K_8()}var q7,mO5;var d8=L(()=>{c4();q7=$1(()=>{return(process.env.CLAUDE_CONFIG_DIR??$M7(uO5(),".claude")).normalize("NFC")},()=>process.env.CLAUDE_CONFIG_DIR);mO5=[["claude-haiku-4-5","VERTEX_REGION_CLAUDE_HAIKU_4_5"],["claude-3-5-haiku","VERTEX_REGION_CLAUDE_3_5_HAIKU"],["claude-3-5-sonnet","VERTEX_REGION_CLAUDE_3_5_SONNET"],["claude-3-7-sonnet","VERTEX_REGION_CLAUDE_3_7_SONNET"],["claude-opus-4-1","VERTEX_REGION_CLAUDE_4_1_OPUS"],["claude-opus-4","VERTEX_REGION_CLAUDE_4_0_OPUS"],["claude-sonnet-4-6","VERTEX_REGION_CLAUDE_4_6_SONNET"],["claude-sonnet-4-5","VERTEX_REGION_CLAUDE_4_5_SONNET"],["claude-sonnet-4","VERTEX_REGION_CLAUDE_4_0_SONNET"]]});function J4(q,K,_,z,Y){if(z==="m")throw TypeError("Private method is not writable");if(z==="a"&&!Y)throw TypeError("Private accessor was defined without a setter");if(typeof K==="function"?q!==K||!Y:!K.has(q))throw TypeError("Cannot write private member to an object whose class did not declare it");return z==="a"?Y.call(q,_):Y?Y.value=_:K.set(q,_),_}function x1(q,K,_,z){if(_==="a"&&!z)throw TypeError("Private accessor was defined without a getter");if(typeof K==="function"?q!==K||!z:!K.has(q))throw TypeError("Cannot read private member from an object whose class did not declare it");return _==="m"?z:_==="a"?z.call(q):z?z.value:K.get(q)}var Dl=()=>{};var Ca8=function(){let{crypto:q}=globalThis;if(q?.randomUUID)return Ca8=q.randomUUID.bind(q),q.randomUUID();let K=new Uint8Array(1),_=q?()=>q.getRandomValues(K)[0]:()=>Math.random()*255&255;return"10000000-1000-4000-8000-100000000000".replace(/[018]/g,(z)=>(+z^_()&15>>+z/4).toString(16))};function fl(q){return typeof q==="object"&&q!==null&&(("name"in q)&&q.name==="AbortError"||("message"in q)&&String(q.message).includes("FetchRequestCanceledException"))}var ZI6=(q)=>{if(q instanceof Error)return q;if(typeof q==="object"&&q!==null){try{if(Object.prototype.toString.call(q)==="[object Error]"){let K=Error(q.message,q.cause?{cause:q.cause}:{});if(q.stack)K.stack=q.stack;if(q.cause&&!K.cause)K.cause=q.cause;if(q.name)K.name=q.name;return K}}catch{}try{return Error(JSON.stringify(q))}catch{}}return Error(q)};var mq,nq,c_,mf,IB,GI6,g96,vI6,F96,TI6,kI6,VI6,NI6;var FW=L(()=>{mq=class mq extends Error{};nq=class nq extends mq{constructor(q,K,_,z){super(`${nq.makeMessage(q,K,_)}`);this.status=q,this.headers=z,this.requestID=z?.get("request-id"),this.error=K}static makeMessage(q,K,_){let z=K?.message?typeof K.message==="string"?K.message:JSON.stringify(K.message):K?JSON.stringify(K):_;if(q&&z)return`${q} ${z}`;if(q)return`${q} status code (no body)`;if(z)return z;return"(no status code or body)"}static generate(q,K,_,z){if(!q||!z)return new mf({message:_,cause:ZI6(K)});let Y=K;if(q===400)return new GI6(q,Y,_,z);if(q===401)return new g96(q,Y,_,z);if(q===403)return new vI6(q,Y,_,z);if(q===404)return new F96(q,Y,_,z);if(q===409)return new TI6(q,Y,_,z);if(q===422)return new kI6(q,Y,_,z);if(q===429)return new VI6(q,Y,_,z);if(q>=500)return new NI6(q,Y,_,z);return new nq(q,Y,_,z)}};c_=class c_ extends nq{constructor({message:q}={}){super(void 0,void 0,q||"Request was aborted.",void 0)}};mf=class mf extends nq{constructor({message:q,cause:K}){super(void 0,void 0,q||"Connection error.",void 0);if(K)this.cause=K}};IB=class IB extends mf{constructor({message:q}={}){super({message:q??"Request timed out."})}};GI6=class GI6 extends nq{};g96=class g96 extends nq{};vI6=class vI6 extends nq{};F96=class F96 extends nq{};TI6=class TI6 extends nq{};kI6=class kI6 extends nq{};VI6=class VI6 extends nq{};NI6=class NI6 extends nq{}});function z_8(q){if(typeof q!=="object")return{};return q??{}}function Ia8(q){if(!q)return!0;for(let K in q)return!1;return!0}function HM7(q,K){return Object.prototype.hasOwnProperty.call(q,K)}var BO5,jM7=(q)=>{return BO5.test(q)},ba8=(q)=>(ba8=Array.isArray,ba8(q)),xa8,JM7=(q,K)=>{if(typeof K!=="number"||!Number.isInteger(K))throw new mq(`${q} must be an integer`);if(K<0)throw new mq(`${q} must be a positive integer`);return K},Y_8=(q)=>{try{return JSON.parse(q)}catch(K){return}};var U96=L(()=>{FW();BO5=/^[a-z][a-z0-9+.-]*:/i,xa8=ba8});var MM7=(q)=>new Promise((K)=>setTimeout(K,q));var fe="0.80.0";function gO5(){if(typeof Deno<"u"&&Deno.build!=null)return"deno";if(typeof EdgeRuntime<"u")return"edge";if(Object.prototype.toString.call(typeof globalThis.process<"u"?globalThis.process:0)==="[object process]")return"node";return"unknown"}function UO5(){if(typeof navigator>"u"||!navigator)return null;let q=[{key:"edge",pattern:/Edge(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"ie",pattern:/MSIE(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"ie",pattern:/Trident(?:.*rv\:(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"chrome",pattern:/Chrome(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"firefox",pattern:/Firefox(?:\W+(\d+)\.(\d+)(?:\.(\d+))?)?/},{key:"safari",pattern:/(?:Version\W+(\d+)\.(\d+)(?:\.(\d+))?)?(?:\W+Mobile\S*)?\W+Safari/}];for(let{key:K,pattern:_}of q){let z=_.exec(navigator.userAgent);if(z){let Y=z[1]||0,$=z[2]||0,O=z[3]||0;return{browser:K,version:`${Y}.${$}.${O}`}}}return null}var DM7=()=>{return typeof window<"u"&&typeof window.document<"u"&&typeof navigator<"u"},FO5=()=>{let q=gO5();if(q==="deno")return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fe,"X-Stainless-OS":PM7(Deno.build.os),"X-Stainless-Arch":XM7(Deno.build.arch),"X-Stainless-Runtime":"deno","X-Stainless-Runtime-Version":typeof Deno.version==="string"?Deno.version:Deno.version?.deno??"unknown"};if(typeof EdgeRuntime<"u")return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fe,"X-Stainless-OS":"Unknown","X-Stainless-Arch":`other:${EdgeRuntime}`,"X-Stainless-Runtime":"edge","X-Stainless-Runtime-Version":globalThis.process.version};if(q==="node")return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fe,"X-Stainless-OS":PM7(globalThis.process.platform??"unknown"),"X-Stainless-Arch":XM7(globalThis.process.arch??"unknown"),"X-Stainless-Runtime":"node","X-Stainless-Runtime-Version":globalThis.process.version??"unknown"};let K=UO5();if(K)return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fe,"X-Stainless-OS":"Unknown","X-Stainless-Arch":"unknown","X-Stainless-Runtime":`browser:${K.browser}`,"X-Stainless-Runtime-Version":K.version};return{"X-Stainless-Lang":"js","X-Stainless-Package-Version":fe,"X-Stainless-OS":"Unknown","X-Stainless-Arch":"unknown","X-Stainless-Runtime":"unknown","X-Stainless-Runtime-Version":"unknown"}},XM7=(q)=>{if(q==="x32")return"x32";if(q==="x86_64"||q==="x64")return"x64";if(q==="arm")return"arm";if(q==="aarch64"||q==="arm64")return"arm64";if(q)return`other:${q}`;return"unknown"},PM7=(q)=>{if(q=q.toLowerCase(),q.includes("ios"))return"iOS";if(q==="android")return"Android";if(q==="darwin")return"MacOS";if(q==="win32")return"Windows";if(q==="freebsd")return"FreeBSD";if(q==="openbsd")return"OpenBSD";if(q==="linux")return"Linux";if(q)return`Other:${q}`;return"Unknown"},WM7,fM7=()=>{return WM7??(WM7=FO5())};var ua8=()=>{};function ZM7(){if(typeof fetch<"u")return fetch;throw Error("`fetch` is not defined as a global;
   Either pass `fetch` to the client, `new Anthropic({ fetch })` or polyfill the global, `globalThis.fetch = fetch`")}function ma8(...q){let K=globalThis.ReadableStream;if(typeof K>"u")throw Error("`ReadableStream` is not defined as a global;
   You will need to polyfill it, `globalThis.ReadableStream = ReadableStream`");return new K(...q)}function $_8(q){let K=Symbol.asyncIterator in q?q[Symbol.asyncIterator]():q[Symbol.iterator]();return ma8({start(){},async pull(_){let{done:z,value:Y}=await K.next();if(z)_.close();else _.enqueue(Y)},async cancel(){await K.return?.()}})}function yI6(q){if(q[Symbol.asyncIterator])return q;let K=q.getReader();return{async next(){try{let _=await K.read();if(_?.done)K.releaseLock();return _}catch(_){throw K.releaseLock(),_}},async return(){let _=K.cancel();return K.releaseLock(),await _,{done:!0,value:void 0}},[Symbol.asyncIterator](){return this}}}async function GM7(q){if(q===null||typeof q!=="object")return;if(q[Symbol.asyncIterator]){await q[Symbol.asyncIterator]().return?.();return}let K=q.getReader(),_=K.cancel();K.releaseLock(),await _}var vM7=({headers:q,body:K})=>{return{bodyHeaders:{"content-type":"application/json"},body:JSON.stringify(K)}};function TM7(q){return Object.entries(q).filter(([K,_])=>typeof _<"u").map(([K,_])=>{if(typeof _==="string"||typeof _==="number"||typeof _==="boolean")return`${encodeURIComponent(K)}=${encodeURIComponent(_)}`;if(_===null)return`${encodeURIComponent(K)}=`;throw new mq(`Cannot stringify type ${typeof _}; Expected string, number, boolean, or null. If you need to pass nested query parameters, you can manually encode them, e.g. { query: { 'foo[key1]': value1, 'foo[key2]': value2 } }, and please open a GitHub issue requesting better support for your use case.`)}).join("&")}var kM7=L(()=>{FW()});function yM7(q){let K=0;for(let Y of q)K+=Y.length;let _=new Uint8Array(K),z=0;for(let Y of q)_.set(Y,z),z+=Y.length;return _}function EI6(q){let K;return(VM7??(K=new globalThis.TextEncoder,VM7=K.encode.bind(K)))(q)}function pa8(q){let K;return(NM7??(K=new globalThis.TextDecoder,NM7=K.decode.bind(K)))(q)}var VM7,NM7;class Ze{constructor(){kE.set(this,void 0),VE.set(this,void 0),J4(this,kE,new Uint8Array,"f"),J4(this,VE,null,"f")}decode(q){if(q==null)return[];let K=q instanceof ArrayBuffer?new Uint8Array(q):typeof q==="string"?EI6(q):q;J4(this,kE,yM7([x1(this,kE,"f"),K]),"f");let _=[],z;while((z=cO5(x1(this,kE,"f"),x1(this,VE,"f")))!=null){if(z.carriage&&x1(this,VE,"f")==null){J4(this,VE,z.index,"f");continue}if(x1(this,VE,"f")!=null&&(z.index!==x1(this,VE,"f")+1||z.carriage)){_.push(pa8(x1(this,kE,"f").subarray(0,x1(this,VE,"f")-1))),J4(this,kE,x1(this,kE,"f").subarray(x1(this,VE,"f")),"f"),J4(this,VE,null,"f");continue}let Y=x1(this,VE,"f")!==null?z.preceding-1:z.preceding,$=pa8(x1(this,kE,"f").subarray(0,Y));_.push($),J4(this,kE,x1(this,kE,"f").subarray(z.index),"f"),J4(this,VE,null,"f")}return _}flush(){if(!x1(this,kE,"f").length)return[];return this.decode(`
`)}}function cO5(q,K){for(let Y=K??0;Y<q.length;Y++){if(q[Y]===10)return{preceding:Y,index:Y+1,carriage:!1};if(q[Y]===13)return{preceding:Y,index:Y+1,carriage:!0}}return null}function EM7(q){for(let z=0;z<q.length-1;z++){if(q[z]===10&&q[z+1]===10)return z+2;if(q[z]===13&&q[z+1]===13)return z+2;if(q[z]===13&&q[z+1]===10&&z+3<q.length&&q[z+2]===13&&q[z+3]===10)return z+4}return-1}var kE,VE;var Ba8=L(()=>{Dl();kE=new WeakMap,VE=new WeakMap;Ze.NEWLINE_CHARS=new Set([`
`,"\r"]);Ze.NEWLINE_REGEXP=/\r\n|[\n\r]/g});function LI6(){}function O_8(q,K,_){if(!K||A_8[q]>A_8[_])return LI6;else return K[q].bind(K)}function UW(q){let K=q.logger,_=q.logLevel??"off";if(!K)return lO5;let z=LM7.get(K);if(z&&z[0]===_)return z[1];let Y={error:O_8("error",K,_),warn:O_8("warn",K,_),info:O_8("info",K,_),debug:O_8("debug",K,_)};return LM7.set(K,[_,Y]),Y}var A_8,ga8=(q,K,_)=>{if(!q)return;if(HM7(A_8,q))return q;UW(_).warn(`${K} was set to ${JSON.stringify(q)}, expected one of ${JSON.stringify(Object.keys(A_8))}`);return},lO5,LM7,Zl=(q)=>{if(q.options)q.options={...q.options},delete q.options.headers;if(q.headers)q.headers=Object.fromEntries((q.headers instanceof Headers?[...q.headers]:Object.entries(q.headers)).map(([K,_])=>[K,K.toLowerCase()==="x-api-key"||K.toLowerCase()==="authorization"||K.toLowerCase()==="cookie"||K.toLowerCase()==="set-cookie"?"***":_]));if("retryOfRequestLogID"in q){if(q.retryOfRequestLogID)q.retryOf=q.retryOfRequestLogID;delete q.retryOfRequestLogID}return q};var w_8=L(()=>{U96();A_8={off:0,error:200,warn:300,info:400,debug:500};lO5={error:LI6,warn:LI6,info:LI6,debug:LI6},LM7=new WeakMap});async function*nO5(q,K){if(!q.body){if(K.abort(),typeof globalThis.navigator<"u"&&globalThis.navigator.product==="ReactNative")throw new mq("The default react-native fetch implementation does not support streaming. Please use expo/fetch: https://docs.expo.dev/versions/latest/sdk/expo/#expofetch-api");throw new mq("Attempted to iterate over a response with no body")}let _=new hM7,z=new Ze,Y=yI6(q.body);for await(let $ of iO5(Y))for(let O of z.decode($)){let A=_.decode(O);if(A)yield A}for(let $ of z.flush()){let O=_.decode($);if(O)yield O}}async function*iO5(q){let K=new Uint8Array;for await(let _ of q){if(_==null)continue;let z=_ instanceof ArrayBuffer?new Uint8Array(_):typeof _==="string"?EI6(_):_,Y=new Uint8Array(K.length+z.length);Y.set(K),Y.set(z,K.length),K=Y;let $;while(($=EM7(K))!==-1)yield K.slice(0,$),K=K.slice($)}if(K.length>0)yield K}class hM7{constructor(){this.event=null,this.data=[],this.chunks=[]}decode(q){if(q.endsWith("\r"))q=q.substring(0,q.length-1);if(!q){if(!this.event&&!this.data.length)return null;let Y={event:this.event,data:this.data.join(`
`),raw:this.chunks};return this.event=null,this.data=[],this.chunks=[],Y}if(this.chunks.push(q),q.startsWith(":"))return null;let[K,_,z]=rO5(q,":");if(z.startsWith(" "))z=z.substring(1);if(K==="event")this.event=z;else if(K==="data")this.data.push(z);return null}}function rO5(q,K){let _=q.indexOf(K);if(_!==-1)return[q.substring(0,_),K,q.substring(_+K.length)];return[q,"",""]}var hI6,rv;var Fa8=L(()=>{Dl();FW();Ba8();U96();w_8();FW();rv=class rv{constructor(q,K,_){this.iterator=q,hI6.set(this,void 0),this.controller=K,J4(this,hI6,_,"f")}static fromSSEResponse(q,K,_){let z=!1,Y=_?UW(_):console;async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1;try{for await(let A of nO5(q,K)){if(A.event==="completion")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="message_start"||A.event==="message_delta"||A.event==="message_stop"||A.event==="content_block_start"||A.event==="content_block_delta"||A.event==="content_block_stop")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="ping")continue;if(A.event==="error")throw new nq(void 0,Y_8(A.data)??A.data,void 0,q.headers)}O=!0}catch(A){if(fl(A))return;throw A}finally{if(!O)K.abort()}}return new rv($,K,_)}static fromReadableStream(q,K,_){let z=!1;async function*Y(){let O=new Ze,A=yI6(q);for await(let w of A)for(let j of O.decode(w))yield j;for(let w of O.flush())yield w}async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1;try{for await(let A of Y()){if(O)continue;if(A)yield JSON.parse(A)}O=!0}catch(A){if(fl(A))return;throw A}finally{if(!O)K.abort()}}return new rv($,K,_)}[(hI6=new WeakMap,Symbol.asyncIterator)](){return this.iterator()}tee(){let q=[],K=[],_=this.iterator(),z=(Y)=>{return{next:()=>{if(Y.length===0){let $=_.next();q.push($),K.push($)}return Y.shift()}}};return[new rv(()=>z(q),this.controller,x1(this,hI6,"f")),new rv(()=>z(K),this.controller,x1(this,hI6,"f"))]}toReadableStream(){let q=this,K;return ma8({async start(){K=q[Symbol.asyncIterator]()},async pull(_){try{let{value:z,done:Y}=await K.next();if(Y)return _.close();let $=EI6(JSON.stringify(z)+`
`);_.enqueue($)}catch(z){_.error(z)}},async cancel(){await K.return?.()}})}}});async function j_8(q,K){let{response:_,requestLogID:z,retryOfRequestLogID:Y,startTime:$}=K,O=await(async()=>{if(K.options.stream){if(UW(q).debug("response",_.status,_.url,_.headers,_.body),K.options.__streamClass)return K.options.__streamClass.fromSSEResponse(_,K.controller);return rv.fromSSEResponse(_,K.controller)}if(_.status===204)return null;if(K.options.__binaryResponse)return _;let w=_.headers.get("content-type")?.split(";
  ")[0]?.trim();if(w?.includes("application/json")||w?.endsWith("+json")){if(_.headers.get("content-length")==="0")return;let M=await _.json();return Ua8(M,_)}return await _.text()})();return UW(q).debug(`[${z}] response parsed`,Zl({retryOfRequestLogID:Y,url:_.url,status:_.status,body:O,durationMs:Date.now()-$})),O}function Ua8(q,K){if(!q||typeof q!=="object"||Array.isArray(q))return q;return Object.defineProperty(q,"_request_id",{value:K.headers.get("request-id"),enumerable:!1})}var Qa8=L(()=>{Fa8();w_8()});var RI6,Q96;var H_8=L(()=>{Dl();Qa8();Q96=class Q96 extends Promise{constructor(q,K,_=j_8){super((z)=>{z(null)});this.responsePromise=K,this.parseResponse=_,RI6.set(this,void 0),J4(this,RI6,q,"f")}_thenUnwrap(q){return new Q96(x1(this,RI6,"f"),this.responsePromise,async(K,_)=>Ua8(q(await this.parseResponse(K,_),_),_.response))}asResponse(){return this.responsePromise.then((q)=>q.response)}async withResponse(){let[q,K]=await Promise.all([this.parse(),this.asResponse()]);return{data:q,response:K,request_id:K.headers.get("request-id")}}parse(){if(!this.parsedPromise)this.parsedPromise=this.responsePromise.then((q)=>this.parseResponse(x1(this,RI6,"f"),q));return this.parsedPromise}then(q,K){return this.parse().then(q,K)}catch(q){return this.parse().catch(q)}finally(q){return this.parse().finally(q)}};RI6=new WeakMap});var J_8,da8,M_8,qI,SI6;var uB=L(()=>{Dl();FW();Qa8();H_8();U96();da8=class da8{constructor(q,K,_,z){J_8.set(this,void 0),J4(this,J_8,q,"f"),this.options=z,this.response=K,this.body=_}hasNextPage(){if(!this.getPaginatedItems().length)return!1;return this.nextPageRequestOptions()!=null}async getNextPage(){let q=this.nextPageRequestOptions();if(!q)throw new mq("No next page expected;
   please check `.hasNextPage()` before calling `.getNextPage()`.");return await x1(this,J_8,"f").requestAPIList(this.constructor,q)}async*iterPages(){let q=this;yield q;while(q.hasNextPage())q=await q.getNextPage(),yield q}async*[(J_8=new WeakMap,Symbol.asyncIterator)](){for await(let q of this.iterPages())for(let K of q.getPaginatedItems())yield K}};M_8=class M_8 extends Q96{constructor(q,K,_){super(q,K,async(z,Y)=>new _(z,Y.response,await j_8(z,Y),Y.options))}async*[Symbol.asyncIterator](){let q=await this;for await(let K of q)yield K}};qI=class qI extends da8{constructor(q,K,_,z){super(q,K,_,z);this.data=_.data||[],this.has_more=_.has_more||!1,this.first_id=_.first_id||null,this.last_id=_.last_id||null}getPaginatedItems(){return this.data??[]}hasNextPage(){if(this.has_more===!1)return!1;return super.hasNextPage()}nextPageRequestOptions(){if(this.options.query?.before_id){let K=this.first_id;if(!K)return null;return{...this.options,query:{...z_8(this.options.query),before_id:K}}}let q=this.last_id;if(!q)return null;return{...this.options,query:{...z_8(this.options.query),after_id:q}}}};SI6=class SI6 extends da8{constructor(q,K,_,z){super(q,K,_,z);this.data=_.data||[],this.has_more=_.has_more||!1,this.next_page=_.next_page||null}getPaginatedItems(){return this.data??[]}hasNextPage(){if(this.has_more===!1)return!1;return super.hasNextPage()}nextPageRequestOptions(){let q=this.next_page;if(!q)return null;return{...this.options,query:{...z_8(this.options.query),page:q}}}}});function d96(q,K,_){return la8(),new File(q,K??"unknown_file",_)}function CI6(q,K){let _=typeof q==="object"&&q!==null&&(("name"in q)&&q.name&&String(q.name)||("url"in q)&&q.url&&String(q.url)||("filename"in q)&&q.filename&&String(q.filename)||("path"in q)&&q.path&&String(q.path))||"";return K?_.split(/[\\/]/).pop()||void 0:_}function aO5(q){let K=typeof q==="function"?q:q.fetch,_=RM7.get(K);if(_)return _;let z=(async()=>{try{let Y="Response"in K?K.Response:(await K("data:,")).constructor,$=new FormData;if($.toString()===await new Y($).text())return!1;return!0}catch{return!0}})();return RM7.set(K,z),z}var la8=()=>{if(typeof File>"u"){let{process:q}=globalThis,K=typeof q?.versions?.node==="string"&&parseInt(q.versions.node.split("."))<20;throw Error("`File` is not defined as a global, which is required for file uploads."+(K?" Update to Node 20 LTS or newer, or set `globalThis.File` to `import('node:buffer').File`.":""))}},na8=(q)=>q!=null&&typeof q==="object"&&typeof q[Symbol.asyncIterator]==="function",vP6=async(q,K,_=!0)=>{return{...q,body:await sO5(q.body,K,_)}},RM7,sO5=async(q,K,_=!0)=>{if(!await aO5(K))throw TypeError("The provided fetch function does not support file uploads with the current global FormData class.");let z=new FormData;return await Promise.all(Object.entries(q||{}).map(([Y,$])=>ca8(z,Y,$,_))),z},tO5=(q)=>q instanceof Blob&&("name"in q),ca8=async(q,K,_,z)=>{if(_===void 0)return;if(_==null)throw TypeError(`Received null for "${
    K
  }"; to pass null in FormData, you must use the string 'null'`);if(typeof _==="string"||typeof _==="number"||typeof _==="boolean")q.append(K,String(_));else if(_ instanceof Response){let Y={},$=_.headers.get("Content-Type");if($)Y={type:$};q.append(K,d96([await _.blob()],CI6(_,z),Y))}else if(na8(_))q.append(K,d96([await new Response($_8(_)).blob()],CI6(_,z)));else if(tO5(_))q.append(K,d96([_],CI6(_,z),{type:_.type}));else if(Array.isArray(_))await Promise.all(_.map((Y)=>ca8(q,K+"[]",Y,z)));else if(typeof _==="object")await Promise.all(Object.entries(_).map(([Y,$])=>ca8(q,`${K}[${Y}]`,$,z)));else throw TypeError(`Invalid value given to form, expected a string, number, boolean, object, Array, File or Blob but got ${_} instead`)};var TP6=L(()=>{RM7=new WeakMap});async function X_8(q,K,_){if(la8(),q=await q,K||(K=CI6(q,!0)),eO5(q)){if(q instanceof File&&K==null&&_==null)return q;return d96([await q.arrayBuffer()],K??q.name,{type:q.type,lastModified:q.lastModified,..._})}if(qA5(q)){let Y=await q.blob();return K||(K=new URL(q.url).pathname.split(/[\\/]/).pop()),d96(await ia8(Y),K,_)}let z=await ia8(q);if(!_?.type){let Y=z.find(($)=>typeof $==="object"&&("type"in $)&&$.type);if(typeof Y==="string")_={..._,type:Y}}return d96(z,K,_)}async function ia8(q){let K=[];if(typeof q==="string"||ArrayBuffer.isView(q)||q instanceof ArrayBuffer)K.push(q);else if(SM7(q))K.push(q instanceof Blob?q:await q.arrayBuffer());else if(na8(q))for await(let _ of q)K.push(...await ia8(_));else{let _=q?.constructor?.name;throw Error(`Unexpected data type: ${typeof q}${_?`; constructor: ${_}`:""}${KA5(q)}`)}return K}function KA5(q){if(typeof q!=="object"||q===null)return"";return`; props: [${Object.getOwnPropertyNames(q).map((_)=>`"${
    _
  }"`).join(", ")}]`}var SM7=(q)=>q!=null&&typeof q==="object"&&typeof q.size==="number"&&typeof q.type==="string"&&typeof q.text==="function"&&typeof q.slice==="function"&&typeof q.arrayBuffer==="function",eO5=(q)=>q!=null&&typeof q==="object"&&typeof q.name==="string"&&typeof q.lastModified==="number"&&SM7(q),qA5=(q)=>q!=null&&typeof q==="object"&&typeof q.url==="string"&&typeof q.blob==="function";var CM7=L(()=>{TP6();TP6()});var ra8=L(()=>{CM7()});var bM7=()=>{};class AH{constructor(q){this._client=q}}function*zA5(q){if(!q)return;if(xM7 in q){let{values:z,nulls:Y}=q;yield*z.entries();for(let $ of Y)yield[$,null];return}let K=!1,_;if(q instanceof Headers)_=q.entries();else if(xa8(q))_=q;else K=!0,_=Object.entries(q??{});for(let z of _){let Y=z[0];if(typeof Y!=="string")throw TypeError("expected header name to be a string");let $=xa8(z[1])?z[1]:[z[1]],O=!1;for(let A of $){if(A===void 0)continue;if(K&&!O)O=!0,yield[Y,null];yield[Y,A]}}}var xM7,x3=(q)=>{let K=new Headers,_=new Set;for(let z of q){let Y=new Set;for(let[$,O]of zA5(z)){let A=$.toLowerCase();if(!Y.has(A))K.delete($),Y.add(A);if(O===null)K.delete($),_.add(A);else K.append($,O),_.delete(A)}}return{[xM7]:!0,values:K,nulls:_}};var NE=L(()=>{U96();xM7=Symbol.for("brand.privateNullableHeaders")});function P_8(q){return typeof q==="object"&&q!==null&&bI6 in q}function oa8(q,K){let _=new Set;if(q){for(let z of q)if(P_8(z))_.add(z[bI6])}if(K)for(let z of K){if(P_8(z))_.add(z[bI6]);if(Array.isArray(z.content)){for(let Y of z.content)if(P_8(Y))_.add(Y[bI6])}}return Array.from(_)}function W_8(q,K){let _=oa8(q,K);if(_.length===0)return{};return{"x-stainless-helper":_.join(", ")}}function IM7(q){if(P_8(q))return{"x-stainless-helper":q[bI6]};return{}}var bI6;var xI6=L(()=>{bI6=Symbol("anthropic.sdk.stainlessHelper")});function mM7(q){return q.replace(/[^A-Za-z0-9\-._~!$&'()*+,;=:@]+/g,encodeURIComponent)}var uM7,YA5=(q=mM7)=>function(_,...z){if(_.length===1)return _[0];let Y=!1,$=[],O=_.reduce((H,J,M)=>{if(/[?#]/.test(J))Y=!0;let X=z[M],P=(Y?encodeURIComponent:q)(""+X);if(M!==z.length&&(X==null||typeof X==="object"&&X.toString===Object.getPrototypeOf(Object.getPrototypeOf(X.hasOwnProperty??uM7)??uM7)?.toString))P=X+"",$.push({start:H.length+J.length,length:P.length,error:`Value of type ${Object.prototype.toString.call(X).slice(8,-1)} is not a valid path parameter`});return H+J+(M===z.length?"":P)},""),A=O.split(/[?#]/,1)[0],w=/(?<=^|\/)(?:\.|%2e){1,2}(?=\/|$)/gi,j;while((j=w.exec(A))!==null)$.push({start:j.index,length:j[0].length,error:`Value "${
    j[0]
  }" can't be safely passed as a path parameter`});if($.sort((H,J)=>H.start-J.start),$.length>0){let H=0,J=$.reduce((M,X)=>{let P=" ".repeat(X.start-H),W="^".repeat(X.length);return H=X.start+X.length,M+P+W},"");throw new mq(`Path parameters result in path with invalid segments:
${$.map((M)=>M.error).join(`
`)}
${O}
${J}`)}return O},jj;var Ge=L(()=>{FW();uM7=Object.freeze(Object.create(null)),jj=YA5(mM7)});var II6;var aa8=L(()=>{uB();NE();xI6();TP6();Ge();II6=class II6 extends AH{list(q={},K){let{betas:_,...z}=q??{};return this._client.getAPIList("/v1/files",qI,{query:z,...K,headers:x3([{"anthropic-beta":[..._??[],"files-api-2025-04-14"].toString()},K?.headers])})}delete(q,K={},_){let{betas:z}=K??{};return this._client.delete(jj`/v1/files/${q}`,{..._,headers:x3([{"anthropic-beta":[...z??[],"files-api-2025-04-14"].toString()},_?.headers])})}download(q,K={},_){let{betas:z}=K??{};return this._client.get(jj`/v1/files/${q}/content`,{..._,headers:x3([{"anthropic-beta":[...z??[],"files-api-2025-04-14"].toString(),Accept:"application/binary"},_?.headers]),__binaryResponse:!0})}retrieveMetadata(q,K={},_){let{betas:z}=K??{};return this._client.get(jj`/v1/files/${q}`,{..._,headers:x3([{"anthropic-beta":[...z??[],"files-api-2025-04-14"].toString()},_?.headers])})}upload(q,K){let{betas:_,...z}=q;return this._client.post("/v1/files",vP6({body:z,...K,headers:x3([{"anthropic-beta":[..._??[],"files-api-2025-04-14"].toString()},IM7(z.file),K?.headers])},this._client))}}});var uI6;var sa8=L(()=>{uB();NE();Ge();uI6=class uI6 extends AH{retrieve(q,K={},_){let{betas:z}=K??{};return this._client.get(jj`/v1/models/${q}?beta=true`,{..._,headers:x3([{...z?.toString()!=null?{"anthropic-beta":z?.toString()}:void 0},_?.headers])})}list(q={},K){let{betas:_,...z}=q??{};return this._client.getAPIList("/v1/models?beta=true",qI,{query:z,...K,headers:x3([{..._?.toString()!=null?{"anthropic-beta":_?.toString()}:void 0},K?.headers])})}}});var ve=L(()=>{FW()});var D_8;var ta8=L(()=>{D_8={"claude-opus-4-20250514":8192,"claude-opus-4-0":8192,"claude-4-opus-20250514":8192,"anthropic.claude-opus-4-20250514-v1:0":8192,"claude-opus-4@20250514":8192,"claude-opus-4-1-20250805":8192,"anthropic.claude-opus-4-1-20250805-v1:0":8192,"claude-opus-4-1@20250805":8192}});function pM7(q){return q?.output_format??q?.output_config?.format}function ea8(q,K,_){let z=pM7(K);if(!K||!("parse"in(z??{})))return{...q,content:q.content.map((Y)=>{if(Y.type==="text"){let $=Object.defineProperty({...Y},"parsed_output",{value:null,enumerable:!1});return Object.defineProperty($,"parsed",{get(){return _.logger.warn("The `parsed` property on `text` blocks is deprecated, please use `parsed_output` instead."),null},enumerable:!1})}return Y}),parsed_output:null};return qs8(q,K,_)}function qs8(q,K,_){let z=null,Y=q.content.map(($)=>{if($.type==="text"){let O=AA5(K,$.text);if(z===null)z=O;let A=Object.defineProperty({...$},"parsed_output",{value:O,enumerable:!1});return Object.defineProperty(A,"parsed",{get(){return _.logger.warn("The `parsed` property on `text` blocks is deprecated, please use `parsed_output` instead."),O},enumerable:!1})}return $});return{...q,content:Y,parsed_output:z}}function AA5(q,K){let _=pM7(q);if(_?.type!=="json_schema")return null;try{if("parse"in _)return _.parse(K);return JSON.parse(K)}catch(z){throw new mq(`Failed to parse structured output: ${z}`)}}var Ks8=L(()=>{FW()});var wA5=(q)=>{let K=0,_=[];while(K<q.length){let z=q[K];if(z==="\\"){K++;continue}if(z==="{
    "){_.push({type:"brace",value:"{
      "}),K++;continue}if(z==="
    }"){_.push({type:"brace",value:"
  }"}),K++;continue}if(z==="["){_.push({type:"paren",value:"["}),K++;continue}if(z==="]"){_.push({type:"paren",value:"]"}),K++;continue}if(z===":"){_.push({type:"separator",value:":"}),K++;continue}if(z===","){_.push({type:"delimiter",value:","}),K++;continue}if(z==='"'){let A="",w=!1;z=q[++K];while(z!=='"'){if(K===q.length){w=!0;break}if(z==="\\"){if(K++,K===q.length){w=!0;break}A+=z+q[K],z=q[++K]}else A+=z,z=q[++K]}if(z=q[++K],!w)_.push({type:"string",value:A});continue}if(z&&/\s/.test(z)){K++;continue}let $=/[0-9]/;if(z&&$.test(z)||z==="-"||z==="."){let A="";if(z==="-")A+=z,z=q[++K];while(z&&$.test(z)||z===".")A+=z,z=q[++K];_.push({type:"number",value:A});continue}let O=/[a-z]/i;if(z&&O.test(z)){let A="";while(z&&O.test(z)){if(K===q.length)break;A+=z,z=q[++K]}if(A=="true"||A=="false"||A==="null")_.push({type:"name",value:A});else{K++;continue}continue}K++}return _},kP6=(q)=>{if(q.length===0)return q;let K=q[q.length-1];switch(K.type){case"separator":return q=q.slice(0,q.length-1),kP6(q);break;case"number":let _=K.value[K.value.length-1];if(_==="."||_==="-")return q=q.slice(0,q.length-1),kP6(q);case"string":let z=q[q.length-2];if(z?.type==="delimiter")return q=q.slice(0,q.length-1),kP6(q);else if(z?.type==="brace"&&z.value==="{
    ")return q=q.slice(0,q.length-1),kP6(q);break;case"delimiter":return q=q.slice(0,q.length-1),kP6(q);break}return q},jA5=(q)=>{let K=[];if(q.map((_)=>{if(_.type==="brace")if(_.value==="{
      ")K.push("
    }");else K.splice(K.lastIndexOf("
  }"),1);if(_.type==="paren")if(_.value==="[")K.push("]");else K.splice(K.lastIndexOf("]"),1)}),K.length>0)K.reverse().map((_)=>{if(_==="
} /* confidence: 95% */

/* original: O */ let composed_value=A31(z,{
  offset:Y,length:$-Y
} /* confidence: 30% */

/* original: rg7 */ var it_is_was_this_they_are_were_t=B((dd$,ig7)=>{
  var rQ5={
    pronoun:"it",is:"is",was:"was",this:"this"
  },oQ5={
    pronoun:"they",is:"are",was:"were",this:"these"
  };
  ig7.exports=class{
    constructor(K,_){
      this.singular=K,this.plural=_
    }pluralize(K){
      let _=K===1,z=_?rQ5:oQ5,Y=_?this.singular:this.plural;
      return{
        ...z,count:K,noun:Y
      }
    }
  }
} /* confidence: 65% */

/* original: O */ let composed_value=this.sessionCache.get(_)||new md7; /* confidence: 30% */

/* original: VDq */ var composed_value=class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
}; /* confidence: 30% */

/* original: lb9 */ var function_function=VDq,NDq=CWq(),yDq=class{
  constructor(K){
    this.middlewareStack=(0,NDq.constructStack)(),this.config=K
  }send(K,_,z){
    let Y=typeof _!=="function"?_:void 0,$=typeof _==="function"?_:z,O=K.resolveMiddleware(this.middlewareStack,this.config,Y);
    if($)O(K).then((A)=>$(null,A.output),(A)=>$(A)).catch(()=>{
      
    });
    else return O(K).then((A)=>A.output)
  }destroy(){
    if(this.config.requestHandler.destroy)this.config.requestHandler.destroy()
  }
}; /* confidence: 65% */

/* original: Bfq */ var composed_value=L(()=>{
  QP8()
} /* confidence: 30% */

/* original: fZq */ var composed_value=L(()=>{
  rP8();
  hQ6()
} /* confidence: 30% */

/* original: uZq */ var composed_value=L(()=>{
  rP8();
  bZq=w6(lB6(),1),xZq=w6(hZq(),1),RZq=[],Tu9=new Map
} /* confidence: 30% */

/* original: bi */ var composed_value=L(()=>{
  qf8();
  FA();
  yf1();
  zf8();
  pP();
  dQ6();
  uyq();
  bY6();
  BP();
  MN=v9("MsalClient")
} /* confidence: 30% */

/* original: Of8 */ var SseTransport=L(()=>{
  bi();
  BP();
  pP();
  FA();
  wZ();
  YEq=v9("ClientAssertionCredential")
} /* confidence: 30% */

/* original: EF */ var composed_value=L(()=>{
  FA()
} /* confidence: 30% */

/* original: AEq */ var composed_value=L(()=>{
  FA()
} /* confidence: 30% */

/* original: dv1 */ var composed_value=L(()=>{
  pP();
  FA();
  wZ();
  Qv1=v9("ChainedTokenCredential")
} /* confidence: 30% */

/* original: cv1 */ var composed_value=L(()=>{
  bi();
  BP();
  FA();
  wZ();
  ZEq=v9(Cc6)
} /* confidence: 30% */

/* original: lv1 */ var composed_value=L(()=>{
  bi();
  BP();
  pP();
  FA();
  EF();
  wZ();
  GEq=v9("ClientSecretCredential")
} /* confidence: 30% */

/* original: nv1 */ var composed_value=L(()=>{
  bi();
  BP();
  pP();
  FA();
  EF();
  wZ();
  Vn9=v9("UsernamePasswordCredential")
} /* confidence: 30% */

/* original: iv1 */ var AZURE_TENANT_ID_AZURE_CLIENT_I=L(()=>{
  pP();
  FA();
  cv1();
  lv1();
  nv1();
  BP();
  wZ();
  Nn9=["AZURE_TENANT_ID","AZURE_CLIENT_ID","AZURE_CLIENT_SECRET","AZURE_CLIENT_CERTIFICATE_PATH","AZURE_CLIENT_CERTIFICATE_PASSWORD","AZURE_USERNAME","AZURE_PASSWORD","AZURE_ADDITIONALLY_ALLOWED_TENANTS","AZURE_CLIENT_SEND_CERTIFICATE_CHAIN"];
  mi=v9(jf8)
} /* confidence: 65% */

/* original: TEq */ var composed_value=L(()=>{
  BP();
  FA();
  EF();
  wZ();
  bi();
  jF();
  av1=v9("InteractiveBrowserCredential")
} /* confidence: 30% */

/* original: kEq */ var composed_value=L(()=>{
  BP();
  FA();
  EF();
  wZ();
  bi();
  jF();
  tv1=v9("DeviceCodeCredential")
} /* confidence: 30% */

/* original: yEq */ var composed_value=L(()=>{
  BP();
  BP();
  FA();
  EF();
  wZ();
  bi();
  NEq=v9("AuthorizationCodeCredential")
} /* confidence: 30% */

/* original: LEq */ var composed_value=L(()=>{
  bi();
  FA();
  BP();
  pP();
  EF();
  wZ();
  _T1=v9(ZG6)
} /* confidence: 30% */

/* original: nIq */ var __esModule_function_error_warn=B((cIq)=>{
  Object.defineProperty(cIq,"__esModule",{
    value:!0
  });
  cIq.createLogLevelDiagLogger=void 0;
  var ai=VZ8();
  function a7_(q,K){
    if(q<ai.DiagLogLevel.NONE)q=ai.DiagLogLevel.NONE;
    else if(q>ai.DiagLogLevel.ALL)q=ai.DiagLogLevel.ALL;
    K=K||{
      
    };
    function _(z,Y){
      let $=K[z];
      if(typeof $==="function"&&q>=Y)return $.bind(K);
      return function(){
        
      }
    }return{
      error:_("error",ai.DiagLogLevel.ERROR),warn:_("warn",ai.DiagLogLevel.WARN),info:_("info",ai.DiagLogLevel.INFO),debug:_("debug",ai.DiagLogLevel.DEBUG),verbose:_("verbose",ai.DiagLogLevel.VERBOSE)
    }
  }cIq.createLogLevelDiagLogger=a7_
} /* confidence: 65% */

/* original: R */ let composed_value=r54.resolveProps(v??[],{
  flow:A,indicator:"map-value-ind",next:k,offset:E.range[2],onError:Y,parentIndent:z.indent,startOnNewline:!1
} /* confidence: 30% */

/* original: $34 */ var options=B((EC_)=>{
  var kC_=Ri6(),Y34=z34(),VC_=tT6(),NC_=Ci6();
  function yC_(q,K,{
    offset:_,start:z,value:Y,end:$
  },O){
    let A=Object.assign({
      _directives:K
    },q),w=new kC_.Document(void 0,A),j={
      atKey:!1,atRoot:!0,directives:w.directives,options:w.options,schema:w.schema
    },H=NC_.resolveProps(z,{
      indicator:"doc-start",next:Y??$?.[0],offset:_,onError:O,parentIndent:0,startOnNewline:!0
    });
    if(H.found){
      if(w.directives.docStart=!0,Y&&(Y.type==="block-map"||Y.type==="block-seq")&&!H.hasNewline)O(H.end,"MISSING_CHAR","Block collection cannot start on same line with directives-end marker")
    }w.contents=Y?Y34.composeNode(j,Y,H,O):Y34.composeEmptyNode(j,H.end,z,null,H,O);
    let J=w.contents.range[2],M=VC_.resolveEnd($,J,!1,O);
    if(M.comment)w.comment=M.comment;
    return w.range=[_,J,M.offset],w
  }EC_.composeDoc=yC_
} /* confidence: 70% */

/* original: Y */ let composed_value={
  type:"block-map",offset:q.offset,indent:q.indent,items:[{
    start:_,key:q,sep:z
  }]
}; /* confidence: 30% */

/* original: O */ let composed_value={
  type:"block-map",offset:q.offset,indent:q.indent,items:[{
    start:Y,key:q,sep:$
  }]
}; /* confidence: 30% */

/* original: LA6 */ var composed_value=L(()=>{
  kt8();
  kt8()
} /* confidence: 30% */

/* original: $ */ let composed_value=Y&&"retain"in Y?{
  ...q,retain:Y.retain,startTime:Y.startTime,messages:Y.messages,diskLoaded:Y.diskLoaded,pendingMessages:Y.pendingMessages
} /* confidence: 30% */

/* original: KNz */ var Logger=(q,...K)=>{
  let _;
  if(q>=wE6){
    switch(q){
      case KK6.LogVerbosity.DEBUG:_=Q26.debug;
      break;
      case KK6.LogVerbosity.INFO:_=Q26.info;
      break;
      case KK6.LogVerbosity.ERROR:_=Q26.error;
      break
    }if(!_)_=Q26.error;
    if(_)_.bind(Q26)(...K)
  }
}; /* confidence: 44% */

/* original: O */ let composed_value={
  info:$?Y:K,timestamp:Date.now()
}; /* confidence: 30% */

/* original: Y */ let composed_value=/<mcp-polling-update\s+type="([^"]+)"\s+server="([^"]+)"\s+tool="([^"]+)"[^>]*>(?:[\s\S]*?<reason>([^<]+)<\/reason>)?/g;while((z=composed_value.exec(q))!==null)K.push({kind:"polling",server:z[2]??"",target:z[3]??"",reason:z[4]});return K}function Hcz(q){if(q.startsWith("file://")){let K=q.slice(7),_=K.split("/");return _[_.length-1]||K}if(q.length>40)return q.slice(0,39)+"â¦";return q}function y3K(q){let K=Y6(12),{addMargin:_,param:z}=q,{text:composed_value}=z,$,O,A,w,j;if(K[0]!==_||K[1]!==composed_value){j=Symbol.for("react.early_return_sentinel");q:{let J=jcz(composed_value);if(J.length===0){j=null;break q}$=u,O="column",A=_?1:0,w=J.map(Jcz)}K[0]=_,K[1]=composed_value,K[2]=$,K[3]=O,K[4]=A,K[5]=w,K[6]=j}else $=K[2],O=K[3],A=K[4],w=K[5],j=K[6];if(j!==Symbol.for("react.early_return_sentinel"))return j;let H;if(K[7]!==$||K[8]!==O||K[9]!==A||K[10]!==w)H=hk.createElement($,{flexDirection:O,marginTop:A},w),K[7]=$,K[8]=O,K[9]=A,K[10]=w,K[11]=H;else H=K[11];return H}function Jcz(q,K){return hk.createElement(u,{key:K},hk.createElement(T,null,hk.createElement(T,{color:"success"},zb7)," ",hk.createElement(T,{dimColor:!0},q.server,":")," ",hk.createElement(T,{color:"suggestion"},q.kind==="resource"?Hcz(q.target):q.target),q.reason&&hk.createElement(T,{dimColor:!0}," Â· ",q.reason)))}var hk;var E3K=L(()=>{t6();S_();i6();hk=w6(D6(),1)});function Mcz(q){let K=Y6(7),{request:_}=q,z;if(K[0]!==_.from)z=wO.createElement(u,{marginBottom:1},wO.createElement(T,{color:"warning",bold:!0},"Shutdown request from ",_.from)),K[0]=_.from,K[1]=z;else z=K[1];let composed_value;if(K[2]!==_.reason)composed_value=_.reason&&wO.createElement(u,null,wO.createElement(T,null,"Reason: ",_.reason)),K[2]=_.reason,K[3]=composed_value;else composed_value=K[3];let $;if(K[4]!==z||K[5]!==composed_value)$=wO.createElement(u,{flexDirection:"column",marginY:1},wO.createElement(u,{borderStyle:"round",borderColor:"warning",flexDirection:"column",paddingX:1,paddingY:1},z,composed_value)),K[4]=z,K[5]=composed_value,K[6]=$;else $=K[6];return $}function Xcz(q){let K=Y6(8),{response:_}=q,z;if(K[0]!==_.from)z=wO.createElement(T,{color:"subtle",bold:!0},"Shutdown rejected by ",_.from),K[0]=_.from,K[1]=z;else z=K[1];let composed_value;if(K[2]!==_.reason)composed_value=wO.createElement(u,{marginTop:1,borderStyle:"dashed",borderColor:"subtle",borderLeft:!1,borderRight:!1,paddingX:1},wO.createElement(T,null,"Reason: ",_.reason)),K[2]=_.reason,K[3]=composed_value;else composed_value=K[3];let $;if(K[4]===Symbol.for("react.memo_cache_sentinel"))$=wO.createElement(u,{marginTop:1},wO.createElement(T,{dimColor:!0},"Teammate is continuing to work. You may request shutdown again later.")),K[4]=$;else $=K[4];let O;if(K[5]!==z||K[6]!==composed_value)O=wO.createElement(u,{flexDirection:"column",marginY:1},wO.createElement(u,{borderStyle:"round",borderColor:"subtle",flexDirection:"column",paddingX:1,paddingY:1},z,composed_value,$)),K[5]=z,K[6]=composed_value,K[7]=O;else O=K[7];return O}function L3K(q){let K=xK6(q);if(K)return wO.createElement(Mcz,{request:K});if(Ek(q))return null;let _=sx8(q);if(_)return wO.createElement(Xcz,{response:_});return null}function h3K(q){let K=xK6(q);if(K)return`[Shutdown Request from ${K.from}]${K.reason?` ${K.reason}`:""}`;let _=Ek(q);if(_)return`[Shutdown Approved] ${_.from} is now exiting`;let z=sx8(q);if(z)return`[Shutdown Rejected] ${z.from}: ${z.reason}`;return null}var wO;var Ja1=L(()=>{t6();i6();uJ();wO=w6(D6(),1)});function Pcz(q){let K=Y6(11),{assignment:_}=q,z;if(K[0]!==_.assignedBy||K[1]!==_.taskId)z=XW.createElement(u,{marginBottom:1},XW.createElement(T,{color:"cyan_FOR_SUBAGENTS_ONLY",bold:!0},"Task #",_.taskId," assigned by ",_.assignedBy)),K[0]=_.assignedBy,K[1]=_.taskId,K[2]=z;else z=K[2];let composed_value;if(K[3]!==_.subject)composed_value=XW.createElement(u,null,XW.createElement(T,{bold:!0},_.subject)),K[3]=_.subject,K[4]=composed_value;else composed_value=K[4];let $;if(K[5]!==_.description)$=_.description&&XW.createElement(u,{marginTop:1},XW.createElement(T,{dimColor:!0},_.description)),K[5]=_.description,K[6]=$;else $=K[6];let O;if(K[7]!==z||K[8]!==composed_value||K[9]!==$)O=XW.createElement(u,{flexDirection:"column",marginY:1},XW.createElement(u,{borderStyle:"round",borderColor:"cyan_FOR_SUBAGENTS_ONLY",flexDirection:"column",paddingX:1,paddingY:1},z,composed_value,$)),K[7]=z,K[8]=composed_value,K[9]=$,K[10]=O;else O=K[10];return O}function R3K(q){let K=tx8(q);if(K)return XW.createElement(Pcz,{assignment:K});return null}function S3K(q){let K=tx8(q);if(K)return`[Task Assigned] #${K.taskId} - ${K.subject}`;return null}var XW;var Ma1=L(()=>{t6();i6();uJ();XW=w6(D6(),1)});function Wcz(q){let K=Y6(10),{request:_}=q,z;if(K[0]!==_.from)z=B3.createElement(u,{marginBottom:1},B3.createElement(T,{color:"planMode",bold:!0},"Plan Approval Request from ",_.from)),K[0]=_.from,K[1]=z;else z=K[1];let composed_value;if(K[2]!==_.planContent)composed_value=B3.createElement(u,{borderStyle:"dashed",borderColor:"subtle",borderLeft:!1,borderRight:!1,flexDirection:"column",paddingX:1,marginBottom:1},B3.createElement(vA,null,_.planContent)),K[2]=_.planContent,K[3]=composed_value;else composed_value=K[3];let $;if(K[4]!==_.planFilePath)$=B3.createElement(T,{dimColor:!0},"Plan file: ",_.planFilePath),K[4]=_.planFilePath,K[5]=$;else $=K[5];let O;if(K[6]!==z||K[7]!==composed_value||K[8]!==$)O=B3.createElement(u,{flexDirection:"column",marginY:1},B3.createElement(u,{borderStyle:"round",borderColor:"planMode",flexDirection:"column",paddingX:1},z,composed_value,$)),K[6]=z,K[7]=composed_value,K[8]=$,K[9]=O;else O=K[9];return O}function Dcz(q){let K=Y6(13),{response:_,senderName:z}=q;if(_.approved){let w;if(K[0]!==z)w=B3.createElement(u,null,B3.createElement(T,{color:"success",bold:!0},"â Plan Approved by ",z)),K[0]=z,K[1]=w;else w=K[1];let j;if(K[2]===Symbol.for("react.memo_cache_sentinel"))j=B3.createElement(u,{marginTop:1},B3.createElement(T,null,"You can now proceed with implementation. Your plan mode restrictions have been lifted.")),K[2]=j;else j=K[2];let H;if(K[3]!==w)H=B3.createElement(u,{flexDirection:"column",marginY:1},B3.createElement(u,{borderStyle:"round",borderColor:"success",flexDirection:"column",paddingX:1,paddingY:1},w,j)),K[3]=w,K[4]=H;else H=K[4];return H}let composed_value;if(K[5]!==z)composed_value=B3.createElement(u,null,B3.createElement(T,{color:"error",bold:!0},"â Plan Rejected by ",z)),K[5]=z,K[6]=composed_value;else composed_value=K[6];let $;if(K[7]!==_.feedback)$=_.feedback&&B3.createElement(u,{marginTop:1,borderStyle:"dashed",borderColor:"subtle",borderLeft:!1,borderRight:!1,paddingX:1},B3.createElement(T,null,"Feedback: ",_.feedback)),K[7]=_.feedback,K[8]=$;else $=K[8];let O;if(K[9]===Symbol.for("react.memo_cache_sentinel"))O=B3.createElement(u,{marginTop:1},B3.createElement(T,{dimColor:!0},"Please revise your plan based on the feedback and call ExitPlanMode again.")),K[9]=O;else O=K[9];let A;if(K[10]!==composed_value||K[11]!==$)A=B3.createElement(u,{flexDirection:"column",marginY:1},B3.createElement(u,{borderStyle:"round",borderColor:"error",flexDirection:"column",paddingX:1,paddingY:1},composed_value,$,O)),K[10]=composed_value,K[11]=$,K[12]=A;else A=K[12];return A}function qu8(q,K){let _=hj6(q);if(_)return B3.createElement(Wcz,{request:_});let z=PL6(q);if(z)return B3.createElement(Dcz,{response:z,senderName:K});return null}function fcz(q){let K=hj6(q);if(K)return`[Plan Approval Request from ${K.from}]`;let _=PL6(q);if(_)if(_.approved)return"[Plan Approved] You can now proceed with implementation";else return`[Plan Rejected] ${_.feedback||"Please revise your plan"}`;return null}function Zcz(q){let K=["Agent idle"];if(q.completedTaskId){let _=q.completedStatus||"completed";K.push(`Task ${q.completedTaskId} ${_}`)}if(q.summary)K.push(`Last DM: ${q.summary}`);return K.join(" Â· ")}function C3K(q){let K=fcz(q);if(K)return K;let _=h3K(q);if(_)return _;let z=d68(q);if(z)return Zcz(z);let composed_value=S3K(q);if(composed_value)return composed_value;try{let $=l8(q);if($?.type==="teammate_terminated"&&$.message)return $.message}catch{}return q}var B3;var Xa1=L(()=>{t6();Hy();i6();r8();uJ();Ja1();Ma1();B3=w6(D6(),1)});function vcz(q){let K=[];for(let _ of q.matchAll(Gcz))if(_[1]&&_[4])K.push({teammateId:_[1],color:_[2],summary:_[3],content:_[4].trim()});return K}function Tcz(q){if(q==="leader")return"leader";return q}function b3K({addMargin:q,param:{text:K},isTranscriptMode:_}){let z=vcz(K).filter((composed_value)=>{if(Ek(composed_value.content))return!1;try{if(l8(composed_value.content)?.type==="teammate_terminated")return!1}catch{}return!0});if(z.length===0)return null;return k_.createElement(u,{flexDirection:"column",marginTop:q?1:0,width:"100%"},z.map((composed_value,$)=>{let O=K0(composed_value.color),A=Tcz(composed_value.teammateId),w=qu8(composed_value.content,A);if(w)return k_.createElement(k_.Fragment,{key:$},w);let j=L3K(composed_value.content);if(j)return k_.createElement(k_.Fragment,{key:$},j);let H=R3K(composed_value.content);if(H)return k_.createElement(k_.Fragment,{key:$},H);let J=null;try{J=l8(composed_value.content)}catch{}if(J?.type==="idle_notification")return null;if(J?.type==="task_completed"){let M=J;return k_.createElement(u,{key:$,flexDirection:"column",marginTop:1},k_.createElement(T,{color:O},`@${A}${o6.pointer}`),k_.createElement(_1,null,k_.createElement(T,{color:"success"},"â"),k_.createElement(T,null," ","Completed task #",M.taskId,M.taskSubject&&k_.createElement(T,{dimColor:!0}," (",M.taskSubject,")"))))}return k_.createElement(Pa1,{key:$,displayName:A,inkColor:O,content:composed_value.content,summary:composed_value.summary,isTranscriptMode:_})}))}function Pa1(q){let K=Y6(14),{displayName:_,inkColor:z,content:composed_value,summary:$,isTranscriptMode:O}=q,A=`@${_}${o6.pointer}`,w;if(K[0]!==z||K[1]!==A)w=k_.createElement(T,{color:z},A),K[0]=z,K[1]=A,K[2]=w;else w=K[2];let j;if(K[3]!==$)j=$&&k_.createElement(T,null," ",$),K[3]=$,K[4]=j;else j=K[4];let H;if(K[5]!==w||K[6]!==j)H=k_.createElement(u,null,w,j),K[5]=w,K[6]=j,K[7]=H;else H=K[7];let J;if(K[8]!==composed_value||K[9]!==O)J=O&&k_.createElement(u,{paddingLeft:2},k_.createElement(T,null,k_.createElement(g5,null,composed_value))),K[8]=composed_value,K[9]=O,K[10]=J;else J=K[10];let M;if(K[11]!==H||K[12]!==J)M=k_.createElement(u,{flexDirection:"column",marginTop:1},H,J),K[11]=H,K[12]=J,K[13]=M;else M=K[13];return M}var k_,Gcz;var Wa1=L(()=>{t6();Iq();O$();i6();Ra();r8();uJ();FK();Xa1();Ja1();Ma1();k_=w6(D6(),1),Gcz=new RegExp(`<${KM}\\s+teammate_id="([^"]+)"(?:\\s+color="([^"]+)")?(?:\\s+summary="([^"]+)")?>\\n?([\\s\\S]*?)\\n?<\\/${
  KM
} /* confidence: 30% */

/* original: w */ let composed_value=await Kf("resume",{
  sessionId:Y
} /* confidence: 30% */

/* original: Y */ let composed_value={
  ...Y0(_.taskId,"remote_agent",_.title,_.toolUseId),type:"remote_agent",remoteTaskType:jiz(_.remoteTaskType)?_.remoteTaskType:"remote-agent",status:"running",sessionId:_.sessionId,command:_.command,title:_.title,todoList:[],log:[],isRemoteReview:_.isRemoteReview,isUltraplan:_.isUltraplan,isLongRunning:_.isLongRunning,startTime:_.spawnedAt,pollStartedAt:Date.now(),remoteTaskMetadata:_.remoteTaskMetadata
}; /* confidence: 30% */

/* original: Vs1 */ var method_handler=B((ozK)=>{
  Object.defineProperty(ozK,"__esModule",{
    value:!0
  });
  ozK.Message=ozK.NotificationType9=ozK.NotificationType8=ozK.NotificationType7=ozK.NotificationType6=ozK.NotificationType5=ozK.NotificationType4=ozK.NotificationType3=ozK.NotificationType2=ozK.NotificationType1=ozK.NotificationType0=ozK.NotificationType=ozK.RequestType9=ozK.RequestType8=ozK.RequestType7=ozK.RequestType6=ozK.RequestType5=ozK.RequestType4=ozK.RequestType3=ozK.RequestType2=ozK.RequestType1=ozK.RequestType=ozK.RequestType0=ozK.AbstractMessageSignature=ozK.ParameterStructures=ozK.ResponseError=ozK.ErrorCodes=void 0;
  var HH6=Oh6(),Ts1;
  (function(q){
    q.ParseError=-32700,q.InvalidRequest=-32600,q.MethodNotFound=-32601,q.InvalidParams=-32602,q.InternalError=-32603,q.jsonrpcReservedErrorRangeStart=-32099,q.serverErrorStart=-32099,q.MessageWriteError=-32099,q.MessageReadError=-32098,q.PendingResponseRejected=-32097,q.ConnectionInactive=-32096,q.ServerNotInitialized=-32002,q.UnknownErrorCode=-32001,q.jsonrpcReservedErrorRangeEnd=-32000,q.serverErrorEnd=-32000
  })(Ts1||(ozK.ErrorCodes=Ts1={
    
  }));
  class ks1 extends Error{
    constructor(q,K,_){
      super(K);
      this.code=HH6.number(q)?q:Ts1.UnknownErrorCode,this.data=_,Object.setPrototypeOf(this,ks1.prototype)
    }toJson(){
      let q={
        code:this.code,message:this.message
      };
      if(this.data!==void 0)q.data=this.data;
      return q
    }
  }ozK.ResponseError=ks1;
  class Dv{
    constructor(q){
      this.kind=q
    }static is(q){
      return q===Dv.auto||q===Dv.byName||q===Dv.byPosition
    }toString(){
      return this.kind
    }
  }ozK.ParameterStructures=Dv;
  Dv.auto=new Dv("auto");
  Dv.byPosition=new Dv("byPosition");
  Dv.byName=new Dv("byName");
  class nj{
    constructor(q,K){
      this.method=q,this.numberOfParams=K
    }get parameterStructures(){
      return Dv.auto
    }
  }ozK.AbstractMessageSignature=nj;
  class LzK extends nj{
    constructor(q){
      super(q,0)
    }
  }ozK.RequestType0=LzK;
  class hzK extends nj{
    constructor(q,K=Dv.auto){
      super(q,1);
      this._parameterStructures=K
    }get parameterStructures(){
      return this._parameterStructures
    }
  }ozK.RequestType=hzK;
  class RzK extends nj{
    constructor(q,K=Dv.auto){
      super(q,1);
      this._parameterStructures=K
    }get parameterStructures(){
      return this._parameterStructures
    }
  }ozK.RequestType1=RzK;
  class SzK extends nj{
    constructor(q){
      super(q,2)
    }
  }ozK.RequestType2=SzK;
  class CzK extends nj{
    constructor(q){
      super(q,3)
    }
  }ozK.RequestType3=CzK;
  class bzK extends nj{
    constructor(q){
      super(q,4)
    }
  }ozK.RequestType4=bzK;
  class xzK extends nj{
    constructor(q){
      super(q,5)
    }
  }ozK.RequestType5=xzK;
  class IzK extends nj{
    constructor(q){
      super(q,6)
    }
  }ozK.RequestType6=IzK;
  class uzK extends nj{
    constructor(q){
      super(q,7)
    }
  }ozK.RequestType7=uzK;
  class mzK extends nj{
    constructor(q){
      super(q,8)
    }
  }ozK.RequestType8=mzK;
  class pzK extends nj{
    constructor(q){
      super(q,9)
    }
  }ozK.RequestType9=pzK;
  class BzK extends nj{
    constructor(q,K=Dv.auto){
      super(q,1);
      this._parameterStructures=K
    }get parameterStructures(){
      return this._parameterStructures
    }
  }ozK.NotificationType=BzK;
  class gzK extends nj{
    constructor(q){
      super(q,0)
    }
  }ozK.NotificationType0=gzK;
  class FzK extends nj{
    constructor(q,K=Dv.auto){
      super(q,1);
      this._parameterStructures=K
    }get parameterStructures(){
      return this._parameterStructures
    }
  }ozK.NotificationType1=FzK;
  class UzK extends nj{
    constructor(q){
      super(q,2)
    }
  }ozK.NotificationType2=UzK;
  class QzK extends nj{
    constructor(q){
      super(q,3)
    }
  }ozK.NotificationType3=QzK;
  class dzK extends nj{
    constructor(q){
      super(q,4)
    }
  }ozK.NotificationType4=dzK;
  class czK extends nj{
    constructor(q){
      super(q,5)
    }
  }ozK.NotificationType5=czK;
  class lzK extends nj{
    constructor(q){
      super(q,6)
    }
  }ozK.NotificationType6=lzK;
  class nzK extends nj{
    constructor(q){
      super(q,7)
    }
  }ozK.NotificationType7=nzK;
  class izK extends nj{
    constructor(q){
      super(q,8)
    }
  }ozK.NotificationType8=izK;
  class rzK extends nj{
    constructor(q){
      super(q,9)
    }
  }ozK.NotificationType9=rzK;
  var EzK;
  (function(q){
    function K(Y){
      let $=Y;
      return $&&HH6.string($.method)&&(HH6.string($.id)||HH6.number($.id))
    }q.isRequest=K;
    function _(Y){
      let $=Y;
      return $&&HH6.string($.method)&&Y.id===void 0
    }q.isNotification=_;
    function z(Y){
      let $=Y;
      return $&&($.result!==void 0||!!$.error)&&(HH6.string($.id)||HH6.number($.id)||$.id===null)
    }q.isResponse=z
  })(EzK||(ozK.Message=EzK={
    
  }))
} /* confidence: 70% */

/* original: J */ let composed_value={
  messages:q.messages,toolUseContext:q.toolUseContext,maxOutputTokensOverride:q.maxOutputTokensOverride,autoCompactTracking:void 0,stopHookActive:void 0,maxOutputTokensRecoveryCount:0,hasAttemptedReactiveCompact:!1,turnCount:1,pendingToolUseSummary:void 0,transition:void 0
}; /* confidence: 30% */

/* original: I */ let composed_value=await Kf("compact",{
  model:K.options.mainLoopModel
} /* confidence: 30% */

/* original: b */ let composed_value=await Kf("compact",{
  model:_.options.mainLoopModel
} /* confidence: 30% */

/* original: P */ let composed_value=await Kf("clear"); /* confidence: 30% */

/* original: D */ let composed_value={
  type:"grouped_tool_use",toolName:J.toolName,messages:X,results:W,displayMessage:P,uuid:`grouped-${P.uuid}`,timestamp:P.timestamp,messageId:J.messageId
}; /* confidence: 30% */

/* original: P */ let composed_value={
  date:H.toISOString().split("T")[0],messages:w,fullPath:A,value:H.getTime(),created:H,modified:H,firstPrompt:J,messageCount:w.length,isSidechain:!1,sessionId:$,customTitle:X,contentReplacements:j
} /* confidence: 30% */

const value_holder = parseInt(this.value) + 8; /* confidence: 70% */

/* original: _7 */ let helper_fn=await r6.beta.messages.create({
  ...b8,stream:!0
} /* confidence: 35% */

/* original: D1 */ let composed_value=(P1)=>yl8(P1,{
  transport:O6,sessionId:a,onInterrupt:V,onSetModel:y,onSetMaxThinkingTokens:E,onSetPermissionMode:R
} /* confidence: 30% */

/* original: O */ let composed_value=K.map((H)=>`- ${H.section}: ${H.change}`).join(`
`),A=await qt({
  messages:[n8({
    content:`You are editing a skill definition file. Apply the following improvements to the skill.

<current_skill_file>
${$}
</current_skill_file>

<improvements>
${composed_value}
</improvements>

Rules:
- Integrate the improvements naturally into the existing structure
- Preserve frontmatter (--- block) exactly as-is
- Preserve the overall format and style
- Do not remove existing content unless an improvement explicitly replaces it
- Output the complete updated file inside <updated_file> tags`
  } /* confidence: 30% */

/* original: L7 */ let UseRefHook=await Kf("resume",{
  sessionId:j8,agentType:C?.agentType,model:W6
} /* confidence: 46% */

var composed_value = new MessageCreateParams   // no Beta prefix â one of only 2 unprefixed
{
  
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 16000,
    Betas = ["compact-2026-01-12"],
    ContextManagement = new BetaContextManagementConfig
    {
    
        Edits = [new BetaCompact20260112Edit()],
    
  },
    Messages = messages,

}; /* confidence: 30% */

const composed_value = await getSessionInfo(sessions[0].sessionId); /* confidence: 30% */

const composed_value = await getSessionMessages(sessions[0].sessionId, {
   limit: 50 
} /* confidence: 30% */

const composed_value = await listSessions({
   limit: 20, offset: 0 
} /* confidence: 30% */

const composed_value = sessions[0]?.composed_value; /* confidence: 30% */

const composed_value = await getSessionInfo(sessionId); /* confidence: 30% */

const composed_value = await getSessionMessages(sessionId, {
   limit: 50, offset: 0 
} /* confidence: 30% */

const composed_value = await client.messages.create({
  
  model: "{{OPUS_ID}}",
  max_tokens: 16000,
  system: [
    {
    
      type: "text",
      text: "You are an expert on this large document...",
      cache_control: {
       type: "ephemeral" 
    }, // default TTL is 5 minutes
    
  },
  ],
  messages: [{
     role: "user", content: "Summarize the key points" 
  }],

} /* confidence: 30% */

const composed_value = await client.messages.create({
  
  model: "{{OPUS_ID}}",
  max_tokens: 16000,
  messages: messages,

} /* confidence: 30% */

const composed_value = await client.beta.messages.create({
  
    betas: ["compact-2026-01-12"],
    model: "{{OPUS_ID}}",
    max_tokens: 16000,
    messages,
    context_management: {
    
      edits: [{
       type: "compact_20260112" 
    }],
    
  },
  
} /* confidence: 30% */

const typed_entity = response.content.find(
    (b): b is Anthropic.Beta.BetaTextBlock => b.type === "text",
  ); /* confidence: 70% */

const McpSamplingHandler = await client.messages.countTokens({
  
  model: "{{OPUS_ID}}",
  messages: messages,
  system: system,

} /* confidence: 30% */

const composed_value = countResponse.input_tokens * 0.000005; /* confidence: 30% */

const OPUS_ID_user_Whatstheweatherin = client.beta.messages.toolRunner({
  
  model: "{{OPUS_ID}}",
  max_tokens: 64000,
  tools: [getWeather],
  messages: [
    {
     role: "user", content: "What's the weather in Paris and London?" 
  },
  ],
  stream: true,

} /* confidence: 65% */

const composed_value = await stream.composed_value(); /* confidence: 30% */

const composed_value = await client.messages.create({
  
    model: "{{OPUS_ID}}",
    max_tokens: 16000,
    tools: tools,
    messages: messages,
  
} /* confidence: 30% */

const typed_entity = response.content.filter(
    (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
  ); /* confidence: 70% */

const McpSamplingHandler = client.messages.McpSamplingHandler({
  
    model: "{{OPUS_ID}}",
    max_tokens: 64000,
    tools,
    messages,
  
} /* confidence: 30% */

const composed_value = await stream.finalMessage(); /* confidence: 30% */

const typed_entity = message.content.filter(
    (b): b is Anthropic.ToolUseBlock => b.type === "tool_use",
  ); /* confidence: 70% */

/* original: lJ7 */ function u_function_function_NFC_NFC_cl(){
  let q="";
  if(typeof process<"u"&&typeof process.cwd==="function"&&typeof QJ7==="function"){
    let _=jO5();
    try{
      q=QJ7(_).normalize("NFC")
    }catch{
      q=_.normalize("NFC")
    }
  }return{
    originalCwd:q,projectRoot:q,totalCostUSD:0,totalAPIDuration:0,totalAPIDurationWithoutRetries:0,totalToolDuration:0,turnHookDurationMs:0,turnToolDurationMs:0,turnClassifierDurationMs:0,turnToolCount:0,turnHookCount:0,turnClassifierCount:0,startTime:Date.now(),lastInteractionTime:Date.now(),totalLinesAdded:0,totalLinesRemoved:0,hasUnknownModelCost:!1,cwd:q,modelUsage:{
      
    },mainLoopModelOverride:void 0,initialMainLoopModel:null,modelStrings:null,isInteractive:!1,kairosActive:!1,strictToolResultPairing:!1,memoryToggledOff:!1,sdkAgentProgressSummariesEnabled:!1,userMsgOptIn:!1,clientType:"cli",sessionSource:void 0,questionPreviewFormat:void 0,sessionIngressToken:void 0,oauthTokenFromFd:void 0,apiKeyFromFd:void 0,flagSettingsPath:void 0,flagSettingsInline:null,allowedSettingSources:["userSettings","projectSettings","localSettings","flagSettings","policySettings"],meter:null,sessionCounter:null,locCounter:null,prCounter:null,commitCounter:null,costCounter:null,tokenCounter:null,codeEditToolDecisionCounter:null,activeTimeCounter:null,statsStore:null,sessionId:cx6(),parentSessionId:void 0,loggerProvider:null,eventLogger:null,meterProvider:null,tracerProvider:null,agentColorMap:new Map,agentColorIndex:0,lastAPIRequest:null,lastAPIRequestMessages:null,lastClassifierRequests:null,cachedClaudeMdContent:null,inMemoryErrorLog:[],inlinePlugins:[],chromeFlagOverride:void 0,useCoworkPlugins:!1,sessionBypassPermissionsMode:!1,scheduledTasksEnabled:!1,sessionCronTasks:[],sessionCreatedTeams:new Set,sessionTrustAccepted:!1,sessionPersistenceDisabled:!1,hasExitedPlanMode:!1,needsPlanModeExitAttachment:!1,needsAutoModeExitAttachment:!1,lspRecommendationShownThisSession:!1,initJsonSchema:null,registeredHooks:null,planSlugCache:new Map,teleportedSessionInfo:null,invokedSkills:new Map,slowOperations:[],sdkBetas:void 0,mainThreadAgentType:void 0,isRemoteMode:!1,...!1,directConnectServerUrl:void 0,systemPromptSectionCache:new Map,lastEmittedDate:null,additionalDirectoriesForClaudeMd:[],allowedChannels:[],hasDevChannels:!1,sessionProjectDir:null,promptCache1hAllowlist:null,afkModeHeaderLatched:null,fastModeHeaderLatched:null,cacheEditingHeaderLatched:null,thinkingClearLatched:null,promptId:null,lastMainRequestId:void 0,lastApiCompletionTimestamp:null,pendingPostCompaction:!1
  }
} /* confidence: 65% */

/* original: XI6 */ function helper_fn(q){
  G8.teleportedSessionInfo={
    isTeleported:!0,hasLoggedFirstMessage:!1,sessionId:q.sessionId
  } /* confidence: 35% */

/* original: E81 */ function string_datetime_string_format(q,K){
  return new q({
    type:"string",format:"datetime",check:"string_format",offset:!1,local:!1,precision:null,...xq(K)
  } /* confidence: 65% */

/* original: X11 */ function helper_fn(q){
  return E81(MY8,q)
} /* confidence: 35% */

/* original: j31 */ function helper_fn(q,K=[],_=LB6.DEFAULT){
  let z={
    type:"array",offset:-1,length:-1,children:[],parent:void 0
  };
   /* confidence: 35% */

/* original: EB7 */ function helper_fn(q){
  let{
    type:K,body:_,resolve:z,stream:Y,length:$
  } /* confidence: 35% */

/* original: $l6 */ function AnthropicSDKERROR_AnthropicSDK(){
  return{
    error:(q,...K)=>console.error("[Anthropic SDK ERROR]",q,...K),warn:(q,...K)=>console.error("[Anthropic SDK WARN]",q,...K),info:(q,...K)=>console.error("[Anthropic SDK INFO]",q,...K),debug:(q,...K)=>console.error("[Anthropic SDK DEBUG]",q,...K)
  }
} /* confidence: 65% */

/* original: a7_ */ function function_error_warn_info_debug(q,K){
  if(q<ai.DiagLogLevel.NONE)q=ai.DiagLogLevel.NONE;
  else if(q>ai.DiagLogLevel.ALL)q=ai.DiagLogLevel.ALL;
  K=K||{
    
  };
  function _(z,Y){
    let $=K[z];
    if(typeof $==="function"&&q>=Y)return $.bind(K);
    return function(){
      
    }
  }return{
    error:_("error",ai.DiagLogLevel.ERROR),warn:_("warn",ai.DiagLogLevel.WARN),info:_("info",ai.DiagLogLevel.INFO),debug:_("debug",ai.DiagLogLevel.DEBUG),verbose:_("verbose",ai.DiagLogLevel.VERBOSE)
  }
} /* confidence: 65% */

/* original: QS_ */ function helper_fn({
  offset:q,props:K
} /* confidence: 35% */

/* original: yC_ */ function helper_fn(q,K,{
  offset:_,start:z,value:Y,end:$
} /* confidence: 35% */

/* original: xi6 */ function number_string(q){
  if(typeof q==="number")return[q,q+1];
  if(Array.isArray(q))return q.length===2?q:[q[0],q[1]];
  let{
    offset:K,source:_
  }=q;
  return[K,K+(typeof _==="string"?_.length:1)]
} /* confidence: 65% */

/* original: gC_ */ function helper_fn(q,K){
  let{
    implicitKey:_=!1,indent:z,inFlow:Y=!1,offset:$=-1,type:O="PLAIN"
  } /* confidence: 35% */

/* original: OC1 */ function typed_entity(q,K,_){
  switch(q.type){
    case"scalar":case"double-quoted-scalar":case"single-quoted-scalar":q.type=_,q.source=K;
    break;
    case"block-scalar":{
      let z=q.props.slice(1),Y=K.length;
      if(q.props[0].type==="block-scalar-header")Y-=q.props[0].source.length;
      for(let $ of z)$.offset+=Y;
      delete q.props,Object.assign(q,{
        type:_,source:K,end:z
      });
      break
    }case"block-map":case"block-seq":{
      let Y={
        type:"newline",offset:q.offset+K.length,indent:q.indent,source:`
`
      };
      delete q.items,Object.assign(q,{
        type:_,source:K,end:[Y]
      });
      break
    }default:{
      let z="indent"in q?q.indent:-1,Y="end"in q&&Array.isArray(q.end)?q.end.filter(($)=>$.type==="space"||$.type==="comment"||$.type==="newline"):[];
      for(let $ of Object.keys(q))if($!=="type"&&$!=="offset")delete q[$];
      Object.assign(q,{
        type:_,indent:z,source:K,end:Y
      })
    }
  } /* confidence: 70% */

/* original: ug1 */ function keepdoingthiseveryday_setthisu(q){
  return`Schedule a prompt to be enqueued at a future time. Use for both recurring schedules and one-shot reminders.

Uses standard 5-field cron in the user's local timezone: minute hour day-of-month month day-of-week. "0 9 * * *" means 9am local â no timezone conversion needed.

## One-shot tasks (recurring: false)

For "remind me at X" or "at <time>, do Y" requests â fire once then auto-delete.
Pin minute/hour/day-of-month/month to specific values:
  "remind me at 2:30pm today to check the deploy" â cron: "30 14 <today_dom> <today_month> *", recurring: false
  "tomorrow morning, run the smoke test" â cron: "57 8 <tomorrow_dom> <tomorrow_month> *", recurring: false

## Recurring jobs (recurring: true, the default)

For "every N minutes" / "every hour" / "weekdays at 9am" requests:
  "*/5 * * * *" (every 5 min), "0 * * * *" (hourly), "0 9 * * 1-5" (weekdays at 9am local)

## Avoid the :00 and :30 minute marks when the task allows it

Every user who asks for "9am" gets \`0 9\`, and every user who asks for "hourly" gets \`0 *\` â which means requests from across the planet land on the API at the same instant. When the user's request is approximate, pick a minute that is NOT 0 or 30:
  "every morning around 9" â "57 8 * * *" or "3 9 * * *" (not "0 9 * * *")
  "hourly" â "7 * * * *" (not "0 * * * *")
  "in an hour or so, remind me to..." â pick whatever minute you land on, don't round

Only use minute 0 or 30 when the user names that exact time and clearly means it ("at 9:00 sharp", "at half past", coordinating with a meeting). When in doubt, nudge a few minutes early or late â the user will not notice, and the fleet will.

${q?`## Durability

By default (durable: false) the job lives only in this Claude session â nothing is written to disk, and the job is gone when Claude exits. Pass durable: true to write to .claude/scheduled_tasks.json so the job survives restarts. Only use durable: true when the user explicitly asks for the task to persist ("keep doing this every day", "set this up permanently"). Most "remind me in 5 minutes" / "check back in an hour" requests should stay session-only.`:`## Session-only

Jobs live only in this Claude session â nothing is written to disk, and the job is gone when Claude exits.`}

## Runtime behavior

Jobs only fire while the REPL is idle (not mid-query). ${q?"Durable jobs persist to .claude/scheduled_tasks.json and survive session restarts â on next launch they resume automatically. One-shot durable tasks that were missed while the REPL was closed are surfaced for catch-up. Session-only jobs die with the process. ":""}The scheduler adds a small deterministic jitter on top of whatever you pick: recurring tasks fire up to 10% of their period late (max 15 min); one-shot tasks landing on :00 or :30 fire up to 90 s early. Picking an off-minute is still the bigger lever.

Recurring tasks auto-expire after ${O46} days â they fire one final time, then are deleted. This bounds session lifetime. Tell the user about the ${O46}-day limit when scheduling recurring jobs.

Returns a job ID you can pass to ${A46}.`
} /* confidence: 65% */

/* original: uTz */ function value_holder(q,K){
  return q.dataPoints.map((_)=>{
    let z=_.value;
    return{
      attributes:(0,tt6.toAttributes)(_.attributes),count:z.count,min:z.min,max:z.max,sum:z.sum,positive:{
        offset:z.positive.offset,bucketCounts:z.positive.bucketCounts
      },negative:{
        offset:z.negative.offset,bucketCounts:z.negative.bucketCounts
      },scale:z.scale,zeroCount:z.zeroCount,startTimeUnixNano:K.encodeHrTime(_.startTime),timeUnixNano:K.encodeHrTime(_.endTime)
    }
  } /* confidence: 70% */

/* original: Kf */ function helper_fn(q,{
  sessionId:K,agentType:_,model:z,forceSyncExecution:Y
} /* confidence: 35% */

/* original: O2K */ function helper_fn(q,K,{
  limit:_,offset:z
} /* confidence: 35% */

/* original: RWK */ function helper_fn({
  messages:q,queryParams:K,description:_,setAppState:z,agentDefinition:Y
} /* confidence: 35% */

/* original: RDK */ function helper_fn(){
  let q={
    messages:[],searchCount:0,readFilePaths:new Set,readOperationCount:0,listCount:0,toolUseIds:new Set,memorySearchCount:0,memoryReadFilePaths:new Set,memoryWriteCount:0,nonMemSearchArgs:[],latestDisplayHint:void 0,hookTotalMs:0,hookCount:0,hookInfos:[]
  };
   /* confidence: 35% */

/* original: CDK */ function helper_fn(q,K){
  let _=[],z=RDK(),Y=[];
   /* confidence: 35% */

/* original: $ */ function composed_value(){
  if(z.messages.length===0)return;
  _.push(p9Y(z));
  for(let O of Y)_.push(O);
  Y=[],z=RDK()
} /* confidence: 30% */

/* original: O0K */ function helper_fn({
  messages:q,summaryRequest:K,appState:_,context:z,preCompactTokenCount:Y,cacheSafeParams:$,stripNonEssential:O=!1
} /* confidence: 35% */

/* original: i$Y */ function helper_fn({
  model:q,messages:K,tools:_,betas:z,containsThinking:Y
} /* confidence: 35% */

/* original: i0K */ function helper_fn({
  file_path:q,offset:K,limit:_,pages:z
} /* confidence: 35% */

/* original: wVK */ function helper_fn({
  abortSignal:q,messages:K,initialDescription:_,onDone:z,backgroundTasks:Y={
    
  } /* confidence: 35% */

/* original: bPY */ function helper_fn(q){
  let K=Y6(73),{
    messages:_,onDone:z
  } /* confidence: 35% */

/* original: MbK */ function helper_fn(q,K,_=!1){
  if(_)return{
    messages:q
  };
   /* confidence: 35% */

/* original: is */ function helper_fn(q,K,_){
  return Array.from({
    length:_
  },()=>({
    pose:q,offset:K
  }))
} /* confidence: 35% */

/* original: eVY */ function helper_fn({
  messages:q,start:K,end:_,offsets:z,getItemTop:Y,getItemElement:$,scrollRef:O
} /* confidence: 35% */

/* original: uy */ function helper_fn(q){
  let K=q?{
    originalCwd:q.originalCwd,worktreePath:q.worktreePath,worktreeName:q.worktreeName,worktreeBranch:q.worktreeBranch,originalBranch:q.originalBranch,originalHeadCommit:q.originalHeadCommit,sessionId:q.sessionId,tmuxSessionName:q.tmuxSessionName,hookBased:q.hookBased
  } /* confidence: 35% */

/* original: FIY */ function helper_fn({
  hook:q,messages:K,hookName:_,toolUseID:z,hookEvent:Y,timeoutMs:$,signal:O
} /* confidence: 35% */

/* original: qt */ function helper_fn({
  messages:q,systemPrompt:K,thinkingConfig:_,tools:z,signal:Y,options:$
} /* confidence: 35% */

/* original: YpY */ function _w(){
  let{
    EXTERNAL_PERMISSION_MODES:q
  }=await Promise.resolve().then(() => (bB6(),Z31)),K=q.join(", "),_=await hcK(),z=_?`  --spawn <mode>                   Spawn mode: same-dir, worktree, session
                                   (default: same-dir)
  --capacity <N>                   Max concurrent sessions in worktree or
                                   same-dir mode (default: ${LcK})
  --[no-]create-session-in-dir     Pre-create a session in the current
                                   directory; in worktree mode this session
                                   stays in cwd while on-demand sessions get
                                   isolated worktrees (default: on)
`:"",O=`
Remote Control - Connect your local environment to claude.ai/code

USAGE
  claude remote-control [options]
OPTIONS
  --name <name>                    Name for the session (shown in claude.ai/code)
  --permission-mode <mode>         Permission mode for spawned sessions
                                   (${K})
  --debug-file <path>              Write debug logs to file
  -v, --verbose                    Enable verbose output
  -h, --help                       Show this help
${z}
DESCRIPTION
  Remote Control allows you to control sessions on your local device from
  claude.ai/code (https://claude.ai/code). Run this command in the
  directory you want to work in, then connect from the Claude app or web.
${_?`
  Remote Control runs as a persistent server that accepts multiple concurrent
  sessions in the current directory. One session is pre-created on start so
  you have somewhere to type immediately. Use --spawn=worktree to isolate
  each on-demand session in its own git worktree, or --spawn=session for
  the classic single-session mode (exits when that session ends). Press 'w'
  during runtime to toggle between same-dir and worktree.
`:""}
NOTES
  - You must be logged in with a Claude account that has a subscription
  - Run \`claude\` first in the directory to accept the workspace trust dialog
${_?`  - Worktree mode requires a git repository or WorktreeCreate/WorktreeRemove hooks
`:""}`;
  console.log(O)
} /* confidence: 65% */

/* original: KO7 */ function helper_fn({
  messages:q,onPreRestore:K,onRestoreMessage:_,onRestoreCode:z,onSummarize:Y,onClose:$,preselectedMessage:O
} /* confidence: 35% */

/* original: NiK */ function helper_fn({
  toolName:q,toolInput:K,toolDescription:_,messages:z,signal:Y
} /* confidence: 35% */

/* original: OUY */ function McpSamplingHandler(q){
  return NiK({
    toolName:q.toolName,toolInput:q.toolInput,toolDescription:q.toolDescription,messages:q.messages,signal:new AbortController().signal
  } /* confidence: 30% */

/* original: ztK */ function helper_fn(q,K,_,z=!1,Y=!1){
  return{
    sessionId:q,getAccessToken:K,orgUuid:_,hasInitialPrompt:z,viewerOnly:Y
  } /* confidence: 35% */

/* original: pnY */ function McpSamplingHandler(q,K){
  return{
    messages:q.messages+K.messages,errors:q.errors+K.errors
  } /* confidence: 30% */

/* original: z */ function composed_value(){
  let[A,w]=P35.useState("validating");
  return _=w,dw.createElement(W35,{
    currentStep:A,sessionId:K
  })
} /* confidence: 30% */

/* original: S35 */ function helper_fn(q){
  let K=Y6(5),{
    messages:_
  } /* confidence: 35% */

/* original: md7 */ class entity_class{
  sessions=[];
   /* confidence: 45% */

/* original: Bi7 */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: ya7 */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: vt7 */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: ot7 */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: K8q */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: C4q */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: Dzq */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: xOq */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: lwq */ class entity_class{
  trace(){
    
  }debug(){
    
  }info(){
    
  }warn(){
    
  }error(){
    
  }
} /* confidence: 45% */

/* original: AN */ class McpLoggingHandler{
  constructor(q,K,_){
    this.level=vH.Info;
    let z=()=>{
      return
    },Y=q||McpLoggingHandler.createDefaultLoggerOptions();
    this.localCallback=Y.loggerCallback||z,this.piiLoggingEnabled=Y.piiLoggingEnabled||!1,this.level=typeof Y.logLevel==="number"?Y.logLevel:vH.Info,this.correlationId=Y.correlationId||Q1.EMPTY_STRING,this.packageName=K||Q1.EMPTY_STRING,this.packageVersion=_||Q1.EMPTY_STRING
  }static createDefaultLoggerOptions(){
    return{
      loggerCallback:()=>{
        
      },piiLoggingEnabled:!1,logLevel:vH.Info
    }
  }clone(q,K,_){
    return new McpLoggingHandler({
      loggerCallback:this.localCallback,piiLoggingEnabled:this.piiLoggingEnabled,logLevel:this.level,correlationId:_||this.correlationId
    },q,K)
  }logMessage(q,K){
    if(K.logLevel>this.level||!this.piiLoggingEnabled&&K.containsPii)return;
    let Y=`${`[${
      new Date().toUTCString()
    }] : [${
      K.correlationId||this.correlationId||""
    }]`} : ${this.packageName}@${this.packageVersion} : ${vH[K.logLevel]} - ${q}`;
    this.executeCallback(K.logLevel,Y,K.containsPii||!1)
  }executeCallback(q,K,_){
    if(this.localCallback)this.localCallback(q,K,_)
  }error(q,K){
    this.logMessage(q,{
      logLevel:vH.Error,containsPii:!1,correlationId:K||Q1.EMPTY_STRING
    })
  }errorPii(q,K){
    this.logMessage(q,{
      logLevel:vH.Error,containsPii:!0,correlationId:K||Q1.EMPTY_STRING
    })
  }warning(q,K){
    this.logMessage(q,{
      logLevel:vH.Warning,containsPii:!1,correlationId:K||Q1.EMPTY_STRING
    })
  }warningPii(q,K){
    this.logMessage(q,{
      logLevel:vH.Warning,containsPii:!0,correlationId:K||Q1.EMPTY_STRING
    })
  }info(q,K){
    this.logMessage(q,{
      logLevel:vH.Info,containsPii:!1,correlationId:K||Q1.EMPTY_STRING
    })
  }infoPii(q,K){
    this.logMessage(q,{
      logLevel:vH.Info,containsPii:!0,correlationId:K||Q1.EMPTY_STRING
    })
  }verbose(q,K){
    this.logMessage(q,{
      logLevel:vH.Verbose,containsPii:!1,correlationId:K||Q1.EMPTY_STRING
    })
  }verbosePii(q,K){
    this.logMessage(q,{
      logLevel:vH.Verbose,containsPii:!0,correlationId:K||Q1.EMPTY_STRING
    })
  }trace(q,K){
    this.logMessage(q,{
      logLevel:vH.Trace,containsPii:!1,correlationId:K||Q1.EMPTY_STRING
    })
  }tracePii(q,K){
    this.logMessage(q,{
      logLevel:vH.Trace,containsPii:!0,correlationId:K||Q1.EMPTY_STRING
    })
  }isPiiLoggingEnabled(){
    return this.piiLoggingEnabled||!1
  }
} /* confidence: 44% */

/* original: gIq */ class DiagComponentLogger_debug_erro{
  constructor(q){
    this._namespace=q.namespace||"DiagComponentLogger"
  }debug(...q){
    return Vl6("debug",this._namespace,q)
  }error(...q){
    return Vl6("error",this._namespace,q)
  }info(...q){
    return Vl6("info",this._namespace,q)
  }warn(...q){
    return Vl6("warn",this._namespace,q)
  }verbose(...q){
    return Vl6("verbose",this._namespace,q)
  }
} /* confidence: 65% */

/* original: YI4 */ class debug_debug_info_warn_error{
  silly(q,...K){
    N(zt6(q,...K),{
      level:"debug"
    })
  }debug(q,...K){
    N(zt6(q,...K),{
      level:"debug"
    })
  }info(q,...K){
    N(zt6(q,...K),{
      level:"info"
    })
  }warn(q,...K){
    N(zt6(q,...K),{
      level:"warn"
    })
  }error(q,...K){
    N(zt6(q,...K),{
      level:"error"
    })
  }
} /* confidence: 65% */

/* original: Cd1 */ class error_warn{
  error(q,...K){
    j6(Error(q)),N(`[3P telemetry] OTEL diag error: ${q}`,{
      level:"error"
    })
  }warn(q,...K){
    j6(Error(q)),N(`[3P telemetry] OTEL diag warn: ${q}`,{
      level:"warn"
    })
  }info(q,...K){
    return
  }debug(q,...K){
    return
  }verbose(q,...K){
    return
  }
} /* confidence: 65% */

/* original: Dv */ class entity_class{
  constructor(q){
    this.kind=q
  }static is(q){
    return q===entity_class.auto||q===entity_class.byName||q===entity_class.byPosition
  }toString(){
    return this.kind
  }
} /* confidence: 45% */

/* original: xA7 */ class entity_class{
  sessionId;
   /* confidence: 45% */

