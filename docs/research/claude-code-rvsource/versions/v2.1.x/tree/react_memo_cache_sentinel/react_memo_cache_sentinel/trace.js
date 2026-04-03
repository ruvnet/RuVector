// Module: trace

/* original: _ */ let composed_value=Hv(),Y=c6(process.env.OTEL_LOG_USER_PROMPTS)?q:"<REDACTED>"; /* confidence: 30% */

/* original: Y */ let composed_value=Hv(),$=r46.getStore(),O=YE6("tool",{
  tool_name:q,...K
} /* confidence: 30% */

/* original: K */ let composed_value=Hv(),_=Ya.getStore(),z=YE6("tool.blocked_on_user"),Y=_?qw.trace.setSpan(qw.context.active(),_.span):qw.context.active(),$=composed_value.startSpan("claude_code.tool.blocked_on_user",{
  attributes:z
} /* confidence: 30% */

/* original: G */ let composed_value=CQ4(H,J,W.length,f); /* confidence: 30% */

/* original: Hv */ function comanthropicclaude_codetracing(){
  return qw.trace.getTracer("com.anthropic.claude_code.tracing","1.0.0")
} /* confidence: 65% */

/* original: RQ4 */ function helper_fn(){
  if(!nm())return qw.trace.getActiveSpan()||Hv().startSpan("dummy");
   /* confidence: 35% */

/* original: CQ4 */ function helper_fn(q,K,_,z){
  if(!pH())return qw.trace.getActiveSpan()||Hv().startSpan("dummy");
   /* confidence: 35% */

