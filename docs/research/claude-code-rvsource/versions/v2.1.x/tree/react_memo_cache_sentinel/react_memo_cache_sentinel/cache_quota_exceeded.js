// Module: cache_quota_exceeded

/* original: aZ1 */ var cache_quota_exceeded_cache_err="cache_quota_exceeded",LW8="cache_error_unknown"; /* confidence: 65% */

/* original: xGq */ function error_handler(q){
  if(!(q instanceof Error))return new yd6(LW8);
  if(q.name==="QuotaExceededError"||q.name==="NS_ERROR_DOM_QUOTA_REACHED"||q.message.includes("exceeded the quota"))return new yd6(aZ1);
  else return new yd6(q.name,q.message)
} /* confidence: 95% */

