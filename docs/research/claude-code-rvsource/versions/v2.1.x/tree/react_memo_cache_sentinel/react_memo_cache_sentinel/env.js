// Module: env

/* original: DSq */ function utility_fn(){
  return!!(process.env.CLOUD_RUN_JOB||process.env.FUNCTION_NAME||process.env.K_SERVICE)
} /* confidence: 40% */

/* original: Nt9 */ function helper_fn(){
  return DSq()||GSq()
} /* confidence: 35% */

