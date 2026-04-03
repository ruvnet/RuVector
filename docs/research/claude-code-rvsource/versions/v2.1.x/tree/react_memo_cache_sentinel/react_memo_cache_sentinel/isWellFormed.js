// Module: isWellFormed

/* original: Nu5 */ var function_function=typeof String.prototype.toWellFormed==="function",yu5=typeof String.prototype.isWellFormed==="function"; /* confidence: 65% */

/* original: mI7 */ function helper_fn(q){
  return Nu5?`${q}`.toWellFormed():aI5.toUSVString(q)
} /* confidence: 35% */

/* original: Eu5 */ function helper_fn(q){
  return yu5?`${q}`.isWellFormed():mI7(q)===`${q}`
} /* confidence: 35% */

