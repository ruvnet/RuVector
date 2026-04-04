// Module: UTCTIME

/* original: Cc_ */ var 19500101T000000Z_20500101T0000=new Date("1950-01-01T00:00:00Z"),bc_=new Date("2050-01-01T00:00:00Z"); /* confidence: 65% */

/* original: lj4 */ function helper_fn(q){
  if(q>=Cc_&&q<bc_)return z8.create(z8.Class.UNIVERSAL,z8.Type.UTCTIME,!1,z8.dateToUtcTime(q));
  else return z8.create(z8.Class.UNIVERSAL,z8.Type.GENERALIZEDTIME,!1,z8.dateToGeneralizedTime(q))
} /* confidence: 35% */

