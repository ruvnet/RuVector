// Module: constants

/* original: Rz */ var Rz=hZ.ed25519;

/* original: _ */ var composed_value=new BN(Rz.constants.PUBLIC_KEY_BYTE_LENGTH),z=new BN(Rz.constants.PRIVATE_KEY_BYTE_LENGTH); /* confidence: 30% */

/* original: _ */ var composed_value=new BN(Rz.constants.PUBLIC_KEY_BYTE_LENGTH); /* confidence: 30% */

/* original: Y */ var composed_value=new BN(Rz.constants.SIGN_BYTE_LENGTH+K.length); /* confidence: 30% */

/* original: $ */ var composed_value=new BN(Rz.constants.SIGN_BYTE_LENGTH); /* confidence: 30% */

/* original: Y */ var composed_value=new BN(Rz.constants.SIGN_BYTE_LENGTH+K.length),$=new BN(Rz.constants.SIGN_BYTE_LENGTH+K.length),O; /* confidence: 30% */

/* original: $ */ var composed_value=new BN(Rz.constants.HASH_BYTE_LENGTH); /* confidence: 30% */

/* original: H */ var composed_value=Yo6(q.subarray(32),_+32); /* confidence: 30% */

/* original: J */ var composed_value=Yo6(q,_+64); /* confidence: 30% */

/* original: j */ var composed_value=Yo6(q,_); /* confidence: 30% */

/* original: Yo6 */ function binary_u_binary(q,K){
  var _=hZ.md.sha512.create(),z=new tI1(q);
  _.update(z.getBytes(K),"binary");
  var Y=_.digest().getBytes();
  if(typeof Buffer<"u")return Buffer.from(Y,"binary");
  var $=new BN(Rz.constants.HASH_BYTE_LENGTH);
  for(var O=0;
  O<64;
  ++O)$[O]=Y.charCodeAt(O);
  return $
} /* confidence: 65% */

