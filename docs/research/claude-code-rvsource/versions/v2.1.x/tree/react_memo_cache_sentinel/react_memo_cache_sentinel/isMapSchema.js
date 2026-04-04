// Module: isMapSchema

/* original: Cw1 */ var Cw1="***SensitiveInformation***";

/* original: xw1 */ function helper_fn(q,K){
  if(K==null)return K;
  let _=DD3.NormalizedSchema.of(q);
  if(_.getMergedTraits().sensitive)return Cw1;
  if(_.isListSchema()){
    if(!!_.getValueSchema().getMergedTraits().sensitive)return Cw1
  }else if(_.isMapSchema()){
    if(!!_.getKeySchema().getMergedTraits().sensitive||!!_.getValueSchema().getMergedTraits().sensitive)return Cw1
  }else if(_.isStructSchema()&&typeof K==="object"){
    let z=K,Y={
      
    };
    for(let[$,O]of _.structIterator())if(z[$]!=null)Y[$]=helper_fn(O,z[$]);
    return Y
  }return K
} /* confidence: 35% */

