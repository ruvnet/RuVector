// Module: sensitive

/* original: TM1 */ var TM1="***SensitiveInformation***";

/* original: VM1 */ function helper_fn(q,K){
  if(K==null)return K;
  let _=C39.NormalizedSchema.of(q);
  if(_.getMergedTraits().sensitive)return TM1;
  if(_.isListSchema()){
    if(!!_.getValueSchema().getMergedTraits().sensitive)return TM1
  }else if(_.isMapSchema()){
    if(!!_.getKeySchema().getMergedTraits().sensitive||!!_.getValueSchema().getMergedTraits().sensitive)return TM1
  }else if(_.isStructSchema()&&typeof K==="object"){
    let z=K,Y={
      
    };
    for(let[$,O]of _.structIterator())if(z[$]!=null)Y[$]=helper_fn(O,z[$]);
    return Y
  }return K
} /* confidence: 35% */

