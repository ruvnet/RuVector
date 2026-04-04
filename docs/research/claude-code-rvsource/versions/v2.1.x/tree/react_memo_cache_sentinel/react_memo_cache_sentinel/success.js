// Module: success

/* original: xK */ var WriteTool="Write"; /* confidence: 31% */

/* original: V9Y */ function helper_fn(q,K){
  if(q!==xK&&q!==N4)return!1;
   /* confidence: 35% */

/* original: L9Y */ function helper_fn(q,K){
  if(q!==xK&&q!==N4)return!1;
   /* confidence: 35% */

/* original: KfK */ function WriteTool(q,K){
  switch(q){
    case pq:{
      let _=uz.inputSchema.safeParse(K);
      return _.success?_.data.file_path:null
    }case N4:{
      let _=Vp8().safeParse(K);
      return _.success?_.data.file_path:null
    }case xK:{
      let _=AP.inputSchema.safeParse(K);
      return _.success?_.data.file_path:null
    }default:return null
  }
} /* confidence: 31% */

/* original: HZK */ function typed_entity(q){
  if(q.type!=="tool_use"||q.name!==N4&&q.name!==xK)return;
   /* confidence: 70% */

