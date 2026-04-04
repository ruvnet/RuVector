// Module: pubea5604404508cdd34afb69e6f42a05bc

/* original: EP_ */ var httpshttpintakelogsus5datadogh="https://http-intake.logs.us5.datadoghq.com/api/v2/logs",LP_="pubea5604404508cdd34afb69e6f42a05bc",hP_=15000,RP_=100,SP_=5000,CP_,bP_,tl6,Hr=null,S08=null,IP_,uP_=30,mP_; /* confidence: 65% */

/* original: NE1 */ function ContentType_applicationjson_DD(){
  if(tl6.length===0)return;
  let q=tl6;
  tl6=[];
  try{
    await O1.post(EP_,q,{
      headers:{
        "Content-Type":"application/json","DD-API-KEY":LP_
      },timeout:SP_
    })
  }catch(K){
    j6(K)
  }
} /* confidence: 65% */

/* original: K76 */ function helper_fn(){
  if(Hr)clearTimeout(Hr),Hr=null;
  await NE1()
} /* confidence: 35% */

