// Module: resolve

/* original: Qi8 */ var Qi8={
  
};

/* original: X1$ */ function default_force_failed(q,K){
  let{
    setup:_
  }=await Promise.resolve().then(() => (di8(),Qi8));
  await _(w1$(),"default",!1,!1,void 0,!1);
  let{
    install:z
  }=await Promise.resolve().then(() => (b35(),C35));
  await new Promise((Y)=>{
    let $=[];
    if(q)$.push(q);
    if(K.force)$.push("--force");
    z.call((O)=>{
      Y(),process.exit(O.includes("failed")?1:0)
    },{
      
    },$)
  })
} /* confidence: 65% */

