// Module: instance

/* original: br1 */ var composed_value=L(()=>{
  T8();
  Zj6=Yd.getInstance()
} /* confidence: 30% */

/* original: Io8 */ function utility_fn(){
  return G8.activeTimeCounter
} /* confidence: 40% */

/* original: Yd */ class user_cli{
  activeOperations=new Set;
  lastUserActivityTime=0;
  lastCLIRecordedTime;
  isCLIActive=!1;
  USER_ACTIVITY_TIMEOUT_MS=5000;
  getNow;
  getActiveTimeCounter;
  static instance=null;
  constructor(q){
    this.getNow=q?.getNow??(()=>Date.now()),this.getActiveTimeCounter=q?.getActiveTimeCounter??Io8,this.lastCLIRecordedTime=this.getNow()
  }static getInstance(){
    if(!user_cli.instance)user_cli.instance=new user_cli;
    return user_cli.instance
  }static resetInstance(){
    user_cli.instance=null
  }static createInstance(q){
    return user_cli.instance=new user_cli(q),user_cli.instance
  }recordUserActivity(){
    if(!this.isCLIActive&&this.lastUserActivityTime!==0){
      let K=(this.getNow()-this.lastUserActivityTime)/1000;
      if(K>0){
        let _=this.getActiveTimeCounter();
        if(_){
          let z=this.USER_ACTIVITY_TIMEOUT_MS/1000;
          if(K<z)_.add(K,{
            type:"user"
          })
        }
      }
    }this.lastUserActivityTime=this.getNow()
  }startCLIActivity(q){
    if(this.activeOperations.has(q))this.endCLIActivity(q);
    let K=this.activeOperations.size===0;
    if(this.activeOperations.add(q),K)this.isCLIActive=!0,this.lastCLIRecordedTime=this.getNow()
  }endCLIActivity(q){
    if(this.activeOperations.delete(q),this.activeOperations.size===0){
      let K=this.getNow(),_=(K-this.lastCLIRecordedTime)/1000;
      if(_>0){
        let z=this.getActiveTimeCounter();
        if(z)z.add(_,{
          type:"cli"
        })
      }this.lastCLIRecordedTime=K,this.isCLIActive=!1
    }
  }async trackOperation(q,K){
    this.startCLIActivity(q);
    try{
      return await K()
    }finally{
      this.endCLIActivity(q)
    }
  }getActivityStates(){
    return{
      isUserActive:(this.getNow()-this.lastUserActivityTime)/1000<this.USER_ACTIVITY_TIMEOUT_MS/1000,isCLIActive:this.isCLIActive,activeOperationCount:this.activeOperations.size
    }
  }
} /* confidence: 65% */

