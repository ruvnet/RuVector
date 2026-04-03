// Module: readwrite

/* original: xW1 */ var xW1="IdentityIds";

/* original: Ljq */ class value_holder{
  dbName;
  constructor(q="aws:cognito-identity-ids"){
    this.dbName=q
  }getItem(q){
    return this.withObjectStore("readonly",(K)=>{
      let _=K.get(q);
      return new Promise((z)=>{
        _.onerror=()=>z(null),_.onsuccess=()=>z(_.result?_.result.value:null)
      })
    }).catch(()=>null)
  }removeItem(q){
    return this.withObjectStore("readwrite",(K)=>{
      let _=K.delete(q);
      return new Promise((z,Y)=>{
        _.onerror=()=>Y(_.error),_.onsuccess=()=>z()
      })
    })
  }setItem(q,K){
    return this.withObjectStore("readwrite",(_)=>{
      let z=_.put({
        id:q,value:K
      });
      return new Promise((Y,$)=>{
        z.onerror=()=>$(z.error),z.onsuccess=()=>Y()
      })
    })
  }getDb(){
    let q=self.indexedDB.open(this.dbName,1);
    return new Promise((K,_)=>{
      q.onsuccess=()=>{
        K(q.result)
      },q.onerror=()=>{
        _(q.error)
      },q.onblocked=()=>{
        _(Error("Unable to access DB"))
      },q.onupgradeneeded=()=>{
        let z=q.result;
        z.onerror=()=>{
          _(Error("Failed to create object store"))
        },z.createObjectStore(xW1,{
          keyPath:"id"
        })
      }
    })
  }withObjectStore(q,K){
    return this.getDb().then((_)=>{
      let z=_.transaction(xW1,q);
      return z.oncomplete=()=>_.close(),new Promise((Y,$)=>{
        z.onerror=()=>$(z.error),Y(K(z.objectStore(xW1)))
      }).catch((Y)=>{
        throw _.close(),Y
      })
    })
  }
} /* confidence: 70% */

