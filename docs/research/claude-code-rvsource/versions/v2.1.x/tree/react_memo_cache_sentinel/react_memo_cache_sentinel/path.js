// Module: path

/* original: DA4 */ var DA4=B((caO,WA4)=>{
  WA4.exports={
    ...cA6(),...GV8(),...JO4(),...UO4(),..._A4(),...zm(),...PA4(),...yV8(),...t76(),...Ir6()
  }
}

/* original: N24 */ var path_handler=B((k24)=>{
  Object.defineProperty(k24,"__esModule",{
    value:!0
  });
  k24.DestroyerOfModules=void 0;
  var mV8=DA4(),dk6=U6("path"),YI1=zI1();
  class T24{
    constructor({
      rootDirectory:q,walker:K,shouldKeepModuleTest:_
    }){
      if(q)this.walker=new YI1.Walker(q);
      else if(K)this.walker=K;
      else throw Error("Must either provide rootDirectory or walker argument");
      if(_)this.shouldKeepFn=_
    }async destroyModule(q,K){
      if(K.get(q)){
        let z=dk6.resolve(q,"node_modules");
        if(!await mV8.pathExists(z))return;
        for(let Y of await mV8.readdir(z))if(Y.startsWith("@"))for(let $ of await mV8.readdir(dk6.resolve(z,Y)))await this.destroyModule(dk6.resolve(z,Y,$),K);
        else await this.destroyModule(dk6.resolve(z,Y),K)
      }else await mV8.remove(q)
    }async collectKeptModules({
      relativePaths:q=!1
    }){
      let K=await this.walker.walkTree(),_=new Map,z=dk6.resolve(this.walker.getRootModule());
      for(let Y of K)if(this.shouldKeepModule(Y)){
        let $=Y.path;
        if(q)$=$.replace(`${z}${dk6.sep}`,"");
        _.set($,Y)
      }return _
    }async destroy(){
      await this.destroyModule(this.walker.getRootModule(),await this.collectKeptModules({
        relativePaths:!1
      }))
    }shouldKeepModule(q){
      let K=q.depType===YI1.DepType.DEV||q.depType===YI1.DepType.DEV_OPTIONAL;
      return this.shouldKeepFn?this.shouldKeepFn(q,K):!K
    }
  }k24.DestroyerOfModules=T24
} /* confidence: 70% */

/* original: mV8 */ var composed_value=DA4(),dk6=U6("path"),YI1=zI1(); /* confidence: 30% */

/* original: T24 */ class path_handler{
  constructor({
    rootDirectory:q,walker:K,shouldKeepModuleTest:_
  }){
    if(q)this.walker=new YI1.Walker(q);
    else if(K)this.walker=K;
    else throw Error("Must either provide rootDirectory or walker argument");
    if(_)this.shouldKeepFn=_
  }async destroyModule(q,K){
    if(K.get(q)){
      let z=dk6.resolve(q,"node_modules");
      if(!await mV8.pathExists(z))return;
      for(let Y of await mV8.readdir(z))if(Y.startsWith("@"))for(let $ of await mV8.readdir(dk6.resolve(z,Y)))await this.destroyModule(dk6.resolve(z,Y,$),K);
      else await this.destroyModule(dk6.resolve(z,Y),K)
    }else await mV8.remove(q)
  }async collectKeptModules({
    relativePaths:q=!1
  }){
    let K=await this.walker.walkTree(),_=new Map,z=dk6.resolve(this.walker.getRootModule());
    for(let Y of K)if(this.shouldKeepModule(Y)){
      let $=Y.path;
      if(q)$=$.replace(`${z}${dk6.sep}`,"");
      _.set($,Y)
    }return _
  }async destroy(){
    await this.destroyModule(this.walker.getRootModule(),await this.collectKeptModules({
      relativePaths:!1
    }))
  }shouldKeepModule(q){
    let K=q.depType===YI1.DepType.DEV||q.depType===YI1.DepType.DEV_OPTIONAL;
    return this.shouldKeepFn?this.shouldKeepFn(q,K):!K
  }
} /* confidence: 70% */

