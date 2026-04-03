// Module: _root

/* original: n51 */ var n51=()=>{
  
};

/* original: xF8 */ var composed_value=L(()=>{
  _8();
  _n()
} /* confidence: 30% */

/* original: EU8 */ var composed_value=L(()=>{
  T8();
  FO();
  XV6();
  dN();
  Cq8();
  TL6();
  Mo();
  yq6();
  ku8();
  Q78();
  $h6();
  Ys();
  qP();
  uD();
  GM();
  jG();
  _n();
  lo();
  Lm8()
} /* confidence: 30% */

/* original: s58 */ var composed_value=L(()=>{
  T8();
  k1();
  _8();
  jG();
  yK();
  _n();
  P5()
} /* confidence: 30% */

/* original: O27 */ var composed_value=L(()=>{
  I7();
  _n();
  P5()
} /* confidence: 30% */

/* original: J27 */ var composed_value=L(()=>{
  _8();
  s58();
  r8();
  O27();
  FA7();
  cA7();
  W55()
} /* confidence: 30% */

/* original: iZ7 */ class ErrorBoundary{
  constructor(q,K={
    
  }){
    this._values={
      
    },this._blockStarts=[],this._constants={
      
    },this.opts={
      ...K,_n:K.lines?`
`:""
    },this._extScope=q,this._scope=new DI.Scope({
      parent:q
    }),this._nodes=[new dZ7]
  }toString(){
    return this._root.render(this.opts)
  }name(q){
    return this._scope.name(q)
  }scopeName(q){
    return this._extScope.name(q)
  }scopeValue(q,K){
    let _=this._extScope.value(q,K);
    return(this._values[_.prefix]||(this._values[_.prefix]=new Set)).add(_),_
  }getScopeValue(q,K){
    return this._extScope.getValue(q,K)
  }scopeRefs(q){
    return this._extScope.scopeRefs(q,this._values)
  }scopeCode(){
    return this._extScope.scopeCode(this._values)
  }_def(q,K,_,z){
    let Y=this._scope.toName(K);
    if(_!==void 0&&z)this._constants[Y.str]=_;
    return this._leafNode(new pZ7(q,Y,_)),Y
  }const(q,K,_){
    return this._def(DI.varKinds.const,q,K,_)
  }let(q,K,_){
    return this._def(DI.varKinds.let,q,K,_)
  }var(q,K,_){
    return this._def(DI.varKinds.var,q,K,_)
  }assign(q,K,_){
    return this._leafNode(new qq1(q,K,_))
  }add(q,K){
    return this._leafNode(new BZ7(q,DV.operators.ADD,K))
  }code(q){
    if(typeof q=="function")q();
    else if(q!==MY.nil)this._leafNode(new QZ7(q));
    return this
  }object(...q){
    let K=["{"];
    for(let[_,z]of q){
      if(K.length>1)K.push(",");
      if(K.push(_),_!==z||this.opts.es5)K.push(":"),(0,MY.addCodeArg)(K,z)
    }return K.push("}"),new MY._Code(K)
  }if(q,K,_){
    if(this._blockNode(new ll(q)),K&&_)this.code(K).else().code(_).endIf();
    else if(K)this.code(K).endIf();
    else if(_)throw Error('CodeGen: "else" body without "then" body');
    return this
  }elseIf(q){
    return this._elseNode(new ll(q))
  }else(){
    return this._elseNode(new am6)
  }endIf(){
    return this._endBlockNode(ll,am6)
  }_for(q,K){
    if(this._blockNode(q),K)this.code(K).endFor();
    return this
  }for(q,K){
    return this._for(new cZ7(q),K)
  }forRange(q,K,_,z,Y=this.opts.es5?DI.varKinds.var:DI.varKinds.let){
    let $=this._scope.toName(q);
    return this._for(new lZ7(Y,$,K,_),()=>z($))
  }forOf(q,K,_,z=DI.varKinds.const){
    let Y=this._scope.toName(q);
    if(this.opts.es5){
      let $=K instanceof MY.Name?K:this.var("_arr",K);
      return this.forRange("_i",0,MY._`${$}.length`,(O)=>{
        this.var(Y,MY._`${$}[${O}]`),_(Y)
      })
    }return this._for(new t71("of",z,Y,K),()=>_(Y))
  }forIn(q,K,_,z=this.opts.es5?DI.varKinds.var:DI.varKinds.const){
    if(this.opts.ownProperties)return this.forOf(q,MY._`Object.keys(${K})`,_);
    let Y=this._scope.toName(q);
    return this._for(new t71("in",z,Y,K),()=>_(Y))
  }endFor(){
    return this._endBlockNode(VW6)
  }label(q){
    return this._leafNode(new gZ7(q))
  }break(q){
    return this._leafNode(new FZ7(q))
  }return(q){
    let K=new sY8;
    if(this._blockNode(K),this.code(q),K.nodes.length!==1)throw Error('CodeGen: "return" should have one node');
    return this._endBlockNode(sY8)
  }try(q,K,_){
    if(!K&&!_)throw Error('CodeGen: "try" without "catch" and "finally"');
    let z=new nZ7;
    if(this._blockNode(z),this.code(q),K){
      let Y=this.name("e");
      this._currNode=z.catch=new tY8(Y),K(Y)
    }if(_)this._currNode=z.finally=new eY8,this.code(_);
    return this._endBlockNode(tY8,eY8)
  }throw(q){
    return this._leafNode(new UZ7(q))
  }block(q,K){
    if(this._blockStarts.push(this._nodes.length),q)this.code(q).endBlock(K);
    return this
  }endBlock(q){
    let K=this._blockStarts.pop();
    if(K===void 0)throw Error("CodeGen: not in self-balancing block");
    let _=this._nodes.length-K;
    if(_<0||q!==void 0&&_!==q)throw Error(`CodeGen: wrong number of nodes: ${_} vs ${q} expected`);
    return this._nodes.length=K,this
  }func(q,K=MY.nil,_,z){
    if(this._blockNode(new aY8(q,K,_)),z)this.code(z).endFunc();
    return this
  }endFunc(){
    return this._endBlockNode(aY8)
  }optimize(q=1){
    while(q-- >0)this._root.optimizeNodes(),this._root.optimizeNames(this._root.names,this._constants)
  }_leafNode(q){
    return this._currNode.nodes.push(q),this
  }_blockNode(q){
    this._currNode.nodes.push(q),this._nodes.push(q)
  }_endBlockNode(q,K){
    let _=this._currNode;
    if(_ instanceof q||K&&_ instanceof K)return this._nodes.pop(),this;
    throw Error(`CodeGen: not in block "${K?`${
      q.kind
    }/${
      K.kind
    }`:q.kind}"`)
  }_elseNode(q){
    let K=this._currNode;
    if(!(K instanceof ll))throw Error('CodeGen: "else" without "if"');
    return this._currNode=K.else=q,this
  }get _root(){
    return this._nodes[0]
  }get _currNode(){
    let q=this._nodes;
    return q[q.length-1]
  }set _currNode(q){
    let K=this._nodes;
    K[K.length-1]=q
  }
} /* confidence: 85% */

