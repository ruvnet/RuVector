// Module: flex_start

/* original: Ieq */ var composed_value=L(()=>{
  beq()
} /* confidence: 30% */

/* original: kn6 */ var inkvirtualtext_inklink_inkprog=(q)=>{
  let _={
    nodeName:q,style:{
      
    },attributes:{
      
    },childNodes:[],parentNode:void 0,yogaNode:q!=="ink-virtual-text"&&q!=="ink-link"&&q!=="ink-progress"?xeq():void 0,dirty:!1
  };
   /* confidence: 65% */

/* original: $ */ let composed_value=q==="ink-text"&&z.isInsideText?"ink-virtual-text":q,O=kn6(composed_value); /* confidence: 30% */

/* original: $ */ let composed_value={
  col:Y.lo,row:z
} /* confidence: 30% */

/* original: z */ let composed_value={
  col:0,row:_
} /* confidence: 30% */

/* original: X */ let composed_value=(P,W)=>{
  if(W<_)return{
    col:0,row:_
  };
  if(W>z)return{
    col:Y-1,row:z
  };
  return{
    col:P.col,row:W
  }
}; /* confidence: 30% */

/* original: row */ const row = document.createElement('div');

/* original: Ceq */ function helper_fn(){
  return new IL1(otq.Node.create())
} /* confidence: 35% */

/* original: nG8 */ function helper_fn(q,K,_){
  q.anchor={
    col:K,row:_
  } /* confidence: 35% */

/* original: IL1 */ class rowreverse_columnreverse_wrapr{
  yoga;
  constructor(q){
    this.yoga=q
  }insertChild(q,K){
    this.yoga.insertChild(q.yoga,K)
  }removeChild(q){
    this.yoga.removeChild(q.yoga)
  }getChildCount(){
    return this.yoga.getChildCount()
  }getParent(){
    let q=this.yoga.getParent();
    return q?new rowreverse_columnreverse_wrapr(q):null
  }calculateLayout(q,K){
    this.yoga.calculateLayout(q,void 0,KG8.LTR)
  }setMeasureFunc(q){
    this.yoga.setMeasureFunc((K,_)=>{
      let z=_===q_.Exactly?rv6.Exactly:_===q_.AtMost?rv6.AtMost:rv6.Undefined;
      return q(K,z)
    })
  }unsetMeasureFunc(){
    this.yoga.unsetMeasureFunc()
  }markDirty(){
    this.yoga.markDirty()
  }getComputedLeft(){
    return this.yoga.getComputedLeft()
  }getComputedTop(){
    return this.yoga.getComputedTop()
  }getComputedWidth(){
    return this.yoga.getComputedWidth()
  }getComputedHeight(){
    return this.yoga.getComputedHeight()
  }getComputedBorder(q){
    return this.yoga.getComputedBorder(UO6[q])
  }getComputedPadding(q){
    return this.yoga.getComputedPadding(UO6[q])
  }setWidth(q){
    this.yoga.setWidth(q)
  }setWidthPercent(q){
    this.yoga.setWidthPercent(q)
  }setWidthAuto(){
    this.yoga.setWidthAuto()
  }setHeight(q){
    this.yoga.setHeight(q)
  }setHeightPercent(q){
    this.yoga.setHeightPercent(q)
  }setHeightAuto(){
    this.yoga.setHeightAuto()
  }setMinWidth(q){
    this.yoga.setMinWidth(q)
  }setMinWidthPercent(q){
    this.yoga.setMinWidthPercent(q)
  }setMinHeight(q){
    this.yoga.setMinHeight(q)
  }setMinHeightPercent(q){
    this.yoga.setMinHeightPercent(q)
  }setMaxWidth(q){
    this.yoga.setMaxWidth(q)
  }setMaxWidthPercent(q){
    this.yoga.setMaxWidthPercent(q)
  }setMaxHeight(q){
    this.yoga.setMaxHeight(q)
  }setMaxHeightPercent(q){
    this.yoga.setMaxHeightPercent(q)
  }setFlexDirection(q){
    let K={
      row:MM.Row,"row-reverse":MM.RowReverse,column:MM.Column,"column-reverse":MM.ColumnReverse
    };
    this.yoga.setFlexDirection(K[q])
  }setFlexGrow(q){
    this.yoga.setFlexGrow(q)
  }setFlexShrink(q){
    this.yoga.setFlexShrink(q)
  }setFlexBasis(q){
    this.yoga.setFlexBasis(q)
  }setFlexBasisPercent(q){
    this.yoga.setFlexBasisPercent(q)
  }setFlexWrap(q){
    let K={
      nowrap:Wr.NoWrap,wrap:Wr.Wrap,"wrap-reverse":Wr.WrapReverse
    };
    this.yoga.setFlexWrap(K[q])
  }setAlignItems(q){
    let K={
      auto:T9.Auto,stretch:T9.Stretch,"flex-start":T9.FlexStart,center:T9.Center,"flex-end":T9.FlexEnd
    };
    this.yoga.setAlignItems(K[q])
  }setAlignSelf(q){
    let K={
      auto:T9.Auto,stretch:T9.Stretch,"flex-start":T9.FlexStart,center:T9.Center,"flex-end":T9.FlexEnd
    };
    this.yoga.setAlignSelf(K[q])
  }setJustifyContent(q){
    let K={
      "flex-start":ZZ.FlexStart,center:ZZ.Center,"flex-end":ZZ.FlexEnd,"space-between":ZZ.SpaceBetween,"space-around":ZZ.SpaceAround,"space-evenly":ZZ.SpaceEvenly
    };
    this.yoga.setJustifyContent(K[q])
  }setDisplay(q){
    this.yoga.setDisplay(q==="flex"?Pr.Flex:Pr.None)
  }getDisplay(){
    return this.yoga.getDisplay()===Pr.None?vN.None:vN.Flex
  }setPositionType(q){
    this.yoga.setPositionType(q==="absolute"?uO6.Absolute:uO6.Relative)
  }setPosition(q,K){
    this.yoga.setPosition(UO6[q],K)
  }setPositionPercent(q,K){
    this.yoga.setPositionPercent(UO6[q],K)
  }setOverflow(q){
    let K={
      visible:IO6.Visible,hidden:IO6.Hidden,scroll:IO6.Scroll
    };
    this.yoga.setOverflow(K[q])
  }setMargin(q,K){
    this.yoga.setMargin(UO6[q],K)
  }setPadding(q,K){
    this.yoga.setPadding(UO6[q],K)
  }setBorder(q,K){
    this.yoga.setBorder(UO6[q],K)
  }setGap(q,K){
    this.yoga.setGap(Bf_[q],K)
  }free(){
    this.yoga.free()
  }freeRecursive(){
    this.yoga.freeRecursive()
  }
} /* confidence: 65% */

