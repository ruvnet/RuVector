// Module: mapToQueryString

/* original: HZ */ var HZ={
  
};

/* original: K */ let composed_value=HZ.mapToQueryString(q); /* confidence: 30% */

/* original: PL */ class headers{
  constructor(q,K,_){
    this.httpMethod=q,this._baseEndpoint=K,this.headers={
      
    },this.bodyParameters={
      
    },this.queryParameters={
      
    },this.retryPolicy=_||new eD8
  }computeUri(){
    let q=new Map;
    if(this.queryParameters)v4.addExtraQueryParameters(q,this.queryParameters);
    let K=HZ.mapToQueryString(q);
    return b9.appendQueryString(this._baseEndpoint,K)
  }computeParametersBodyString(){
    let q=new Map;
    if(this.bodyParameters)v4.addExtraQueryParameters(q,this.bodyParameters);
    return HZ.mapToQueryString(q)
  }
} /* confidence: 70% */

