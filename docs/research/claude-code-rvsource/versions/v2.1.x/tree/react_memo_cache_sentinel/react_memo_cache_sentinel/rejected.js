// Module: rejected

/* original: uE1 */ function utility_fn(){
  
} /* confidence: 40% */

/* original: LW_ */ function status(q){
  switch(q.status){
    case"fulfilled":return q.value;
    case"rejected":throw q.reason;
    default:switch(typeof q.status==="string"?q.then(uE1,uE1):(q.status="pending",q.then(function(K){
      q.status==="pending"&&(q.status="fulfilled",q.value=K)
    },function(K){
      q.status==="pending"&&(q.status="rejected",q.reason=K)
    })),q.status){
      case"fulfilled":return q.value;
      case"rejected":throw q.reason
    }
  } /* confidence: 70% */

