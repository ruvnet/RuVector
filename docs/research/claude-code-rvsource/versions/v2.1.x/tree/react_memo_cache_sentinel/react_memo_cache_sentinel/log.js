// Module: log

/* original: YG5 */ var YG5={
  log(){
    
  }

/* original: $G5 */ function ErrorLogger(q){
  if(q===!1)return YG5;
  if(q===void 0)return console;
  if(q.log&&q.warn&&q.error)return q;
  throw Error("logger must implement log, warn and error methods")
} /* confidence: 58% */

