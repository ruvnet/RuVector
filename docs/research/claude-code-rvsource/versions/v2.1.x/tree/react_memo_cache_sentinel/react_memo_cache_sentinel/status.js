// Module: status

/* original: OyY */ var OyY=30000;

/* original: z_7 */ function status(q){
  return{
    ...q,retain:!1,messages:void 0,diskLoaded:!1,evictAfter:Lo(q.status)?Date.now()+OyY:void 0
  } /* confidence: 70% */

