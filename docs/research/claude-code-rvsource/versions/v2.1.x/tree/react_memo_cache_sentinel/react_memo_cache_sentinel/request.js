// Module: request

/* original: Xa */ var channel_subchannel_server_sock={
  ["channel"]:new a26.OrderedMap,["subchannel"]:new a26.OrderedMap,["server"]:new a26.OrderedMap,["socket"]:new a26.OrderedMap
} /* confidence: 65% */

/* original: z */ let composed_value=Xa[q]; /* confidence: 30% */

/* original: A */ let headers=parseInt(q.request.socket_id,10),w=Xa.socket.getElementByKey(headers); /* confidence: 70% */

/* original: QRz */ function helper_fn(q){
  Xa[q.kind].eraseElementByKey(q.id)
} /* confidence: 35% */

/* original: nRz */ function request(q,K){
  let _=parseInt(q.request.channel_id,10),z=Xa.channel.getElementByKey(_);
   /* confidence: 70% */

/* original: iRz */ function request(q,K){
  let _=parseInt(q.request.max_results,10)||rl1,z=[],Y=parseInt(q.request.start_channel_id,10),$=Xa.channel,O;
   /* confidence: 70% */

/* original: rRz */ function request(q,K){
  let _=parseInt(q.request.server_id,10),Y=Xa.server.getElementByKey(_);
   /* confidence: 70% */

/* original: oRz */ function request(q,K){
  let _=parseInt(q.request.max_results,10)||rl1,z=parseInt(q.request.start_server_id,10),Y=Xa.server,$=[],O;
   /* confidence: 70% */

/* original: aRz */ function request(q,K){
  let _=parseInt(q.request.subchannel_id,10),z=Xa.subchannel.getElementByKey(_);
   /* confidence: 70% */

/* original: tRz */ function request(q,K){
  let _=parseInt(q.request.server_id,10),z=Xa.server.getElementByKey(_);
   /* confidence: 70% */

/* original: ja4 */ function helper_fn(){
  return{
    GetChannel:nRz,GetTopChannels:iRz,GetServer:rRz,GetServers:oRz,GetSubchannel:aRz,GetSocket:sRz,GetServerSockets:tRz
  } /* confidence: 35% */

