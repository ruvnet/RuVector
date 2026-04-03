// Module: assistant

const container = document.getElementById('hour-histogram');

/* original: Sb6 */ function assistant_message_assistant_to(q,K){
  return{
    type:"assistant",uuid:vnY(),message:{
      id:`remote-${K}`,type:"message",role:"assistant",content:[{
        type:"tool_use",id:q.tool_use_id,name:q.tool_name,input:q.input
      }],model:"",stop_reason:null,stop_sequence:null,container:null,context_management:null,usage:{
        input_tokens:0,output_tokens:0,cache_creation_input_tokens:0,cache_read_input_tokens:0
      }
    },requestId:void 0,timestamp:new Date().toISOString()
  }
} /* confidence: 65% */

