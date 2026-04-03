// Module: base64

const imagepng_base64 = fs.readFileSync("image.png").toString("base64"); /* confidence: 65% */

const composed_value = await client.messages.create({
  
  model: "{{OPUS_ID}}",
  max_tokens: 16000,
  messages: [
    {
    
      role: "user",
      content: [
        {
      
          type: "image",
          source: {
         type: "base64", media_type: "image/png", data: imageData 
      },
        
    },
        {
       type: "text", text: "What's in this image?" 
    },
      ],
    
  },
  ],

} /* confidence: 30% */

