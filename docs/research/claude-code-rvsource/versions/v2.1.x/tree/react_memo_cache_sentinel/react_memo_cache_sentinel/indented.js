// Module: indented

/* original: Vv */ var Vv={
  
};

/* original: K */ var composed_value={
  rules:Vv,headingStyle:"setext",hr:"* * *",bulletListMarker:"*",codeBlockStyle:"indented",fence:"```",emDelimiter:"_",strongDelimiter:"**",linkStyle:"inlined",linkReferenceStyle:"full",br:"  ",preformattedCode:!1,blankReplacement:function(_,z){
    return z.isBlock?`

`:""
  },keepReplacement:function(_,z){
    return z.isBlock?`

`+z.outerHTML+`

`:z.outerHTML
  },defaultReplacement:function(_,z){
    return z.isBlock?`

`+_+`

`:_
  }
}; /* confidence: 30% */

