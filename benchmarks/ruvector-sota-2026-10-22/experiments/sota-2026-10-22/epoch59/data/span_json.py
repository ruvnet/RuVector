"""Unchanged pure SpanJSON class extracted from CP-pinned E53 custody94bc...; no controller.
"""
import json,re
NUMBER=re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
WS=b' \t\r\n'
class SpanJSON:
 def __init__(self,data):
  if type(data)is not bytes:raise ValueError('BYTES_REQUIRED')
  self.data=data;self.decoded={'field_name':0,'metadata':0,'selected_document':0,'selected_hypothesis':0,'legacy_dev_text':0};self.gold_values_decoded=0
 def ws(self,i):
  while i<len(self.data)and self.data[i]in WS:i+=1
  return i
 def string_end(self,i):
  if i>=len(self.data)or self.data[i]!=34:raise ValueError('STRING_REQUIRED')
  j=i+1
  while j<len(self.data):
   x=self.data[j]
   if x==34:return j+1
   if x<32:raise ValueError('UNESCAPED_CONTROL')
   if x==92:
    j+=1
    if j>=len(self.data):raise ValueError('INCOMPLETE_ESCAPE')
    if self.data[j]==117:
     if j+4>=len(self.data)or any(c not in b'0123456789abcdefABCDEF'for c in self.data[j+1:j+5]):raise ValueError('UNICODE_ESCAPE')
     j+=4
    elif self.data[j]not in b'"\\/bfnrt':raise ValueError('JSON_ESCAPE')
   j+=1
  raise ValueError('UNTERMINATED_STRING')
 def value(self,i,depth=0):
  if depth>64:raise ValueError('JSON_DEPTH')
  i=self.ws(i)
  if i>=len(self.data):raise ValueError('MISSING_VALUE')
  c=self.data[i]
  if c==34:return(i,self.string_end(i),'string')
  if c in(123,91):
   close=125 if c==123 else 93;j=self.ws(i+1)
   if j<len(self.data)and self.data[j]==close:return(i,j+1,'object'if c==123 else'array')
   while True:
    if c==123:
     j=self.string_end(j);j=self.ws(j)
     if j>=len(self.data)or self.data[j]!=58:raise ValueError('MISSING_COLON')
     j+=1
    _,j,_=self.value(j,depth+1);j=self.ws(j)
    if j>=len(self.data):raise ValueError('UNFINISHED_CONTAINER')
    if self.data[j]==close:return(i,j+1,'object'if c==123 else'array')
    if self.data[j]!=44:raise ValueError('MISSING_COMMA')
    j=self.ws(j+1)
  for literal,kind in [(b'true','bool'),(b'false','bool'),(b'null','null')]:
   if self.data.startswith(literal,i):return(i,i+len(literal),kind)
  m=NUMBER.match(self.data,i)
  if m:return(i,m.end(),'number')
  raise ValueError('INVALID_JSON_VALUE')
 def decode_string(self,span,field):
  if field not in self.decoded:raise ValueError('FORBIDDEN_VALUE_DECODER')
  a,b,kind=span
  if kind!='string':raise ValueError('STRING_DECODER_TYPE')
  value=json.loads(self.data[a:b]);self.decoded[field]+=1
  if type(value)is not str:raise ValueError('STRING_DECODER_RESULT')
  return value
 def members(self,span):
  a,b,kind=span
  if kind!='object':raise ValueError('OBJECT_REQUIRED')
  i=self.ws(a+1);out={}
  if i==b-1:return out
  while i<b-1:
   end=self.string_end(i);key=self.decode_string((i,end,'string'),'field_name');i=self.ws(end)
   if i>=b or self.data[i]!=58:raise ValueError('OBJECT_COLON')
   value=self.value(i+1);i=self.ws(value[1])
   if key in out:raise ValueError('DUPLICATE_JSON_KEY')
   out[key]=value
   if i==b-1:return out
   if self.data[i]!=44:raise ValueError('OBJECT_COMMA')
   i=self.ws(i+1)
  raise ValueError('OBJECT_TERMINATION')
 def items(self,span):
  a,b,kind=span
  if kind!='array':raise ValueError('ARRAY_REQUIRED')
  i=self.ws(a+1)
  if i==b-1:return
  while i<b-1:
   value=self.value(i);yield value;i=self.ws(value[1])
   if i==b-1:return
   if self.data[i]!=44:raise ValueError('ARRAY_COMMA')
   i=self.ws(i+1)
  raise ValueError('ARRAY_TERMINATION')
 def root_records(self):
  root=self.value(0)
  if root[2]!='array'or self.ws(root[1])!=len(self.data):raise ValueError('EXACT_ROOT_ARRAY_REQUIRED')
  for i,span in enumerate(self.items(root)):
   if span[2]!='object':raise ValueError('RECORD_OBJECT_REQUIRED')
   yield i,span
 def string_nonempty(self,span):
  """Compute a whitespace-only flag without constructing/decoding text value."""
  a,b,kind=span
  if kind!='string':return False
  i=a+1
  while i<b-1:
   c=self.data[i]
   if c==92:
    i+=1;x=self.data[i]
    if x==117:ch=chr(int(self.data[i+1:i+5],16));i+=5
    else:ch={34:'"',92:'\\',47:'/',98:'\b',102:'\f',110:'\n',114:'\r',116:'\t'}[x];i+=1
   elif c<128:ch=chr(c);i+=1
   else:
    n=2 if c&224==192 else 3 if c&240==224 else 4 if c&248==240 else 0
    if not n:raise ValueError('INVALID_UTF8')
    ch=self.data[i:i+n].decode('utf-8');i+=n
   if not ch.isspace():return True
  return False
 def raw(self,span):return self.data[span[0]:span[1]]
