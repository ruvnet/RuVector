// Module: join

/* original: A$K */ var linux_darwin=process.env.XDG_RUNTIME_DIR,Joz=new Map([["linux",107],["darwin",103]]); /* confidence: 65% */

/* original: Moz */ function hex_win32(){
  let q=(0,Hoz.randomBytes)(21).toString("hex");
  if(process.platform==="win32")return`\\\\.\\pipe\\vscode-jsonrpc-${q}-sock`;
  let K;
  if(A$K)K=O$K.join(A$K,`vscode-ipc-${q}.sock`);
  else K=O$K.join(joz.tmpdir(),`vscode-${q}.sock`);
  let _=Joz.get(process.platform);
  if(_!==void 0&&K.length>_)(0,Hh6.default)().console.warn(`WARNING: IPC handle "${K}" is longer than ${_} characters.`);
  return K
} /* confidence: 65% */

