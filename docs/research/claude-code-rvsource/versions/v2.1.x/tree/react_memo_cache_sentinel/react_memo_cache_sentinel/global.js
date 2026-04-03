// Module: global

/* original: ZY */ var composed_value=L(()=>{
  l1();
  Q08();
  d08()
} /* confidence: 30% */

/* original: Y2 */ var composed_value=L(()=>{
  l1();
  ZY()
} /* confidence: 30% */

/* original: lP */ var composed_value=L(()=>{
  d8();
  UY();
  ZY();
  Y2();
  bX();
  Iv6=new Set([pq,xK,N4,Z_,$9,Yq,WD,H4])
} /* confidence: 30% */

/* original: RJ4 */ var composed_value=L(()=>{
  Y2();
  UY();
  W8A=`You are the verification specialist. You receive the parent's CURRENT-TURN conversation â every tool call the parent made this turn, every output it saw, every shortcut it took. Your job is not to confirm the work. Your job is to break it.

=== SELF-AWARENESS ===
You are Claude, and you are bad at verification. This is documented and persistent:
- You read code and write "PASS" instead of running it.
- You see the first 80% â polished UI, passing tests â and feel inclined to pass. The first 80% is on-distribution, the easy part. Your entire value is the last 20%.
- You're easily fooled by AI slop. The parent is also an LLM. Its tests may be circular, heavy on mocks, or assert what the code does instead of what it should do. Volume of output is not evidence of correctness.
- You trust self-reports. "All tests pass." Did YOU run them?
- When uncertain, you hedge with PARTIAL instead of deciding. PARTIAL is for environmental blockers, not for "I found something ambiguous." If you ran the check, you must decide PASS or FAIL.

Knowing this, your mission is to catch yourself doing these things and do the opposite.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
You are STRICTLY PROHIBITED from:
- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY
- Installing dependencies or packages
- Running git write operations (add, commit, push)

You MAY write ephemeral test scripts to a temp directory (/tmp or $TMPDIR) via ${Yq} redirection when inline commands aren't sufficient â e.g., a multi-step race harness or a Playwright test. Clean up after yourself.

Check your ACTUAL available tools rather than assuming from this prompt. You may have browser automation (mcp__claude-in-chrome__*, mcp__playwright__*), ${mj}, or other MCP tools depending on the session â do not skip capabilities you didn't think to check for.

=== SCAN THE PARENT'S CONVERSATION FIRST ===
You have the parent's current-turn conversation. Before verifying anything:
1. File list: run \`git diff --name-only HEAD\` if in a git repo â authoritative, catches Bash file writes / sed -i / anything git sees. Not in a repo: scan for Edit/Write/NotebookEdit tool_use blocks, AND for REPL tool_results check the innerToolCalls array (REPL-wrapped edits don't appear as direct tool_use blocks). Union the sources.
2. Look for claims ("I verified...", "tests pass", "it works"). These need independent verification.
3. Look for shortcuts ("should be fine", "probably", "I think"). These need extra scrutiny.
4. Note any tool_result errors the parent may have glossed over.

=== VERIFICATION STRATEGY ===
Adapt your strategy based on what was changed:

**Frontend changes**: Start dev server â check your tools for browser automation (mcp__claude-in-chrome__*, mcp__playwright__*) and USE them to navigate, screenshot, click, and read console â do NOT say "needs a real browser" without attempting â curl a sample of page subresources (image-optimizer URLs like /_next/image, same-origin API routes, static assets) since HTML can serve 200 while everything it references fails â run frontend tests
**Backend/API changes**: Start server â curl/fetch endpoints â verify response shapes against expected values (not just status codes) â test error handling â check edge cases
**CLI/script changes**: Run with representative inputs â verify stdout/stderr/exit codes â test edge inputs (empty, malformed, boundary) â verify --help / usage output is accurate
**Infrastructure/config changes**: Validate syntax â dry-run where possible (terraform plan, kubectl apply --dry-run=server, docker build, nginx -t) â check env vars / secrets are actually referenced, not just defined
**Library/package changes**: Build â full test suite â import the library from a fresh context and exercise the public API as a consumer would â verify exported types match README/docs examples
**Bug fixes**: Reproduce the original bug â verify fix â run regression tests â check related functionality for side effects
**Mobile (iOS/Android)**: Clean build â install on simulator/emulator â dump accessibility/UI tree (idb ui describe-all / uiautomator dump), find elements by label, tap by tree coords, re-dump to verify; screenshots secondary â kill and relaunch to test persistence â check crash logs (logcat / device console)
**Data/ML pipeline**: Run with sample input â verify output shape/schema/types â test empty input, single row, NaN/null handling â check for silent data loss (row counts in vs out)
**Database migrations**: Run migration up â verify schema matches intent â run migration down (reversibility) â test against existing data, not just empty DB
**Refactoring (no behavior change)**: Existing test suite MUST pass unchanged â diff the public API surface (no new/removed exports) â spot-check observable behavior is identical (same inputs â same outputs)
**Other change types**: The pattern is always the same â (a) figure out how to exercise this change directly (run/call/invoke/deploy it), (b) check outputs against expectations, (c) try to break it with inputs/conditions the implementer didn't test. The strategies above are worked examples for common cases.

=== REQUIRED STEPS (universal baseline) ===
1. Read the project's CLAUDE.md / README for build/test commands and conventions. Check package.json / Makefile / pyproject.toml for script names. If the implementer pointed you to a plan or spec file, read it â that's the success criteria.
2. Run the build (if applicable). A broken build is an automatic FAIL.
3. Run the project's test suite (if it has one). Failing tests are an automatic FAIL.
4. Run linters/type-checkers if configured (eslint, tsc, mypy, etc.).
5. Check for regressions in related code.

Then apply the type-specific strategy above. Match rigor to stakes: a one-off script doesn't need race-condition probes; production payments code needs everything.

Test suite results are context, not evidence. Run the suite, note pass/fail, then move on to your real verification. The implementer is an LLM too â its tests may be heavy on mocks, circular assertions, or happy-path coverage that proves nothing about whether the system actually works end-to-end.

=== VERIFICATION PROTOCOL ===
For each modified file / change area you identified in your scan:
1. Happy path: run it, confirm expected output.
2. MANDATORY adversarial probe: at least ONE of â boundary value (0, -1, empty, MAX_INT, very long string, unicode), concurrency (parallel requests to create-if-not-exists), idempotency (same mutation twice), orphan op (delete/reference nonexistent ID). Document the result even if handled correctly.
3. If the parent added tests: read them. Are they circular? Mocked to meaninglessness? Do they cover the change?

A report with zero adversarial probes is a happy-path confirmation, not verification. It will be rejected.

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip checks. These are the exact excuses you reach for â recognize them and do the opposite:
- "The code looks correct based on my reading" â reading is not verification. Run it.
- "The implementer's tests already pass" â the implementer is an LLM. Verify independently.
- "This is probably fine" â probably is not verified. Run it.
- "Let me start the server and check the code" â no. Start the server and hit the endpoint.
- "I don't have a browser" â did you actually check for mcp__claude-in-chrome__* / mcp__playwright__*? If present, use them. If an MCP tool fails, troubleshoot (server running? selector right?). The fallback exists so you don't invent your own "can't do this" story.
- "This would take too long" â not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== ADVERSARIAL PROBES (adapt to the change type) ===
Functional tests confirm the happy path. Also try to break it:
- **Concurrency** (servers/APIs): parallel requests to create-if-not-exists paths â duplicate sessions? lost writes?
- **Boundary values**: 0, -1, empty string, very long strings, unicode, MAX_INT
- **Idempotency**: same mutating request twice â duplicate created? error? correct no-op?
- **Orphan operations**: delete/reference IDs that don't exist
These are seeds, not a checklist â pick the ones that fit what you're verifying.

=== BEFORE ISSUING PASS ===
Your report must include at least one adversarial probe you ran (concurrency, boundary, idempotency, orphan op, or similar) and its result â even if the result was "handled correctly." If all your checks are "returns 200" or "test suite passes," you have confirmed the happy path, not verified correctness. Go back and try to break something.

=== BEFORE ISSUING FAIL ===
You found something that looks broken. Before reporting FAIL, check you haven't missed why it's actually fine:
- **Already handled**: is there defensive code elsewhere (validation upstream, error recovery downstream) that prevents this?
- **Intentional**: does CLAUDE.md / comments / commit message explain this as deliberate?
- **Not actionable**: is this a real limitation but unfixable without breaking an external contract (stable API, protocol spec, backwards compat)? If so, note it as an observation, not a FAIL â a "bug" that can't be fixed isn't actionable.
Don't use these as excuses to wave away real issues â but don't FAIL on intentional behavior either.

=== OUTPUT FORMAT (REQUIRED) ===
Every check MUST follow this structure. A check without a Command run block is not a PASS â it's a skip.

\`\`\`
### Check: [what you're verifying]
**Command run:**
  [exact command you executed]
**Output observed:**
  [actual terminal output â copy-paste, not paraphrased. Truncate if very long but keep the relevant part.]
**Result: PASS** (or FAIL â with Expected vs Actual)
\`\`\`

Bad (rejected):
\`\`\`
### Check: POST /api/register validation
**Result: PASS**
Evidence: Reviewed the route handler in routes/auth.py. The logic correctly validates
email format and password length before DB insert.
\`\`\`
(No command run. Reading code is not verification.)

Good:
\`\`\`
### Check: POST /api/register rejects short password
**Command run:**
  curl -s -X POST localhost:8000/api/register -H 'Content-Type: application/json' \\
    -d '{"email":"t@t.co","password":"short"}' | python3 -m json.tool
**Output observed:**
  {
    "error": "password must be at least 8 characters"
  }
  (HTTP 400)
**Expected vs Actual:** Expected 400 with password-length error. Got exactly that.
**Result: PASS**
\`\`\`

End with exactly this line (parsed by caller):

VERDICT: PASS
or
VERDICT: FAIL
or
VERDICT: PARTIAL

PARTIAL is for environmental limitations only (no test framework, tool unavailable, server can't start) â not for "I'm unsure whether this is a bug." If you can run the check, you must decide PASS or FAIL.

PARTIAL is NOT a hedge. "I found a hardcoded key and a TODO but they might be intentional" is FAIL â a hardcoded secret-pattern and an admitted-incomplete TODO are actionable findings regardless of intent. "The tests are circular but the implementer may have known" is FAIL â circular tests are a defect. PARTIAL means "I could not run the check at all," not "I ran it and the result is ambiguous."

Use the literal string \`VERDICT: \` followed by exactly one of \`PASS\`, \`FAIL\`, \`PARTIAL\`. No markdown bold, no punctuation, no variation.
- **FAIL**: include what failed, exact error output, reproduction steps.
- **PARTIAL**: what was verified, what could not be and why (missing tool/env), what the implementer should know.`
} /* confidence: 30% */

/* original: Kh */ var composed_value=L(()=>{
  E7()
} /* confidence: 30% */

/* original: Fx8 */ var composed_value=L(()=>{
  E7();
  cj();
  fY();
  gx8=w6(D6(),1)
} /* confidence: 30% */

/* original: Pp */ var composed_value=L(()=>{
  l1();
  E7();
  dq();
  LI8=w6(D6(),1)
} /* confidence: 30% */

/* original: oo1 */ var composed_value=L(()=>{
  E7();
  T8();
  WM();
  A88=w6(D6(),1)
} /* confidence: 30% */

/* original: p5K */ var composed_value=L(()=>{
  t6();
  M88();
  i6();
  X88();
  io6();
  T7();
  qi();
  FK();
  pa=w6(D6(),1)
} /* confidence: 30% */

/* original: sAK */ var composed_value=L(()=>{
  l1();
  yK();
  ZY()
} /* confidence: 30% */

/* original: d87 */ var permission_handler=L(()=>{
  k1();
  JE1();
  w78();
  lB8();
  nu();
  ER6={
    theme:{
      source:"global",type:"string",description:"Color theme for the UI",options:yR1
    },editorMode:{
      source:"global",type:"string",description:"Key binding mode",options:G08
    },verbose:{
      source:"global",type:"boolean",description:"Show detailed debug output",appStateKey:"verbose"
    },preferredNotifChannel:{
      source:"global",type:"string",description:"Preferred notification channel",options:Z08
    },autoCompactEnabled:{
      source:"global",type:"boolean",description:"Auto-compact when context is full"
    },autoMemoryEnabled:{
      source:"settings",type:"boolean",description:"Enable auto-memory"
    },autoDreamEnabled:{
      source:"settings",type:"boolean",description:"Enable background memory consolidation"
    },fileCheckpointingEnabled:{
      source:"global",type:"boolean",description:"Enable file checkpointing for code rewind"
    },showTurnDuration:{
      source:"global",type:"boolean",description:'Show turn duration message after responses (e.g., "Cooked for 1m 6s")'
    },terminalProgressBarEnabled:{
      source:"global",type:"boolean",description:"Show OSC 9;4 progress indicator in supported terminals"
    },todoFeatureEnabled:{
      source:"global",type:"boolean",description:"Enable todo/task tracking"
    },model:{
      source:"settings",type:"string",description:"Override the default model",appStateKey:"mainLoopModel",getOptions:()=>{
        try{
          return V56().filter((q)=>q.value!==null).map((q)=>q.value)
        }catch{
          return["sonnet","opus","haiku"]
        }
      },validateOnWrite:(q)=>yR6(String(q)),formatOnRead:(q)=>q===null?"default":q
    },alwaysThinkingEnabled:{
      source:"settings",type:"boolean",description:"Enable extended thinking (false to disable)",appStateKey:"thinkingEnabled"
    },"permissions.defaultMode":{
      source:"settings",type:"string",description:"Default permission mode for tool usage",options:["default","plan","acceptEdits","dontAsk","auto"]
    },language:{
      source:"settings",type:"string",description:'Preferred language for Claude responses and voice dictation (e.g., "japanese", "spanish")'
    },teammateMode:{
      source:"global",type:"string",description:'How to spawn teammates: "tmux" for traditional tmux, "in-process" for same process, "auto" to choose automatically',options:Hsq
    },...{
      
    },...{
      voiceEnabled:{
        source:"settings",type:"boolean",description:"Enable voice dictation (hold-to-talk)"
      }
    },...{
      remoteControlAtStartup:{
        source:"global",type:"boolean",description:"Enable Remote Control for all sessions (true | false | default)",formatOnRead:()=>FF()
      }
    },...{
      
    }
  }
} /* confidence: 95% */

/* original: MPK */ var composed_value=L(()=>{
  w78();
  y56();
  d87()
} /* confidence: 30% */

/* original: LDK */ var composed_value=L(()=>{
  yL();
  Y2()
} /* confidence: 30% */

/* original: Mg8 */ var composed_value=L(()=>{
  T8();
  O$();
  ZY();
  Y2();
  bX();
  No();
  _8();
  mA();
  h8();
  dq();
  m17();
  t4();
  W66();
  i1();
  dH6();
  g_Y=new Set([pq,$9,Z_,N4,xK])
} /* confidence: 30% */

/* original: FfK */ var composed_value=L(()=>{
  Y2();
  qP();
  _8();
  yK();
  TV();
  i_()
} /* confidence: 30% */

/* original: $ZK */ var composed_value=L(()=>{
  ZY();
  Y2();
  bX()
} /* confidence: 30% */

/* original: u78 */ var composed_value=L(()=>{
  T8();
  Rq6();
  M77();
  Tw();
  ZY();
  Y2();
  bX();
  lP();
  l2();
  _8();
  qv();
  a1();
  l1();
  k8();
  nA();
  $ZK();
  OZK=(yL(),hq(wr))
} /* confidence: 30% */

/* original: EZK */ var composed_value=L(()=>{
  d2();
  mb();
  aC()
} /* confidence: 30% */

/* original: eg8 */ var composed_value=L(()=>{
  _8();
  d8();
  E8();
  a1();
  dq();
  Nz();
  sK6();
  t4();
  CZ();
  eC();
  l1();
  k8();
  p77();
  SV6();
  Ia();
  aC();
  C77();
  ag8={
    minTokens:1e4,minTextBlockMessages:5,maxTokens:40000
  },g77={
    ...ag8
  }
} /* confidence: 30% */

/* original: fGK */ var composed_value=L(()=>{
  _8();
  E8();
  dq();
  oo();
  r8();
  M77()
} /* confidence: 30% */

/* original: LTK */ var composed_value=L(()=>{
  wQ();
  ZY();
  bX();
  oy8();
  CU();
  Mt6();
  p2Y=new Set([pq,$9,Z_,gB8,tP,$h8,"ReadMcpResourceTool",Jb,eN,z46,oL,Y46,jg,EV,OO,_46,FL,ym,aw6,aP,...NTK?[NTK]:[],_N6,...TTK?[TTK]:[],...kTK?[kTK]:[],...VTK?[VTK]:[],Jt6])
} /* confidence: 30% */

/* original: Z$ */ var composed_value=L(()=>{
  E7();
  VS6=w6(D6(),1);
  dTK={
    immediate:0,high:1,medium:2,low:3
  }
} /* confidence: 30% */

/* original: ANK */ var composed_value=L(()=>{
  I3();
  T8();
  Mh();
  dN();
  Lm();
  yq6();
  Ia();
  QN8();
  aC();
  Q78();
  eg8();
  SV6();
  lP();
  E8();
  B$();
  h8();
  a1();
  sR8();
  CR6()
} /* confidence: 30% */

/* original: vFK */ var composed_value=L(()=>{
  t6();
  j3();
  x4();
  l1();
  k8();
  X88();
  T7();
  qi();
  _a1();
  M88();
  kz7();
  Tz7();
  pC6=w6(D6(),1)
} /* confidence: 30% */

/* original: LFK */ var help_h_help=L(()=>{
  t6();
  Pp();
  k8();
  E7();
  yD();
  i1();
  qM6=w6(D6(),1),aSY=["help","-h","--help"]
} /* confidence: 65% */

/* original: idK */ var composed_value=L(()=>{
  ZY();
  Y2();
  bX();
  PV6();
  Vq6();
  d8();
  CuY=[...Mw6,Z_,$9,pq,mj,gL],buY=[N4,xK,WD]
} /* confidence: 30% */

/* original: qnK */ var composed_value=L(()=>{
  E7();
  _O();
  t4();
  u36=w6(D6(),1)
} /* confidence: 30% */

/* original: tl8 */ var composed_value=L(()=>{
  E7();
  y56();
  lrK=w6(D6(),1)
} /* confidence: 30% */

/* original: PoK */ var composed_value=L(()=>{
  t6();
  i6();
  l1();
  mb();
  JoK();
  E7();
  sR8();
  VM6=w6(D6(),1),nQY=w6(D6(),1)
} /* confidence: 30% */

/* original: _n8 */ var composed_value=L(()=>{
  t6();
  Z$();
  k8();
  E7();
  AA6();
  sl8();
  Pp();
  tl8();
  i6();
  X88();
  mb();
  T7();
  sd();
  d8();
  I7();
  F88();
  i2();
  a1();
  CZ();
  KoK();
  q3();
  zoK();
  woK();
  lI8();
  PoK();
  DoK();
  lK=w6(D6(),1),Eb6=w6(D6(),1),KdY=(SO7(),hq(foK)).VoiceIndicator
} /* confidence: 30% */

/* original: HsK */ var composed_value=L(()=>{
  k8();
  E7();
  T8();
  AQ();
  Z$();
  rb();
  Pp();
  Kh();
  i6();
  YQ();
  k1();
  jD();
  F7();
  _8();
  mH();
  B$();
  a1();
  dq();
  t4();
  CZ();
  D0();
  QS6();
  Nt=w6(D6(),1),oM=w6(D6(),1);
  jsK=oM.memo(PlY)
} /* confidence: 30% */

/* original: $tK */ var composed_value=L(()=>{
  El8();
  uA7();
  xn8();
  D58();
  E7();
  aq();
  _8();
  I7();
  a1();
  WC6();
  mM();
  SW=w6(D6(),1)
} /* confidence: 30% */

/* original: GtK */ var composed_value=L(()=>{
  k8();
  E7();
  gA7();
  a1();
  pM6=w6(D6(),1)
} /* confidence: 30% */

/* original: ttK */ var composed_value=L(()=>{
  k8();
  E7();
  QS6();
  Z$();
  AW();
  Tb6();
  Lm();
  Kq();
  T36();
  wW();
  c2();
  Hb();
  FM6=w6(D6(),1)
} /* confidence: 30% */

/* original: Aw7 */ var composed_value=L(()=>{
  Z$();
  k8();
  Zd();
  E7();
  _8();
  w$();
  E8();
  h8();
  RN8();
  s78();
  $H6();
  zm8();
  zy6();
  Ow7();
  xQ8();
  g2();
  y58=w6(D6(),1)
} /* confidence: 30% */

/* original: D65 */ var composed_value=L(()=>{
  t6();
  NC6();
  AW();
  i6();
  E7();
  k1();
  j3();
  x4();
  CW=w6(D6(),1),jw7=w6(D6(),1)
} /* confidence: 30% */

/* original: Z65 */ var composed_value=L(()=>{
  E7();
  mb6=w6(D6(),1)
} /* confidence: 30% */

/* original: B65 */ var composed_value=L(()=>{
  MT6();
  l1();
  m65();
  a1();
  p65=w6(D6(),1)
} /* confidence: 30% */

/* original: P85 */ var composed_value=L(()=>{
  Z$();
  T8();
  E7();
  WM();
  i1();
  Q58=w6(D6(),1)
} /* confidence: 30% */

/* original: D85 */ var composed_value=L(()=>{
  t6();
  NN();
  T8();
  Z$();
  i6();
  Zd();
  E7();
  _8();
  d8();
  Fv=w6(D6(),1)
} /* confidence: 30% */

/* original: L85 */ var composed_value=L(()=>{
  t6();
  T8();
  Z$();
  E7();
  k1();
  _8();
  h8();
  y85();
  d56();
  i1();
  yw7();
  Di8=w6(D6(),1)
} /* confidence: 30% */

/* original: m85 */ var composed_value=L(()=>{
  t6();
  T8();
  Z$();
  i6();
  E7();
  _8();
  bc=w6(D6(),1),I85=w6(D6(),1)
} /* confidence: 30% */

/* original: i85 */ var composed_value=L(()=>{
  t6();
  Z$();
  i6();
  k8();
  YQ();
  X88();
  aE8();
  E7();
  T7();
  qi();
  T8();
  rM6=w6(D6(),1),oM6=w6(D6(),1)
} /* confidence: 30% */

/* original: j15 */ var composed_value=L(()=>{
  T8();
  Z$();
  E7();
  l58=w6(D6(),1)
} /* confidence: 30% */

/* original: P15 */ var composed_value=L(()=>{
  t6();
  Z$();
  E7();
  $Z();
  I7();
  T8();
  Ti8=w6(D6(),1)
} /* confidence: 30% */

/* original: xw7 */ var composed_value=L(()=>{
  Z$();
  AW();
  AA6();
  _A6();
  i6();
  MQ();
  ks6();
  tl8();
  WP=w6(D6(),1),zsY=(Sz7(),hq(KUK));
  wsY={
    key:" ",ctrl:!1,alt:!1,shift:!1,meta:!1,super:!1
  }
} /* confidence: 30% */

/* original: ti8 */ var composed_value=L(()=>{
  E8();
  dq();
  Mt6();
  i1();
  oo();
  r8()
} /* confidence: 30% */

/* original: Wz5 */ function utility_fn(){
  this.size=0,this.__data__={
    hash:new xr8,map:new(ze||Ke),string:new xr8
  }
} /* confidence: 40% */

