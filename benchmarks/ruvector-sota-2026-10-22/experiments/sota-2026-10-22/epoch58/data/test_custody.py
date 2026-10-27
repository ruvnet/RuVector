"""Synthetic structural tests only: no official archive, DOC17 text or Gold access."""
import copy,hashlib,importlib.util,json,math,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load(name):
 spec=importlib.util.spec_from_file_location('e58_test_'+name,HERE/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class CustodyStructural(unittest.TestCase):
 def setUp(self):
  self.c=load('custody');self.s=load('span_json');self.catalog=load('source_catalog')
  self.fixture={'labels':{h:{'hypothesis':'Synthetic hypothesis '+h}for h in self.c.HYPOTHESES},'documents':[{'id':i,'file_name':'synthetic.txt'if i==17 else'UNSELECTED_NAME','text':'Synthetic sentence one. Synthetic sentence two.'if i==17 else'UNSELECTED_TEXT_MUST_NOT_DECODE','annotation_sets':[{'annotations':{h:{'choice':'OPAQUE_SYNTHETIC_TOKEN_'+h}for h in self.c.HYPOTHESES}}]}for i in range(61)]}
  self.c.DOC_SHA=self.c.sha('synthetic sentence one. synthetic sentence two.'.encode());self.c.FILENAME_SHA=self.c.sha(b'synthetic.txt')
 def extract(self,fixture=None):
  b=json.dumps(self.fixture if fixture is None else fixture,separators=(',',':')).encode();self.c.MEMBER_BYTES=len(b);self.c.MEMBER_SHA=self.c.sha(b);p={'actual_hypothesis_exposed_case_ids':[],'source_text_exposed_case_ids':[]}
  return self.c.extract(b,self.s,self.catalog,p),p
 def test_exact_source_selection_and_opaque_tokens(self):
  (doc,cases,tokens,counts),p=self.extract();self.assertEqual(len(cases),17);self.assertEqual(len(tokens),17);self.assertEqual(counts['selected_document_text_decodes'],1);self.assertEqual(counts['semantic_gold_decodes'],0);self.assertEqual(counts['unselected_document_or_choice_decodes'],0);self.assertEqual(p['source_text_exposed_case_ids'],list(self.c.IDS));self.assertEqual(p['actual_hypothesis_exposed_case_ids'],list(self.c.IDS));self.assertEqual(''.join(x['sentence']for x in doc['evidence']),'Synthetic sentence one. Synthetic sentence two.')
 def test_partial_source_exposure_survives_hash_rejection(self):
  self.c.DOC_SHA='0'*64;b=json.dumps(self.fixture).encode();self.c.MEMBER_BYTES=len(b);self.c.MEMBER_SHA=self.c.sha(b);p={'actual_hypothesis_exposed_case_ids':[],'source_text_exposed_case_ids':[]}
  with self.assertRaisesRegex(ValueError,'EXACT_DOC17_NORMALIZED_SOURCE'):self.c.extract(b,self.s,self.catalog,p)
  self.assertEqual(p['source_text_exposed_case_ids'],list(self.c.IDS));self.assertEqual(p['value_decoder_counts']['selected_document'],1);self.assertEqual(p['value_decoder_counts']['selected_hypothesis'],17)
 def test_source_exposure_callbacks_bracket_actual_decode(self):
  b=json.dumps(self.fixture).encode();self.c.MEMBER_BYTES=len(b);self.c.MEMBER_SHA=self.c.sha(b);p={'actual_hypothesis_exposed_case_ids':[],'source_text_exposed_case_ids':[]};events=[]
  self.c.extract(b,self.s,self.catalog,p,lambda:events.append(('BEFORE',p['value_decoder_counts']['selected_document'])),lambda:events.append(('AFTER',p['value_decoder_counts']['selected_document'])))
  self.assertEqual(events,[('BEFORE',0),('AFTER',1)])
 def test_duplicate_document_identity_rejected(self):
  x=copy.deepcopy(self.fixture);x['documents'][-1]['id']=0
  with self.assertRaisesRegex(ValueError,'UNIQUE_SOURCE_ID'):self.extract(x)
 def test_missing_choice_or_hypothesis_is_not_imputed(self):
  x=copy.deepcopy(self.fixture);del x['documents'][17]['annotation_sets'][0]['annotations'][self.c.HYPOTHESES[-1]]
  with self.assertRaisesRegex(ValueError,'EXACT17_ANNOTATIONS'):self.extract(x)
 def test_catalog_keeps_exact_offsets_and_rejects_gaps(self):
  spans=self.catalog.sentence_catalog('Number 3.14.  Dr. A signed.\nNext!');self.catalog.validate_spans(spans);self.assertEqual(''.join(x['sentence']for x in spans),'Number 3.14.  Dr. A signed.\nNext!');spans[-1]['start']+=1
  with self.assertRaisesRegex(ValueError,'NONCONTIGUOUS'):self.catalog.validate_spans(spans)
 def forecasts(self):
  a={'kind':'SYNTHETIC_STRUCTURAL_TEST_ONLY','plan_ref':{'sha256':'a'*64},'source_freeze_ref':{'sha256':'b'*64},'frozen_window_count':2}
  rows=[{'case_id':cid,'arm':arm,'probabilities':[.2,.3,.5],'log_probabilities':[math.log(.2),math.log(.3),math.log(.5)],'elapsed_seconds':.01,'actual_model_forward_calls':1,'actual_pair_inferences':2,'context_window_ids':[0,1],'all_source_windows_covered':True,'early_exit':False,'numeric_flags':[],'prediction':'NotMentioned'}for cid in self.c.IDS for arm in self.c.ARMS]
  return {'schema_version':1,'epoch':58,'kind':a['kind'],'phase':'dev','plan_sha256':'a'*64,'source_freeze_sha256':'b'*64,'labels':list(self.c.LABELS),'arms':list(self.c.ARMS),'selected_query_ids':list(self.c.IDS),'reserved_ids':list(self.c.IDS),'rows':rows},a
 def test_exact_51_forecasts_accepted_without_reading_labels(self):
  f,a=self.forecasts();self.assertEqual(len(self.c.validate_forecasts(f,a)),51)
 def test_missing_duplicate_or_reordered_forecasts_rejected(self):
  for transform in [lambda f:f['rows'].pop(),lambda f:f['rows'].__setitem__(1,copy.deepcopy(f['rows'][0])),lambda f:f['rows'].reverse()]:
   f,a=self.forecasts();transform(f)
   with self.assertRaises(ValueError):self.c.validate_forecasts(f,a)
 def test_zero_nan_or_inconsistent_probabilities_rejected(self):
  for key,value in [('probabilities',[0,.5,.5]),('probabilities',[float('nan'),.5,.5]),('log_probabilities',[-1.,-1.,-1.])]:
   f,a=self.forecasts();f['rows'][0][key]=value
   with self.assertRaises(ValueError):self.c.validate_forecasts(f,a)
 def test_retrieval_miss_never_proves_notmentioned(self):
  f,a=self.forecasts();r=f['rows'][0];r.update(context_window_ids=[0],actual_pair_inferences=1,all_source_windows_covered=False,early_exit=True)
  with self.assertRaisesRegex(ValueError,'INSUFFICIENT_EVIDENCE'):self.c.validate_forecasts(f,a)
 def test_numeric_flags_disable_exit_and_full_reference_cannot_exit(self):
  for index,flags in [(0,['claim_has_numerals']),(2,[])]:
   f,a=self.forecasts();f['rows'][index].update(numeric_flags=flags,early_exit=True)
   with self.assertRaises(ValueError):self.c.validate_forecasts(f,a)
 def test_boolean_observed_counts_cannot_authorize_gold(self):
  for counts in [True,{}]:
   q={'checks_passed':True,'all_owned_processes_closed':True,'actual_execution_independently_derived':counts}
   with self.assertRaisesRegex(ValueError,'INDEPENDENT_OBSERVED_COUNTS'):self.c.physically_closed(q)
 def test_catalog_source_slices_and_claims_revalidated_without_tokenizer(self):
  (doc,cases,_,_),_=self.extract();kind='SYNTHETIC_STRUCTURAL_TEST_ONLY';bank={'epoch':58,'kind':kind,'source_checkpoint':self.c.BASE,'role':'dev','documents':[doc],'cases':cases};text=''.join(x['sentence']for x in doc['evidence']);catalog={'schema_version':1,'epoch':58,'kind':kind,'role':'dev','selected_query_ids':list(self.c.IDS),'reserved_ids':list(self.c.IDS),'document_id':17,'source_document_sha256':doc['source_document_sha256'],'frozen_window_count':1,'windows':[{'window_id':0,'start':0,'end':len(text),'source_unit_ids':list(range(len(doc['evidence']))),'text':text,'text_sha256':self.c.sha(text.encode())}],'cases':cases,'all17_pair_capacity_checked':True,'BGE_query_prefix':'Synthetic prefix','max_tokens':512,'max_windows':24,'no_label_or_model_output_access':True}
  self.assertEqual(len(self.c.validate_catalog(catalog,bank,1,kind)),1)
  catalog['windows'][0]['text']='changed source';catalog['windows'][0]['text_sha256']=self.c.sha(b'changed source')
  with self.assertRaisesRegex(ValueError,'EXACT_UNREWRITTEN'):self.c.validate_catalog(catalog,bank,1,kind)
 def test_joint_actual_counts_match_exact42_raw_worker_counters(self):
  f,a=self.forecasts();schema=json.loads((HERE/'COUNTER_SCHEMA.v1.json').read_text());identity={'pid':99999999,'startticks':1,'pgid':99999999,'sid':99999999};raws={};native={};observed={};mapping={};q={'report_type':'JOINT_LATE_GOLD_CLOSURE','epoch':58,'kind':a['kind'],'owned_process_identities':[identity],'plan_ref':a['plan_ref'],'source_freeze_ref':a['source_freeze_ref']};a.update(sealed_forecasts_ref={'path':'SYNTHETIC_FORECAST'},catalog_ref={'path':'SYNTHETIC_CATALOG'})
  for phase in ['engineering','dev']:
   count={k:0 for k in schema['keys']}
   for k in schema['positive_actual_'+phase+'_completed_keys']:count[k]=1
   if phase=='dev':count['forecasts_completed']=51
   for attempt,complete in schema['attempt_completion_pairs']:count[attempt]=count[complete]
   observed[phase]=count;raws[phase]={'path':'SYNTHETIC_RAW_'+phase};native[phase]={'path':'SYNTHETIC_NATIVE_'+phase};worker={'schema_version':1,'epoch':58,'kind':a['kind'],'phase':phase,'errors':[],'plan_sha256':a['plan_ref']['sha256'],'source_freeze_sha256':a['source_freeze_ref']['sha256'],'counters':copy.deepcopy(count),'native_results_ref':native[phase],'owned_process_identities':[identity],'status':'ENGINEERING_EXECUTED_PENDING_INDEPENDENT_AUDIT'if phase=='engineering'else'DEV_FORECASTS_SEALED_NO_GOLD','rows':f['rows']if phase=='dev'else[],'sealed_forecasts_ref':a['sealed_forecasts_ref'],'catalog_ref':a['catalog_ref'],'selected_query_ids':list(self.c.IDS),'reserved_ids':list(self.c.IDS)};mapping[raws[phase]['path']]=worker
  q.update(actual_execution_independently_derived=observed,raw_worker_results=raws,native_result_refs=native);self.c.load_ref=lambda r:mapping[r['path']];self.c.read_ref=lambda r:b'SYNTHETIC_NATIVE';self.c.validate_observed_execution(q,a,f)
  for bad in [{},True,{'arbitrary':1}]:
   broken=copy.deepcopy(q);broken['actual_execution_independently_derived']['dev']=bad
   with self.assertRaisesRegex(ValueError,'EXACT42_NAMED'):self.c.validate_observed_execution(broken,a,f)
  broken=copy.deepcopy(q);broken['actual_execution_independently_derived']['dev']['forecasts_completed']=50
  with self.assertRaisesRegex(ValueError,'EXACTLY_EQUAL_PHYSICAL_RAW_WORKER'):self.c.validate_observed_execution(broken,a,f)
  for badtype in [True,1.0]:
   mapping[raws['dev']['path']]['counters']['NLI_model_initializations_completed']=badtype
   with self.assertRaisesRegex(ValueError,'RAW_WORKER_EXACT42_STRICT'):self.c.validate_observed_execution(q,a,f)

if __name__=='__main__':unittest.main()
