---
base_model: microsoft/deberta-v3-base
datasets:
- nyu-mll/glue
- aps/super_glue
- facebook/anli
- tasksource/babi_nli
- sick
- snli
- scitail
- hans
- alisawuffles/WANLI
- tasksource/recast
- sileod/probability_words_nli
- joey234/nan-nli
- pietrolesci/nli_fever
- pietrolesci/breaking_nli
- pietrolesci/conj_nli
- pietrolesci/fracas
- pietrolesci/dialogue_nli
- pietrolesci/mpe
- pietrolesci/dnc
- pietrolesci/recast_white
- pietrolesci/joci
- pietrolesci/robust_nli
- pietrolesci/robust_nli_is_sd
- pietrolesci/robust_nli_li_ts
- pietrolesci/gen_debiased_nli
- pietrolesci/add_one_rte
- tasksource/imppres
- hlgd
- paws
- medical_questions_pairs
- Anthropic/model-written-evals
- truthful_qa
- nightingal3/fig-qa
- tasksource/bigbench
- blimp
- cos_e
- cosmos_qa
- dream
- openbookqa
- qasc
- quartz
- quail
- head_qa
- sciq
- social_i_qa
- wiki_hop
- wiqa
- piqa
- hellaswag
- pkavumba/balanced-copa
- 12ml/e-CARE
- art
- winogrande
- codah
- ai2_arc
- definite_pronoun_resolution
- swag
- math_qa
- metaeval/utilitarianism
- mteb/amazon_counterfactual
- SetFit/insincere-questions
- SetFit/toxic_conversations
- turingbench/TuringBench
- trec
- tals/vitaminc
- hope_edi
- strombergnlp/rumoureval_2019
- ethos
- tweet_eval
- discovery
- pragmeval
- silicone
- lex_glue
- papluca/language-identification
- imdb
- rotten_tomatoes
- ag_news
- yelp_review_full
- financial_phrasebank
- poem_sentiment
- dbpedia_14
- amazon_polarity
- app_reviews
- hate_speech18
- sms_spam
- humicroedit
- snips_built_in_intents
- hate_speech_offensive
- yahoo_answers_topics
- pacovaldez/stackoverflow-questions
- zapsdcn/hyperpartisan_news
- zapsdcn/sciie
- zapsdcn/citation_intent
- go_emotions
- allenai/scicite
- liar
- relbert/lexical_relation_classification
- tasksource/linguisticprobing
- tasksource/crowdflower
- metaeval/ethics
- emo
- google_wellformed_query
- tweets_hate_speech_detection
- has_part
- blog_authorship_corpus
- launch/open_question_type
- health_fact
- commonsense_qa
- mc_taco
- ade_corpus_v2
- prajjwal1/discosense
- circa
- PiC/phrase_similarity
- copenlu/scientific-exaggeration-detection
- quarel
- mwong/fever-evidence-related
- numer_sense
- dynabench/dynasent
- raquiba/Sarcasm_News_Headline
- sem_eval_2010_task_8
- demo-org/auditor_review
- medmcqa
- RuyuanWan/Dynasent_Disagreement
- RuyuanWan/Politeness_Disagreement
- RuyuanWan/SBIC_Disagreement
- RuyuanWan/SChem_Disagreement
- RuyuanWan/Dilemmas_Disagreement
- lucasmccabe/logiqa
- wiki_qa
- tasksource/cycic_classification
- tasksource/cycic_multiplechoice
- tasksource/sts-companion
- tasksource/commonsense_qa_2.0
- tasksource/lingnli
- tasksource/monotonicity-entailment
- tasksource/arct
- tasksource/scinli
- tasksource/naturallogic
- onestop_qa
- demelin/moral_stories
- corypaik/prost
- aps/dynahate
- metaeval/syntactic-augmentation-nli
- tasksource/autotnli
- lasha-nlp/CONDAQA
- openai/webgpt_comparisons
- Dahoas/synthetic-instruct-gptj-pairwise
- metaeval/scruples
- metaeval/wouldyourather
- metaeval/defeasible-nli
- tasksource/help-nli
- metaeval/nli-veridicality-transitivity
- tasksource/lonli
- tasksource/dadc-limit-nli
- ColumbiaNLP/FLUTE
- tasksource/strategy-qa
- openai/summarize_from_feedback
- tasksource/folio
- yale-nlp/FOLIO
- tasksource/tomi-nli
- tasksource/avicenna
- stanfordnlp/SHP
- GBaker/MedQA-USMLE-4-options-hf
- sileod/wikimedqa
- declare-lab/cicero
- amydeng2000/CREAK
- tasksource/mutual
- inverse-scaling/NeQA
- inverse-scaling/quote-repetition
- inverse-scaling/redefine-math
- tasksource/puzzte
- tasksource/implicatures
- race
- tasksource/race-c
- tasksource/spartqa-yn
- tasksource/spartqa-mchoice
- tasksource/temporal-nli
- riddle_sense
- tasksource/clcd-english
- maximedb/twentyquestions
- metaeval/reclor
- tasksource/counterfactually-augmented-imdb
- tasksource/counterfactually-augmented-snli
- metaeval/cnli
- tasksource/boolq-natural-perturbations
- metaeval/acceptability-prediction
- metaeval/equate
- tasksource/ScienceQA_text_only
- Jiangjie/ekar_english
- tasksource/implicit-hate-stg1
- metaeval/chaos-mnli-ambiguity
- IlyaGusev/headline_cause
- tasksource/logiqa-2.0-nli
- tasksource/oasst2_dense_flat
- sileod/mindgames
- metaeval/ambient
- metaeval/path-naturalness-prediction
- civil_comments
- AndyChiang/cloth
- AndyChiang/dgen
- tasksource/I2D2
- webis/args_me
- webis/Touche23-ValueEval
- tasksource/starcon
- PolyAI/banking77
- tasksource/ConTRoL-nli
- tasksource/tracie
- tasksource/sherliic
- tasksource/sen-making
- tasksource/winowhy
- tasksource/robustLR
- CLUTRR/v1
- tasksource/logical-fallacy
- tasksource/parade
- tasksource/cladder
- tasksource/subjectivity
- tasksource/MOH
- tasksource/VUAC
- tasksource/TroFi
- sharc_modified
- tasksource/conceptrules_v2
- metaeval/disrpt
- tasksource/zero-shot-label-nli
- tasksource/com2sense
- tasksource/scone
- tasksource/winodict
- tasksource/fool-me-twice
- tasksource/monli
- tasksource/corr2cause
- lighteval/lsat_qa
- tasksource/apt
- zeroshot/twitter-financial-news-sentiment
- tasksource/icl-symbol-tuning-instruct
- tasksource/SpaceNLI
- sihaochen/propsegment
- HannahRoseKirk/HatemojiBuild
- tasksource/regset
- tasksource/esci
- lmsys/chatbot_arena_conversations
- neurae/dnd_style_intents
- hitachi-nlp/FLD.v2
- tasksource/SDOH-NLI
- allenai/scifact_entailment
- tasksource/feasibilityQA
- tasksource/simple_pair
- tasksource/AdjectiveScaleProbe-nli
- tasksource/resnli
- tasksource/SpaRTUN
- tasksource/ReSQ
- tasksource/semantic_fragments_nli
- MoritzLaurer/dataset_train_nli
- tasksource/stepgame
- tasksource/nlgraph
- tasksource/oasst2_pairwise_rlhf_reward
- tasksource/hh-rlhf
- tasksource/ruletaker
- qbao775/PARARULE-Plus
- tasksource/proofwriter
- tasksource/logical-entailment
- tasksource/nope
- tasksource/LogicNLI
- kiddothe2b/contract-nli
- AshtonIsNotHere/nli4ct_semeval2024
- tasksource/lsat-ar
- tasksource/lsat-rc
- AshtonIsNotHere/biosift-nli
- tasksource/brainteasers
- Anthropic/persuasion
- erbacher/AmbigNQ-clarifying-question
- tasksource/SIGA-nli
- unigram/FOL-nli
- tasksource/goal-step-wikihow
- GGLab/PARADISE
- tasksource/doc-nli
- tasksource/mctest-nli
- tasksource/patent-phrase-similarity
- tasksource/natural-language-satisfiability
- tasksource/idioms-nli
- tasksource/lifecycle-entailment
- nvidia/HelpSteer
- nvidia/HelpSteer2
- sadat2307/MSciNLI
- pushpdeep/UltraFeedback-paired
- tasksource/AES2-essay-scoring
- tasksource/english-grading
- tasksource/wice
- Dzeniks/hover
- tasksource/tasksource_dpo_pairs
library_name: transformers
pipeline_tag: zero-shot-classification
tags:
  - text-classification
  - zero-shot-classification
license: apache-2.0
---

# Model Card for Model ID

deberta-v3-base with context length of 1280 fine-tuned on tasksource for 250k steps. I oversampled long NLI tasks (ConTRoL, doc-nli).
Training data include helpsteer v1/v2, logical reasoning tasks (FOLIO, FOL-nli, LogicNLI...), OASST, hh/rlhf, linguistics oriented NLI tasks, tasksource-dpo, fact verification tasks.

This checkpoint has strong zero-shot validation performance on many tasks (e.g. 70% on WNLI), and can be used for:
- Zero-shot entailment-based classification for arbitrary labels [ZS].
- Natural language inference [NLI]
- Further fine-tuning on a new task or tasksource task (classification, token classification, reward modeling or multiple-choice) [FT].

| dataset                   |   accuracy |
|:----------------------------|----------------:|
| anli/a1                     |            63.3 |
| anli/a2                     |            47.2 |
| anli/a3                     |            49.4 |
| nli_fever                   |            79.4 |
| FOLIO                       |            61.8 |
| ConTRoL-nli                 |            63.3 |
| cladder                     |            71.1 |
| zero-shot-label-nli         |            74.4 |
| chatbot_arena_conversations |            72.2 |
| oasst2_pairwise_rlhf_reward |            73.9 |
| doc-nli                     |            90.0 |

Zero-shot GPT-4 scores 61% on FOLIO (logical reasoning), 62% on cladder (probabilistic reasoning) and 56.4% on ConTRoL (long context NLI).

# [ZS] Zero-shot classification pipeline
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",model="tasksource/deberta-base-long-nli")

text = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(text, candidate_labels)
```
NLI training data of this model includes [label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli), a NLI dataset specially constructed to improve this kind of zero-shot classification.

# [NLI] Natural language inference pipeline

```python
from transformers import pipeline
pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli")
pipe([dict(text='there is a cat',
  text_pair='there is a black cat')]) #list of (premise,hypothesis)
# [{'label': 'neutral', 'score': 0.9952911138534546}]
```

# [TA] Tasksource-adapters: 1 line access to hundreds of tasks 

```python
# !pip install tasknet
import tasknet as tn
pipe = tn.load_pipeline('tasksource/deberta-base-long-nli','glue/sst2') # works for 500+ tasksource tasks
pipe(['That movie was great !', 'Awful movie.'])
# [{'label': 'positive', 'score': 0.9956}, {'label': 'negative', 'score': 0.9967}]
```
The list of tasks is available in model config.json.
This is more efficient than ZS since it requires only one forward pass per example, but it is less flexible.


# [FT] Tasknet: 3 lines fine-tuning

```python
# !pip install tasknet
import tasknet as tn
hparams=dict(model_name='tasksource/deberta-base-long-nli', learning_rate=2e-5)
model, trainer = tn.Model_Trainer([tn.AutoTask("glue/rte")], hparams)
trainer.train()
```


# Citation

More details on this [article:](https://aclanthology.org/2024.lrec-main.1361/) 
```
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361",
    pages = "15655--15684",
}
```
