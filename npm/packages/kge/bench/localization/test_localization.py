import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import io
import tarfile
import unittest
from prepare import chunks,gold_files,snapshot_documents
from retrieval import bm25,evaluate,file_ranking
from graph_rerank import FEATURES,build_graph,fit_pairwise,graph_features,score_ranking,select_l2_loro


class LocalizationTests(unittest.TestCase):
    def test_legacy_syntax_keeps_candidate(self):
        rows=chunks("pkg/a.py","print 'legacy'\n")
        self.assertEqual(rows[0]["kind"],"syntax-fallback")
        self.assertEqual(rows[0]["path"],"pkg/a.py")

    def test_source_never_executes(self):
        rows=chunks("pkg/a.py","raise RuntimeError('do not execute')\ndef f():\n return 1\n")
        self.assertEqual(len(rows),2)

    def test_archive_traversal_rejected(self):
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
            item=tarfile.TarInfo("root/../evil.py");item.size=1;archive.addfile(item,io.BytesIO(b"x"))
        with self.assertRaises(ValueError):snapshot_documents(buffer.getvalue(),"pkg/")

    def test_ranking_and_complete_universe(self):
        docs=[dict(path="a.py",text="decode json response"),dict(path="b.py",text="network socket")]
        ranking=file_ranking(docs,bm25("decode json",docs))
        self.assertEqual(ranking[0]["path"],"a.py")
        self.assertEqual(evaluate(ranking,["a.py"])["mrr"],1)
        with self.assertRaises(ValueError):evaluate(ranking,["missing.py"])
        with self.assertRaises(ValueError):file_ranking(docs,[float("nan"),0])

    def test_labels_separate_and_production_only(self):
        patch="--- a/pkg/a.py\n+++ b/pkg/a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py"
        self.assertEqual(gold_files(patch,"pkg/"),["pkg/a.py"])




class ReceiptTests(unittest.TestCase):
    def run_fixture(self, corruption=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folders={n:root/n for n in ['run','data','model','code','dataset']}
            for p in folders.values():p.mkdir()
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            def write(path, value):
                path.write_text(json.dumps(value)+"\n")
                return digest(path)
            metric=dict(mrr=1.,hit1=1,recall5=1.,recall10=1.,goldCount=1,goldRanks=[1])
            artifacts={}
            for label in ['model','code','dataset']:
                name='data.parquet' if label=='dataset' else 'config.json'
                artifact=folders[label]/name;artifact.write_bytes(b'fixture')
                metadata=folders[label]/'.cache/huggingface/download';metadata.mkdir(parents=True)
                (metadata/(name+'.metadata')).write_text('a'*40+'\n'+digest(artifact)+'\n0\n')
                if label!='dataset':artifacts[str(artifact.resolve())]=digest(artifact)
            cases=[];records=[]
            for i in range(6):
                ident=f'case-{i}'
                case=dict(id=ident,repo='fixture/repo',baseCommit='b'*40,documents=[dict(path='a.py',text='x')])
                input_hash=write(folders['data']/(ident+'.json'),case)
                label_hash=write(folders['data']/(ident+'.labels.json'),dict(id=ident,goldFiles=['a.py']))
                predictions={arm:[dict(path='a.py',score=1.)] for arm in ['bm25','swerank']}
                pred_hash=write(folders['run']/(ident+'.predictions.json'),predictions)
                cases.append(dict(id=ident,repo=case['repo'],baseCommit=case['baseCommit'],inputHash=input_hash,
                                  labelHash=label_hash,sourceFileHashes={'a.py':'fixture'},fileCount=1,documentCount=1))
                records.append(dict(id=ident,repo=case['repo'],baseCommit=case['baseCommit'],inputHash=input_hash,
                    labelHash=label_hash,predictionHash=pred_hash,documents=1,files=1,bm25=metric.copy(),swerank=metric.copy(),
                    indexing={'seconds':1.},queryEncoding={'seconds':1.},rankingSeconds=0.,bm25Seconds=0.))
            manifest=dict(sourceHashes={},artifacts=artifacts,modelRevision='a'*40,codeRevision='a'*40,
                          data=dict(cases=cases,revision='a'*40,parquetHash=digest(folders['dataset']/'data.parquet')))
            manifest_path=root/'manifest.json';write(manifest_path,manifest)
            aggregate={arm:{k:1. for k in ['mrr','hit1','recall5','recall10']} for arm in ['bm25','swerank']}
            receipt=dict(manifest=manifest,records=records,aggregate=aggregate,initializationSeconds=1.)
            if corruption=='aggregate':receipt['aggregate']={}
            if corruption=='rank':receipt['records'][0]['swerank']['mrr']=.5
            if corruption=='provenance':(folders['model']/'.cache/huggingface/download/config.json.metadata').write_text('a'*40+'\n'+'0'*64+'\n0\n')
            write(folders['run']/'receipt.json',receipt)
            command=[sys.executable,str(Path(__file__).with_name('verify.py'))]
            for name,path in folders.items():command.extend(['--'+name,str(path)])
            command.extend(['--manifest',str(manifest_path),'--output',str(root/'verified.json'),'--source-commit','c'*40])
            result=subprocess.run(command,capture_output=True,text=True)
            return result.returncode,(root/'verified.json').exists(),result.stderr

    def test_valid_receipt(self):
        code,written,error=self.run_fixture()
        self.assertEqual(code,0,error);self.assertTrue(written)

    def test_corrupt_receipts_rejected(self):
        for corruption in ['aggregate','rank','provenance']:
            with self.subTest(corruption=corruption):
                code,written,_=self.run_fixture(corruption)
                self.assertNotEqual(code,0);self.assertFalse(written)


class GraphRerankerTests(unittest.TestCase):
    def fixture(self):
        sources={
            "pkg/a.py":"from pkg import b\ndef entry():\n    return helper()\n",
            "pkg/b.py":"class Base:\n    pass\ndef helper():\n    return 1\n",
            "pkg/c.py":"from pkg.b import Base\nclass Child(Base):\n    pass\n",
        }
        return sources

    def test_graph_is_typed_and_inductive(self):
        graph=build_graph(self.fixture())
        self.assertIn("pkg/b.py",graph["import"]["pkg/a.py"])
        self.assertIn("pkg/b.py",graph["call"]["pkg/a.py"])
        self.assertIn("pkg/b.py",graph["inherit"]["pkg/c.py"])
        self.assertNotIn("pkg/a.py",graph["import"].get("pkg/c.py",()))

    def test_features_train_and_apply_with_unseen_paths(self):
        sources=self.fixture();graph=build_graph(sources)
        docs=[dict(path=path,text=source) for path,source in sources.items()]
        train_ranking=file_ranking(docs,[3.,2.,1.])
        rankings={"bm25":train_ranking,"swerank":train_ranking[::-1]}
        features=graph_features(graph,rankings,[d["path"] for d in docs])
        weights=fit_pairwise([dict(features=features,gold=["pkg/b.py"])],l2=0.1)
        self.assertEqual(len(weights),len(FEATURES))
        test_sources={"new/a.py":"from new import b\ndef entry(): return helper()",
                      "new/b.py":"def helper(): return 1"}
        test_graph=build_graph(test_sources)
        test_docs=[dict(path=path,text=source) for path,source in test_sources.items()]
        ranking=file_ranking(test_docs,[1.,0.])
        test_rankings={"bm25":ranking,"swerank":ranking[::-1]}
        test_features=graph_features(test_graph,test_rankings,[d["path"] for d in test_docs])
        result=score_ranking(ranking,test_features,weights)
        self.assertEqual({r["path"] for r in result},{"new/a.py","new/b.py"})

    def test_invalid_pair_data_rejected(self):
        with self.assertRaises(ValueError):fit_pairwise([],l2=1.0)
        with self.assertRaises(ValueError):fit_pairwise([dict(features={"x":(1.,)},gold=["missing"])])

    def test_loro_selects_deterministically_and_retrains_on_other_repos(self):
        examples=[]
        for i in range(3):
            good,bad=f"repo{i}/good.py",f"repo{i}/bad.py"
            features={good:(1.,1.,1.,0.,0.,0.,0.,0.),bad:(0.,0.,0.,0.,0.,0.,0.,0.)}
            ranking=[dict(path=bad,score=1.),dict(path=good,score=0.)]
            examples.append(dict(repo=f"org/repo{i}",features=features,ranking=ranking,gold=[good]))
        selected=select_l2_loro(examples)
        self.assertEqual(selected["l2"],0.01)
        self.assertEqual(len(selected["results"]),4)
        self.assertEqual(len(selected["results"][0]["repositoryMrr"]),3)


if __name__=="__main__":unittest.main()
