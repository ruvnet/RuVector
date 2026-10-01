"""Real CPU pretrained causal-LM LoRA training and handle-bound generation."""
from contextlib import contextmanager
import threading
import time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import HTTPException

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
TRAIN_STEPS = 2
TRAIN_TOKENS = 128
PROMPT_TOKENS = 512


class Engine:
    def __init__(self):
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                dtype=torch.float32, attn_implementation='eager')
        self.model = get_peft_model(base, self.config(2))
        self.lock = threading.Lock()
        self.model.eval()

    @staticmethod
    def config(rank):
        return LoraConfig(r=rank, lora_alpha=rank, target_modules=['q_proj', 'v_proj'],
                          lora_dropout=0, task_type='CAUSAL_LM', bias='none')

    @contextmanager
    def execution(self, deadline):
        if not self.lock.acquire(timeout=1):
            raise HTTPException(503, "Runtime busy; retry after current operation")
        try:
            if deadline <= time.time():
                raise HTTPException(401, "Authority expired before execution")
            yield
        finally:
            self.lock.release()

    def adapt(self, request, handle, directory, deadline):
        started = time.monotonic()
        with self.execution(deadline):
            self.model.add_adapter(handle, self.config(request.rank))
            self.model.set_adapter(handle)
            try:
                self.model.train()
                optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=0.001)
                tokens = 0
                losses = []
                # User summaries are training data. Evaluation holdouts never enter here.
                for _ in range(TRAIN_STEPS):
                    for summary in request.interaction_summaries:
                        if deadline <= time.time():
                            raise HTTPException(401, "Authority expired during training")
                        batch = self.tokenizer(summary, return_tensors='pt', max_length=TRAIN_TOKENS, truncation=True)
                        count = batch.input_ids.shape[1]
                        if count < 2:
                            raise ValueError('Summary must tokenize to at least two tokens')
                        output = self.model(**batch, labels=batch.input_ids)
                        loss = output.loss * request.quality
                        if not torch.isfinite(loss):
                            raise ValueError('Non-finite training loss')
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        tokens += count
                        losses.append(float(output.loss.detach()))
                changed = any(torch.count_nonzero(p.detach()).item() > 0 for n, p in self.model.named_parameters()
                              if handle in n and 'lora_B' in n)
                if not changed:
                    raise ValueError('Training produced no changed adapter weights')
                self.model.save_pretrained(directory, selected_adapters=[handle], safe_serialization=True)
                return {'training_tokens': tokens, 'training_loss': losses[-1],
                        'elapsed_ms': round((time.monotonic() - started) * 1000),
                        'artifact_path': handle}
            finally:
                self.model.eval()
                self.model.set_adapter('default')
                self.model.delete_adapter(handle)

    def complete(self, request, artifact=None, deadline=float("inf")):
        started = time.monotonic()
        with self.execution(deadline):
            name = 'evaluation'
            if artifact:
                self.model.load_adapter(artifact, adapter_name=name, is_trainable=False)
                self.model.set_adapter(name)
            try:
                self.model.eval()
                # Same serialization for baseline and candidate; adapter is the only difference.
                prompt = '\n'.join(f'{m.role}: {m.content}' for m in request.messages) + '\nassistant:'
                batch = self.tokenizer(prompt, return_tensors='pt', truncation=False)
                if batch.input_ids.shape[1] > PROMPT_TOKENS:
                    raise ValueError('Prompt exceeds 512 tokens')
                with torch.no_grad():
                    if artifact:
                        output = self.model.generate(**batch, max_new_tokens=request.max_tokens,
                            do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                    else:
                        with self.model.disable_adapter():
                            output = self.model.generate(**batch, max_new_tokens=request.max_tokens,
                                do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                count = batch.input_ids.shape[1]
                generated = output[0, count:]
                return {'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                        'usage': {'prompt_tokens': count, 'completion_tokens': len(generated)},
                        'elapsed_ms': round((time.monotonic() - started) * 1000)}
            finally:
                if artifact:
                    self.model.set_adapter('default')
                    self.model.delete_adapter(name)
