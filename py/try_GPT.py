import torch
from transformers import pipeline, set_seed
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizerFast

tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-chinese',
    additional_special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"],
    pad_token='<pad>' ,max_len=512)

configuration = GPT2Config(vocab_size=25000, n_layer=8)
model = GPT2LMHeadModel(config=configuration)
#%%
path2pytorch_model = "./pretrained/ZhiGuoLiZheng-GPT2/pytorch_model.bin"
model.load_state_dict(torch.load(path2pytorch_model))
#%%
generator = pipeline('text-generation', model=model, tokenizer=tokenizer_bert)
set_seed(42)

print(generator("要 紧 抓", max_length=50, num_return_sequences=5, num_beams=10, top_p=0.999, repetition_penalty=1.5))
print(generator("要 落 实", max_length=50, num_return_sequences=5, num_beams=10, top_p=0.999, repetition_penalty=1.5))

