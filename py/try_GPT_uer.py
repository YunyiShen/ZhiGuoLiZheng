import torch
from transformers import pipeline, set_seed
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./pretrained/ZhiGuoLiZheng-GPT2-uer',
    pad_token='<pad>' ,max_len=256)

model = GPT2LMHeadModel.from_pretrained('./pretrained/ZhiGuoLiZheng-GPT2-uer')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)

print(generator("要 紧 抓", max_length=50, num_return_sequences=5, num_beams=10, top_p=0.999, repetition_penalty=1.5))
print(generator("要 落 实", max_length=50, num_return_sequences=5, num_beams=10, top_p=0.999, repetition_penalty=1.5))

