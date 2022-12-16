import torch
from transformers import pipeline, set_seed
from transformers import BertForMaskedLM
from transformers import BertTokenizer

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-chinese',
    pad_token='[PAD]' ,max_len=512)


model = BertForMaskedLM.from_pretrained("./pretrained/ZhiGuoLiZheng-BERT")#%%

#%%
generator = pipeline('fill-mask', model=model, tokenizer=tokenizer_bert)
set_seed(42)

print(generator("要 紧 抓[MASK]产"))
print(generator("要 落 实[MASK]政建设"))

