from pathlib import Path
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForPreTraining, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
#from transformers import AutoModel, AutoConfig



def main():
    # tokenizer
    #bert_tokenizer = BertTokenizerFast.from_pretrained('/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/bert-base-chinese',
    bert_tokenizer = BertTokenizerFast.from_pretrained('./pretrained/bert-base-chinese',
             additional_special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"],
             pad_token='<pad>' ,max_len=5)

    # prepare dataset
    dataset = load_dataset("text", data_files=
              {"train": "./data/wiki/political_text/political_text_sentences.txt", })
        
    print(dataset['train'][5]) # take a look 
    dataset = dataset.map(lambda examples: bert_tokenizer(examples["text"], truncation=True, padding="max_length"), batched=True)
    print(len(dataset))
    
    # prepare model
    #configuration = AutoConfig.from_pretrained('bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('./pretrained/bert-base-chinese')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer, mlm=False,
    )
    # how to train
    training_args = TrainingArguments(
        output_dir="./cache",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
    )
    trainer.train()
    #model.save_pretrained("/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/ZhiGuoLiZheng-GPT2")
    model.save_pretrained("./pretrained/ZhiGuoLiZheng-BERT")
    bert_tokenizer.save_pretrained("./pretrained/ZhiGuoLiZheng-BERT")
    #bert_tokenizer.save_pretrained("/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/ZhiGuoLiZheng-GPT2")


if __name__ == "__main__":
    main()
