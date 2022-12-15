from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer, BertTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


def main():
    # tokenizer
    #bert_tokenizer = BertTokenizerFast.from_pretrained('/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/bert-base-chinese',
             
    # prepare dataset
    tokenizer = BertTokenizer.from_pretrained("./pretrained/gpt2-chinese-cluecorpussmall",max_len=256)
    model = GPT2LMHeadModel.from_pretrained("./pretrained/gpt2-chinese-cluecorpussmall")
    dataset = load_dataset("text", data_files=
              {"train": "./data/wiki/political_text/political_text_sentences.txt", })
        
    print(dataset['train'][5]) # take a look 
    dataset = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"), batched=True)
    print(len(dataset))
    
    # prepare model
    #configuration = GPT2Config(vocab_size=25000, n_layer=8)
    #model = GPT2LMHeadModel(config=configuration)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # how to train
    training_args = TrainingArguments(
        output_dir="./cache",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=10,
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
    model.save_pretrained("./pretrained/ZhiGuoLiZheng-GPT2-uer")
    bert_tokenizer.save_pretrained("./pretrained/ZhiGuoLiZheng-GPT2-uer")
    #bert_tokenizer.save_pretrained("/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/ZhiGuoLiZheng-GPT2")


if __name__ == "__main__":
    main()
