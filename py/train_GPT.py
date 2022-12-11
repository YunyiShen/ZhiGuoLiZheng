from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


def main():
    # tokenizer
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese',
        additional_special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"],
        pad_token='<pad>' ,max_len=512)

    # prepare dataset
    dataset = load_dataset("text", data_files=
              {"train": "./data/wiki/political_text/political_text_sentences.txt", })
    print(dataset['train'][10]) # take a look 
    
    # prepare model
    configuration = GPT2Config(vocab_size=25000, n_layer=8)
    model = GPT2LMHeadModel(config=configuration)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer, mlm=False,
    )

    # how to train
    training_args = TrainingArguments(
        output_dir="./cache",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=64,
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
    model.save_pretrained("./pretrained/ZhiGuoLiZheng-GPT2")
    bert_tokenizer.save_pretrained("./pretrained/ZhiGuoLiZheng-GPT2")


if __name__ == "__main__":
    main()