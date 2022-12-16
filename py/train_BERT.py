from pathlib import Path
from datasets import load_dataset
from transformers import BertTokenizer, BertForPreTraining, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
#from transformers import AutoModel, AutoConfig

def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples,chunk_size = 512):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    # tokenizer
    #bert_tokenizer = BertTokenizerFast.from_pretrained('/state/partition1/user/yyshen/ZhiGuoLiZheng/pretrained/bert-base-chinese',
    bert_tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese',
             pad_token='[PAD]' ,max_len=512)

    # prepare dataset
    dataset = load_dataset("text", data_files=
              {"train": "./data/wiki/political_text/political_text_sentences.txt", })
        
    print(dataset['train'][110]) # take a look 
    
    dataset = dataset.map(lambda example: tokenize_function(example, bert_tokenizer), 
    	batched=True, 
    	remove_columns=["text"])
    
    dataset = dataset.map(group_texts, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer, mlm_probability=0.15,
    )
    
    samples = [dataset["train"][i] for i in range(2)]
    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {bert_tokenizer.decode(chunk)}'")
    
    model = BertForMaskedLM.from_pretrained('./pretrained/bert-base-chinese')
    # how to train
    training_args = TrainingArguments(
        output_dir="./cache",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
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

