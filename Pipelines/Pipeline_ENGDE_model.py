# pip install -r requirements.txt

import pandas 
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

model_name = "D:\\sha\\mbart-simplification-eng-de\\checkpoint-273"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

df = pandas.read_csv("D:\\sha\\Daten_BA\\DEPlain\\E__Sentence-level_Corpus\\DEplain-APA-sent\\train_gekuerzt.csv", sep=",")
train_df, eval_df = train_test_split(df, test_size=0.33, random_state=42)
print(len(train_df), len(eval_df))

class SimplificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.src_lang ="de_DE"
        self.tgt_lang ="de_DE"
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data.iloc[idx]['original']
        tgt = self.data.iloc[idx]['simplification']
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        
        model_inputs = self.tokenizer(
            src, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )["input_ids"]
        model_inputs["labels"] = labels.squeeze()
        return {key: val.squeeze() for key, val in model_inputs.items()}
    
t_dataset = SimplificationDataset(train_df, tokenizer)
e_dataset = SimplificationDataset(eval_df, tokenizer)

training_args = TrainingArguments(
    output_dir="D:\\sha\\mbart-simplification-engde",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=3,
    logging_steps=200,
    eval_strategy="steps",
    eval_steps = 1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=t_dataset,
    eval_dataset=e_dataset,
    tokenizer=tokenizer
)

trainer.train()
