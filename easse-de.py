import torch
import pandas
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
#nltk.download('punkt_tab')
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from easse.bertscore import corpus_bertscore
from easse.textstat_metrics import corpus_fre
from rouge_score import rouge_scorer


device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available
df_test_de = pandas.read_csv("path_to_test_data", sep=",")

############ Evaluation ENG -> DE ############
model_name_or_path = "path_to_last_checkpoint_eng-model"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

input = []   
preds = []
refs_sents = []
for _, row in df_test_de.iterrows():
    tokenizer.src_lang = "de_DE"
    inp = tokenizer(row["column_of_original_sents"], return_tensors="pt").to(device)
    gen = model.generate(**inp, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    input.append(row["column_of_original_sents"])
    preds.append(pred)
    refs_sents.append(row["column_of_simple_sents"])

refs = [refs_sents]

print("Evaluation of ENG->DE test set:")

fre_score = corpus_fre(preds)
print("FRE:", fre_score)
bleu_score = corpus_bleu(preds,refs)
print("BLEU:", bleu_score)
bert_score = corpus_bertscore(preds,refs)
print("BERTscore:", bert_score)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = []
for pred, ref in zip(preds, refs):
    score = scorer.score(ref[0], pred)['rougeL'].fmeasure
    rouge_l_scores.append(score)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("ROUGE-L:", avg_rouge_l)
sari_score = corpus_sari(input, preds, refs)
print("SARI:", sari_score)

############ Evaluation ENG+DE -> DE ############
model_name_or_path = "path_to_last_checkpoint_engde-model"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

preds = []
refs_sents = []
for _, row in df_test_de.iterrows():
    tokenizer.src_lang = "de_DE"
    inp = tokenizer(row["original"], return_tensors="pt").to(device)
    gen = model.generate(**inp, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    preds.append(pred)
    refs_sents.append(row["column_of_simple_sents"])
refs = [refs_sents]

print("Evaluation of ENG+DE->DE test set:")

fre_score = corpus_fre(preds)
print("FRE:", fre_score)
bleu_score = corpus_bleu(preds,refs)
print("BLEU:", bleu_score)
bert_score = corpus_bertscore(preds,refs)
print("BERTscore:", bert_score)
for pred, ref in zip(preds, refs):
    score = scorer.score(ref[0], pred)['rougeL'].fmeasure
    rouge_l_scores.append(score)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("ROUGE-L:", avg_rouge_l)
sari_score = corpus_sari(input, preds, refs)
print("SARI:", sari_score)

############ Evaluation DE+ENG -> DE ############
model_name_or_path = "path_to_last_checkpoint_deeng-model"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

preds = []
refs_sents = []
for _, row in df_test_de.iterrows():
    tokenizer.src_lang = "de_DE"
    inp = tokenizer(row["original"], return_tensors="pt").to(device)
    gen = model.generate(**inp, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    preds.append(pred)
    refs_sents.append(row["column_of_simple_sents"])
refs = [refs_sents]

print("Evaluation of DE+ENG->DE test set:")

fre_score = corpus_fre(preds)
print("FRE:", fre_score)
bleu_score = corpus_bleu(preds,refs)
print("BLEU:", bleu_score)
bert_score = corpus_bertscore(preds,refs)
print("BERTscore:", bert_score)
for pred, ref in zip(preds, refs):
    score = scorer.score(ref[0], pred)['rougeL'].fmeasure
    rouge_l_scores.append(score)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("ROUGE-L:", avg_rouge_l)
sari_score = corpus_sari(input, preds, refs)
print("SARI:", sari_score)

############ Evaluation EdNeG -> DE ############

model_name_or_path = "path_to_last_checkpoint_edneg-model"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

preds = []
refs_sents = []
for _, row in df_test_de.iterrows():
    tokenizer.src_lang = "de_DE"
    inp = tokenizer(row["column_of_original_sents"], return_tensors="pt").to(device)
    gen = model.generate(**inp, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    preds.append(pred)
    refs_sents.append(row["column_of_simple_sents"])
refs = [refs_sents]

print("Evaluation of EdNeG->DE test set:")

fre_score = corpus_fre(preds)
print("FRE:", fre_score)
bleu_score = corpus_bleu(preds,refs)
print("BLEU:", bleu_score)
bert_score = corpus_bertscore(preds,refs)
print("BERTscore:", bert_score)
for pred, ref in zip(preds, refs):
    score = scorer.score(ref[0], pred)['rougeL'].fmeasure
    rouge_l_scores.append(score)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("ROUGE-L:", avg_rouge_l)
sari_score = corpus_sari(input, preds, refs)
print("SARI:", sari_score)

############ Evaluation DE -> DE ############
model_name_or_path = "path_to_last_checkpoint_de-model"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name_or_path)
model = MBartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

preds = []
refs_sents = []
for _, row in df_test_de.iterrows():
    tokenizer.src_lang = "de_DE"
    inp = tokenizer(row["column_of_original_sents"], return_tensors="pt").to(device)
    gen = model.generate(**inp, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    pred = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    preds.append(pred)
    refs_sents.append(row["column_of_simple_sents"])
refs = [refs_sents]

print("Evaluation of DE->DE test set:")

fre_score = corpus_fre(preds)
print("FRE:", fre_score)
bleu_score = corpus_bleu(preds,refs)
print("BLEU:", bleu_score)
bert_score = corpus_bertscore(preds,refs)
print("BERTscore:", bert_score)
for pred, ref in zip(preds, refs):
    score = scorer.score(ref[0], pred)['rougeL'].fmeasure
    rouge_l_scores.append(score)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("ROUGE-L:", avg_rouge_l)
sari_score = corpus_sari(input, preds, refs)
print("SARI:", sari_score)
