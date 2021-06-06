import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

trained_model_path = 'your_path/model/'    # Path to the stored model from training.py
trained_tokenizer = 'your_path/tokenizer/' # Path to the stored tokenizer from training.py

model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ",device)
model = model.to(device)


context ='''Your passage or sentence'''

answer = "Your answer"
text = "context: " + context + " " + "answer: " + answer + " </s>"

encoding = tokenizer.encode_plus(text,max_length =512, padding=True, return_tensors="pt")
#print (encoding.keys())
input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

model.eval()
outputs = model.generate(
    input_ids=input_ids,attention_mask=attention_mask,
    max_length=72,
    early_stopping=True,
    num_beams=5,
    num_return_sequences=1)   # Can decide on how many framed questions to return, can set as much as you wish

for i in outputs:
    sent = tokenizer.decode(i, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(sent) # Prints all possible sentences
