from flask import Flask, request, render_template
import os
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Set up MarianMT
MARIAN_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ROMANCE"
tokenizer_marian = MarianTokenizer.from_pretrained(MARIAN_MODEL_NAME)
model_marian = MarianMTModel.from_pretrained(MARIAN_MODEL_NAME)

# Set up GPT-2
GPT2_MODEL_NAME = "gpt2"
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
model_gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)


def translate_text(text, src='auto', dest='en'):
    inputs = tokenizer_marian(text, return_tensors="pt", padding=True, truncation=True)
    translated = model_marian.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer_marian.pad_token_id,
        eos_token_id=tokenizer_marian.eos_token_id,
    )
    return tokenizer_marian.decode(translated[0], skip_special_tokens=True)
def generate_response(input_text):
    input_ids = tokenizer_gpt2.encode(f"Customer: {input_text}\nChatbot:", return_tensors="pt")
    response = model_gpt2.generate(input_ids, num_return_sequences=1)
    response_text = tokenizer_gpt2.decode(response[0], skip_special_tokens=True)
    return response_text.split("Chatbot:")[-1].strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    language_code = request.form['language_code']
    while True:
        input_text = request.form['input_text']
        input_text_en = translate_text(input_text, src=language_code, dest='en')
        response_en = generate_response(input_text_en)
        response = translate_text(response_en, src='en', dest=language_code)
        return response


if __name__ == '__main__':
    app.run(debug=True)
