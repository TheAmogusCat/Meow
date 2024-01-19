import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from flask import Flask, request

model_dir = "pretrained"

tokenizer_path = f'./{model_dir}/tokenizer.json'
with open(tokenizer_path, 'r', encoding="utf-8") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

loaded_model = tf.keras.models.load_model(f"./{model_dir}/model")
max_sequence_len = int(open(f"./{model_dir}/max_sequence_len.txt", "r", encoding="utf-8").read())

def is_bad(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    prediction = loaded_model.predict(padded_sequence)

    scaled_prediction = prediction[0][0] * 100

    if scaled_prediction >= 50:
        return {"state": True, "prediction": prediction, "scaled_prediction": scaled_prediction}
    else:
        return {"state": False, "prediction": prediction, "scaled_prediction": scaled_prediction}

app = Flask(__name__)

@app.route('/meow', methods=['POST'])
def meow():
    input_text = request.form['text']
    result = is_bad(input_text)
    print(f'is_bad: {result["state"]}')
    print(f'{result["scaled_prediction"]}')
    return [result["state"], result["scaled_prediction"]]
