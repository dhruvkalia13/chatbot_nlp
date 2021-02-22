import json
import random
import pickle
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import nltk as nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, wrds, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(wrds)
    print("sentence words is ")
    print(sentence_words)
    for s in sentence_words:
        for i, w in enumerate(wrds):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def predict_class(sentence):
    # filter out predictions below a threshold
    print("Sentence is " + sentence)
    p = bow(sentence, words, show_details=True)

    model_dir = Path("model/chatbot_model.h5")
    modl = load_model(model_dir)
    res = modl.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    print(user_text)
    model = load_model('model/chatbot_model.h5')
    ints = predict_class(user_text)
    res = get_response(ints, intents)
    return res


if __name__ == "__main__":
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    app.run()
