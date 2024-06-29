import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.preprocessing.text import Tokenizer 
import re
import pickle
import string
import tensorflow as tf
import numpy as np

# st.cache(clear=True)
# model = tf.keras.models.load_model('next_word_model.h5',compile = False)
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model = tf.keras.models.load_model('next_word_model_cpu.h5')


st.set_page_config(page_title='Next Word Predictor')
st.title("Next Word Prediction (LSTM)- Ayushman Tomar" )
text = st.text_input("Write something to Predict")
# Function to cleantext
def clean(text): 
#     with open('next_word_prediction.txt', 'r', encoding='utf-8') as f:
#         text = f.read()
    # Remove extra spaces
    text = re.sub(' +', ' ', text)
    # Remove extra blank lines
    text = re.sub('\\n+', '\\n', text)
    punctuation_except_full_stop = ''.join([ch for ch in string.punctuation if ch != '.'])
    text = text.translate(str.maketrans('', '', punctuation_except_full_stop))
    text = re.sub('”', '', text)
    text = re.sub('“', '', text)
    text = re.sub('’', '', text)
    text = re.sub('‘', '', text)
    text = re.sub('\\n', ' ', text)
    text = re.sub('\.\s', '.', text)
    text = text.lower()
    return text

# text = clean() 
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([text])




# with open('tokenizer_new.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




# loading tokenizer pickel file
with open('tokenizer_new.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



# text = "how are"
# Clean
if text:
    for i in range(2):
        text = clean(text)
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=100, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))
        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                st.header(text)  
                # print(text)