import streamlit as st
import pickle

from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords # type: ignore
import string
from nltk.stem import PorterStemmer # type: ignore

def text_transform(data):
    # Converting text to lower case
    text = data.lower()
    text = TreebankWordTokenizer().tokenize(text)

    # Removing special characters

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Removing stopwords and punctuations from text

    for i in text:
        if i not in stopwords.words("english") and string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming: Stemming is the process of reducing a word to its root or base form. This is done by removing suffixes and prefixes to standardize words with similar meanings. It treats different word forms (e.g., "running," "runs," "ran") as the same base word ("run").

    ps = PorterStemmer()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tkidf= pickle.load(open("vectorizer.pkl", "rb"))

print(type(tkidf))

model = pickle.load(open("model.pkl", "rb"))

st.title("Email/Spam Classifier")

input_sms= st.text_area(" Enter the Message")

if st.button("Predict"):

    # Preprocess

    transformed_sms = text_transform(input_sms)

    # Vectorize

    vectorized = tkidf.transform([transformed_sms])

    # Predict

    output = model.predict(vectorized)
    print(output)


    # Print on site

    if output == 1:
        st.header("It is a spam mail")
    elif output==0:
        st.header("It is not a spam mail")




