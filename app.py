import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

stemmer = SnowballStemmer(language='english')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def text_tansformation(text) :
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]   ## Removed special characters
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english')) and  word not in set(string.punctuation)]

    return " ".join(words)


### APP design 
st.title("SMS Spam Detector")

with st.sidebar:
    st.header("How To Use It -")
    st.info("""
        - Enter or paste your SMS in the input box.
        - Click on the **Predict** button.
        - The app will process your SMS.
        - The trained ML model will classify the message.
        - Check the result:
        
        ✅ **No Spam** → The message is safe.
        
        🚨 **Spam** → The message is classified as spam.
    """)

input_sms = st.text_area(
    "Paste your SMS",
    height=150
)

if st.button("Predict") :

    ## Text Transformation
    transformed_sms = text_tansformation(input_sms)

    ## Vectorize
    vectorized_sms = tfidf.transform([transformed_sms])

    ## Feed to the model
    result = model.predict(vectorized_sms)

    ## Display
    if result == 0:
        st.success("✅ No Spam")
    else:
        st.warning("🚨 Spam")