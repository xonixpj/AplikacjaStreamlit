import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Aplikacja do tłumaczenia tekstu i rozpoznawania wydźwięku emocjonalnego")


import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_model_and_tokenizer():
    model_name = "Helsinki-NLP/opus-mt-en-de"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(model, tokenizer, text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

st.write('## Instrukcja')
st.write('1. Wybierz opcję tłumaczenia lub analizy wydźwięku emocjonalnego.')
st.write('2. Wprowadź tekst w odpowiednim polu tekstowym.')
st.write('3. Kliknij przycisk, aby uzyskać wynik.')

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumacz Ang -> Niem",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        classifier = pipeline("sentiment-analysis")
        answer = classifier(text)
        st.write(answer)

if option == "Tłumacz Ang -> Niem":
    st.write("Ta aplikacja tłumaczy tekst z języka angielskiego na język niemiecki.")
    text_input = st.text_area("Wprowadź tekst po angielsku:", "")
    if st.button("Tłumacz"):
        if text_input:
            with st.spinner('Tłumaczenie...'):
                model, tokenizer = load_model_and_tokenizer()
                translated_text = translate_text(model, tokenizer, text_input)
                st.success("Tłumaczenie zakończone!")
                st.write("**Przetłumaczony tekst:**")
                st.write(translated_text)
        else:
            st.error("Wprowadź tekst do przetłumaczenia.")

st.write("Autor: s22582 Patryk Matyjasiak")


