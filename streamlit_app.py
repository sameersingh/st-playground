import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import sys
import csv
import requests
import json

st.title('Test Streamlit')

@st.cache
def fill(model, text):
    unmasker = pipeline('fill-mask', model=model)
    return unmasker(text)

models = ['bert-base-uncased', 'bert-base-cased', 'roberta-base', 'distilbert-base-uncased']
model = st.selectbox('Select an MLM model', models)
text = st.text_input("Masked text", "i really like [MASK].")

if st.button('Fill'):
    st.table(fill(model, text))

dataset = load_dataset("sst")

left_col, right_col = st.beta_columns([2,1])

with left_col:
    label_counts = {True: {}, False: {}}
    all_insts = []
    search_tok = st.text_input("Enter left text", "nice")
    for typ,insts in dataset.items():
        label_counts[True][typ] = 0
        label_counts[False][typ] = 0
        for i in insts:
            if i['sentence'].find(search_tok) != -1:
                label = i['label']>0.5
                label_counts[label][typ] += 1
                all_insts.append({'review': i['sentence'], 'label':label})
    st.bar_chart(label_counts)
    # st.table(all_insts)
with right_col:
    label_counts = {True: {}, False: {}}
    all_insts = []
    search_tok = st.text_input("Enter right text", "nice")
    for typ,insts in dataset.items():
        label_counts[True][typ] = 0
        label_counts[False][typ] = 0
        for i in insts:
            if i['sentence'].find(search_tok) != -1:
                label = i['label']>0.5
                label_counts[label][typ] += 1
                all_insts.append({'review': i['sentence'], 'label':label})
    st.bar_chart(label_counts)
    # st.table(all_insts)    

def query(engine, text, num_completions):
    headers = {"Authorization": "Bearer {API TOKEN}", "Content-Type": "application/json"}
    for i in range(num_completions):
        response = requests.post("https://api.openai.com/v1/engines/"+ engine +"/completions",
            headers=headers,
            json={
                'prompt': text,
                'max_tokens': 6,
            }
        ).json()['choices']
        st.markdown(text + " **" + response[0]['text']  + "**")

text = st.text_input("Enter a prompt", "I really like")
engine = st.sidebar.selectbox('Select engine', ['ada', 'babbage', 'curie', 'davinci'])
num_completions = st.sidebar.slider("Number of completions", 1, 10, 1)
query(engine, text, num_completions)