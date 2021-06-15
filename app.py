import streamlit as st
import json

from document import Document

st.write("## Here you can find top **k** sentences, while maintaining a maximum distance between consecutive sentences")

fh = st.file_uploader("Input JSON File", ["json"])
if fh is not None:
    data = json.load(fh)
else:
    with open('./input.json') as fh:
        data = json.load(fh)

st_k, st_max_sentence_distance = st.beta_columns([1, 1])

k = st_k.number_input(
    "Enter value of K",
    min_value=1,
    max_value=100,
    value=15,
    step=1
)

max_sentence_distance = st_max_sentence_distance.number_input(
    "Enter maximum distance between sentences",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

p = Document(data)
sentences = p.get_top_k_sentence(
    k=k,
    D=max_sentence_distance
)

for idx, (weight, sentence) in sentences:
    st.info(f"""
    (index: ${idx}$, weight: ${weight:.4f}$)    
    **{sentence}**
    """)
