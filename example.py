import json
from document import Document

with open('./input.json') as f:
    data = json.load(f)

p = Document(data)
sentences = p.get_top_k_sentence(
    k=15,
    D=10
)

for idx, (weight, sentence) in sentences:
    print(f"Sentence(index:{idx}, weight:{weight:.4f}): {sentence}")
