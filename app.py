import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from model import MiniGPT
from transformers import AutoTokenizer

st.set_page_config(layout="wide")
st.title("Real-Time Transformer Text Autocomplete")

tokenizer = AutoTokenizer.from_pretrained("gpt2") #SimpleTokenizer()
#tokenizer.pad_token = tokenizer.eos_token

print("VOCAB_SIZE (APP)", tokenizer.vocab_size)
model = MiniGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

text = st.text_input("Enter text", "machine learning is")

if st.button("Predict Next Word"):
    x = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False
    )["input_ids"]
    st.write(x.shape)

    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)

    topk = torch.topk(probs, k=5)

    st.subheader("Next Token Predictions")
    for i, p in zip(topk.indices[0], topk.values[0]):
        word = tokenizer.decode([i.item()])
        st.write(f"'{word}' → {p.item():.3f}")
    fig, ax = plt.subplots()
    words = [tokenizer.decode([i.item()]) for i in topk.indices[0]]
    values = topk.values[0].cpu().numpy()
    ax.bar(words, values)
    ax.set_title("Next Token Probabilities")
    st.pyplot(fig)

    # Attention Heatmap
    attn = model.layers[0].attn.last_attention
    if attn is not None:
        st.subheader("Self-Attention Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(attn[0, 0], ax=ax)
        st.pyplot(fig)

