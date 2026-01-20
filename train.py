import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from model import MiniGPT

texts = [
    "deep learning is powerful",
    "machine learning is useful",
    "deep learning is amazing",
    "transformers are powerful",
    "deep learning models work well"
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("VOCAB SIZE (TRAIN):", tokenizer.vocab_size)  # MUST PRINT 50257

encoded = [
    tokenizer.encode(t, add_special_tokens=False)
    for t in texts
]

model = MiniGPT(vocab_size=tokenizer.vocab_size)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    total_loss = 0
    for seq in encoded:
        x = torch.tensor(seq[:-1]).unsqueeze(0)
        y = torch.tensor(seq[1:]).unsqueeze(0)

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("✅ NEW model.pth saved with GPT-2 vocab")
