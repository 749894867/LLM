import torch
import torch.nn as nn
import tiktoken
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

CKPT_PATH = r"D:\360MoveData\Users\c\Desktop\读书人\亚大那几年\llm\llm\实验数据及权重\解冻+数据增强\gpt2_sentiment_thelastone.pth"


label_map = {0: "negative(负面)", 1: "neutral(中性)", 2: "positive(正面)"}

model_size = "124M"
settings_raw, params = download_and_load_gpt2(model_size, "models")

settings = {
    "vocab_size": 50257,
    "context_length": settings_raw["n_ctx"],
    "emb_dim": settings_raw["n_embd"],
    "n_layers": settings_raw["n_layer"],
    "n_heads": settings_raw["n_head"],
    "drop_rate": settings_raw.get("resid_pdrop", 0.1),
    "qkv_bias": True,
}

model = GPTModel(settings)
load_weights_into_gpt(model, params)

# 替换分类头
model.out_head = nn.Linear(settings["emb_dim"], 3).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model = model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
PAD_ID = 50256


# 文本编码
def encode(text, max_len=64):
    ids = tokenizer.encode(text)[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

# 测试函数
def predict(text):
    x = encode(text)
    logits = model(x)[:, -1, :]
    pred = torch.argmax(logits, dim=1).item()
    return label_map[pred]


while True:
    text = input("请输入一句话（输入 q 退出）：")
    if text.lower() == "q":
        break
    print("情感分类结果 →", predict(text), "\n")
