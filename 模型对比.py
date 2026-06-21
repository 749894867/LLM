import torch
import torch.nn as nn
from tqdm import tqdm
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt
import tiktoken
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random, numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']      # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 数据集
tokenizer = tiktoken.get_encoding("gpt2")
PAD_ID = 50256

class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64, pad_token_id=PAD_ID):
        df = pd.read_csv(csv_file)[["tweet", "sentiment"]].dropna()
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["label_id"] = df["sentiment"].map(label_map)
        self.texts = df["tweet"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token_id

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        ids = ids[:self.max_length]
        ids += [self.pad_token] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.texts)

dataset = TweetDataset(r"D:\LLM\tweet_sentiment_augmented_5000.csv", tokenizer)
test_len = len(dataset) // 10
_, test_set = torch.utils.data.random_split(dataset, [len(dataset) - test_len, test_len],
                                           generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#  GPT初始化
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

def build_model():
    model = GPTModel(settings)
    load_weights_into_gpt(model, params)
    model.out_head = nn.Linear(settings["emb_dim"], 3).to(device)
    return model.to(device)

#  池化版前向
def forward_sentence_logits(model, x, pad_id=PAD_ID):
    tok = model.tok_emb(x)
    pos = model.pos_emb.weight[:x.size(1)].unsqueeze(0)
    h = model.drop_emb(tok + pos)
    for block in model.trf_blocks:
        h = block(h)
    h = model.final_norm(h)
    mask = (x != pad_id).float().unsqueeze(-1)
    sent_emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
    return model.out_head(sent_emb)

#  测试评估
@torch.no_grad()
def evaluate(model, loader, pooling=False):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_sentence_logits(model, x) if pooling else model(x)[:, -1, :]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

# 模型
models = [
    ("分类头",                   r"D:\LLM\分类头\gpt2_sentiment.pth",              False),
    ("数据增强",                 r"D:\LLM\数据增强\gpt2_sentiment.pth",                False),
    ("池化",                     r"D:\LLM\池化\gpt2_sentiment_headonly.pth",                   True),
    ("解冻",                     r"D:\LLM\解冻\gpt2_sentiment_thelastone.pth",               False),
    ("池化+数据增强",                r"D:\LLM\池化+数据增强\gpt2_sentiment_headonly.pth",     True),
    ("池化+解冻",                r"D:\LLM\池化+解冻\gpt2_sentiment_headonly.pth",     True),
    ("解冻+数据增强",            r"D:\LLM\解冻+数据增强\gpt2_sentiment_thelastone.pth",  False),
    ("池化+数据增强+解冻",       r"D:\LLM\池化+数据增强+解冻\gpt2_sentiment_headonly.pth",       True),
]

results = []
for name, ckpt, pooling in models:
    model = build_model()
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    acc = evaluate(model, test_loader, pooling=pooling)
    results.append((name, acc))  # ← 必须记录结果
    print(f"{name: <20} → Test ACC = {acc*100:.2f}%")

# 可视化
import matplotlib.pyplot as plt

names = [x[0] for x in results]
accs = [x[1] * 100 for x in results]

plt.figure(figsize=(10,5))
bars = plt.bar(names, accs)

for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.1f}%", ha='center', va='bottom', fontsize=10)

plt.ylim(0, max(accs) + 10)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.title("Comparison of GPT-2 Fine-tuning Strategies", fontsize=14)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()

print("\n 已保存图像: model_accuracy_comparison.png")
