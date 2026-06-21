from importlib.metadata import version
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
from tqdm import tqdm
import random, numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


# 加载 GPT2 模型
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

model_size = "124M"
settings_raw, params = download_and_load_gpt2(model_size, "models")

settings = {
    "vocab_size": 50257,
    "context_length": settings_raw["n_ctx"],
    "emb_dim": settings_raw["n_embd"],
    "n_layers": settings_raw["n_layer"],
    "n_heads": settings_raw["n_head"],
    "drop_rate": settings_raw.get("resid_pdrop", 0.1),
    "qkv_bias": True
}

model = GPTModel(settings)
load_weights_into_gpt(model, params)
model = model.to(device)

# 数据集
tokenizer = tiktoken.get_encoding("gpt2")

class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64, pad_token_id=50256):
        df = pd.read_csv(csv_file)[["tweet", "sentiment"]].dropna()
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["sentiment"] = df["sentiment"].map(label_map)

        self.texts = df["tweet"].tolist()
        self.labels = df["sentiment"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token_id

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        ids = ids[:self.max_length]
        ids = ids + [self.pad_token] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


dataset = TweetDataset(r"D:\LLM\tweet_sentiment_augmented_5000.csv", tokenizer)

# 划分
total_len = len(dataset)
test_len = total_len // 10
val_len = total_len // 10
train_len = total_len - test_len - val_len

generator = torch.Generator().manual_seed(123)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_len, val_len, test_len], generator=generator
)

batch_size = 32
num_workers = 0
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 修改分类头
num_classes = 3
model.out_head = nn.Linear(settings["emb_dim"], num_classes).to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.out_head.parameters():
    p.requires_grad = True

# 评估
@torch.no_grad()
def evaluate_acc(data_loader, model, device, desc):
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(data_loader, desc=desc, ncols=90):
        x, y = x.to(device), y.to(device)
        logits = model(x)[:, -1, :]
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

@torch.no_grad()
def evaluate_loss(data_loader, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[:, -1, :]
        loss = criterion(logits, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)

# 早停
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, min_epochs=20):
        self.patience = patience
        self.delta = delta
        self.min_epochs = min_epochs
        self.best = None
        self.counter = 0
        self.best_state = None
        self.stop = False

    def __call__(self, epoch_idx, monitor_value, model):
        if self.best is None or monitor_value > self.best + self.delta:
            self.best = monitor_value
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        if epoch_idx + 1 >= self.min_epochs and self.counter >= self.patience:
            print(f"触发早停：已训练 {epoch_idx+1} 轮，无提升 {self.patience} 轮")
            self.stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# 训练
def train_classifier_simple(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    early_stopper = EarlyStopping(patience=3)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=80)

        for x, y in progress:
            x, y = x.to(device), y.to(device)
            logits = model(x)[:, -1, :]
            loss = nn.CrossEntropyLoss()(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_loss(val_loader, model, device)
        train_acc = evaluate_acc(train_loader, model, device, "Train Acc")
        val_acc = evaluate_acc(val_loader, model, device, "Val Acc")

        scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} → Train Loss {avg_train_loss:.4f}, Val Loss {val_loss:.4f}, Train Acc {train_acc*100:.2f}%, Val Acc {val_acc*100:.2f}%")

        early_stopper(epoch,val_acc, model)
        if early_stopper.stop:
            break

    early_stopper.restore_best(model)
    return train_losses, val_losses, train_accs, val_accs

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5,
    weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

train_losses, val_losses, train_accs, val_accs = train_classifier_simple(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=100
)

# 可视化
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

print("已保存图像: loss_curveh.png, accuracy_curve.png")

# 测试
test_acc = evaluate_acc(test_loader, model, device, desc="Test Acc")
print(f"最终测试集准确率: {test_acc*100:.2f}%")

# 保存模型
torch.save(model.state_dict(), "gpt2_sentiment.pth")
print("模型已保存：gpt2_sentiment.pth")