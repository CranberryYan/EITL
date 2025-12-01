import re
import pandas as pd
import matplotlib.pyplot as plt

# ====== 步骤 1：读取日志文件 ======
log_path = "/home/yst/文档/jwj/EITLNet-main/trainCBAM.log"  # 替换为你的日志路径

with open(log_path, 'r') as file:
    log_data = file.read()

# ====== 步骤 2：定义正则表达式 ======
train_pattern = re.compile(
    r'\[Train\] Epoch (\d+)/\d+ - Batch (\d+)/\d+ \| Loss: ([\d.]+) \| F-score: ([\d.]+) \| LR: ([\d.]+)'
)
val_pattern = re.compile(
    r'\[Val\]\s+Epoch (\d+)/\d+ - Batch (\d+)/\d+ \| Loss: ([\d.]+) \| F-score: ([\d.]+) \| LR: ([\d.]+)'
)

# ====== 步骤 3：提取数据 ======
def extract_matches(pattern, log_text):
    matches = pattern.findall(log_text)
    return pd.DataFrame([
        {
            'epoch': int(m[0]),
            'batch': int(m[1]),
            'loss': float(m[2]),
            'fscore': float(m[3]),
            'lr': float(m[4])
        }
        for m in matches
    ])

train_df = extract_matches(train_pattern, log_data)
val_df = extract_matches(val_pattern, log_data)

# ====== 步骤 4：按 epoch 统计平均值 ======
train_epoch_avg = train_df.groupby('epoch').agg({'loss': 'mean', 'fscore': 'mean', 'lr': 'mean'}).reset_index()
val_epoch_avg = val_df.groupby('epoch').agg({'loss': 'mean', 'fscore': 'mean', 'lr': 'mean'}).reset_index()

# ====== 步骤 5：保存数据为 CSV(可选) ======
train_df.to_csv('train_batches.csv', index=False)
val_df.to_csv('val_batches.csv', index=False)
train_epoch_avg.to_csv('train_epoch_avg.csv', index=False)
val_epoch_avg.to_csv('val_epoch_avg.csv', index=False)

# ====== 步骤 6：分别画 Loss 和 F-score 曲线 ======

# -- 绘制 Loss 曲线图 --
plt.figure(figsize=(8, 6))
plt.plot(train_epoch_avg['epoch'], train_epoch_avg['loss'], label='Train Loss', marker='o')
plt.plot(val_epoch_avg['epoch'], val_epoch_avg['loss'], label='Val Loss', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lossCBAM_curve.png")
plt.close()

# -- 绘制 F-score 曲线图 --
plt.figure(figsize=(8, 6))
plt.plot(train_epoch_avg['epoch'], train_epoch_avg['fscore'], label='Train F-score', marker='o')
plt.plot(val_epoch_avg['epoch'], val_epoch_avg['fscore'], label='Val F-score', marker='x')
plt.xlabel("Epoch")
plt.ylabel("F-score")
plt.title("F-score vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fscoreCBAM_curve.png")
plt.close()

print("✅ 图像已生成：loss_curveCB.png 和 fscore_curve.png")
