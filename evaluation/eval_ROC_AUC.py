import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ==========================
# 文件列表
# ==========================
files = ['Our-DL5.csv',"Our-DL2.csv",'Our-DL4.csv','Our-DL3.csv','Our-DL.csv'] #DL

# ==========================
# 初始化存储变量
# ==========================
FPRs = []
TPRs = []
AUCs = []

# ==========================
# 模型标签
# ==========================
models = ['U-Net', 'BGA-Net', "LightM-UNet", 'MBG-Net', 'Our']

# ==========================
# 配置颜色、线条和点样式（更加明显的颜色）
# ==========================
colors = ['#1f77b4', '#ff4500', '#32cd32', '#8b0000', '#ff1493']  # 使用更鲜艳的颜色
line_styles = ['-', '--', '-.', ':', '-']  # 不同的线条样式
markers = ['o', 's', '^', 'D', 'p']       # 不同的点样式

# ==========================
# 设置更精美的背景
# ==========================
plt.style.use('seaborn-darkgrid')  # 使用一个更具有层次感的背景样式

# ==========================
# 遍历文件，计算 ROC & AUC
# ==========================
for i, file in enumerate(files):
    # 读取CSV文件
    df_cdr = pd.read_csv(file, usecols=['CDR'])
    df_glau = pd.read_csv(file, usecols=['Glaucoma'])

    # 转换为列表
    df_cdr = df_cdr.values.tolist()
    df_glau = df_glau.values.tolist()

    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(df_glau, df_cdr)
    roc_auc = auc(fpr, tpr)

    # 保存结果
    FPRs.append(fpr)
    TPRs.append(tpr)
    AUCs.append(roc_auc)

# ==========================
# 绘制 ROC 曲线
# ==========================
plt.figure(figsize=(10, 8))

# 为每个模型选择不同的颜色、线条样式和点样式
for fpr, tpr, roc_auc, model, color, line_style, marker in zip(FPRs, TPRs, AUCs, models, colors, line_styles, markers):
    plt.plot(
        fpr, tpr,
        lw=4,             # 线条细一些
        color=color,
        linestyle=line_style,
        marker=marker,
        markersize=6,     # marker稍小
        alpha=1.0,        # 不透明，更清晰
        label=f'{model} (AUC = {roc_auc:.4f})'
    )

# ==========================
# 设置图像的范围
# ==========================
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# ==========================
# 添加标签和标题
# ==========================
plt.xlabel('False Positive Rate', fontsize=28)
plt.ylabel('True Positive Rate', fontsize=28)
plt.title('Receiver Operating Characteristic Curve', fontsize=28)

# ==========================
# 增加更强的背景网格线
# ==========================
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)

# ==========================
# 添加图例并确保不会遮挡曲线
# 使图例更突出，添加框，设置框透明度和阴影
# ==========================
plt.legend(loc="lower right", fontsize=23, frameon=True, fancybox=True, shadow=True, facecolor='w', edgecolor='black')

# ==========================
# 增加透明度和渐变效果
# ==========================
plt.gca().set_facecolor('whitesmoke')  # 设置背景颜色
plt.tight_layout()

# ==========================
# 保存为高分辨率 PNG
# ==========================
plt.savefig("ROC_curve.png", dpi=400, bbox_inches='tight')  # 高分辨率 PNG

# ==========================
# 显示图像
# ==========================
plt.show()
