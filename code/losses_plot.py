import matplotlib.pyplot as plt
import pickle

# 假设你有多个损失数据文件，每个文件包含一个 epoch 的损失列表
file_names = ['losses-0.pkl', 'losses-L2.pkl', 'losses-temperature.pkl', 'losses-weight.pkl']  # 替换为你的文件名
labels = ['losses-0.pkl', 'losses-L2.pkl', 'losses-temperature.pkl', 'losses-weight.pkl']  # 对应的图例标签
# 创建一个包含四个子图的图形
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 遍历每个文件，加载数据并绘制曲线
for i, (file_name, label) in enumerate(zip(file_names, labels)):
    # 从文件中加载损失数据
    with open(file_name, 'rb') as f:
        losses = pickle.load(f)

    # 计算 epoch
    epochs = range(1, len(losses) + 1)

    # 绘制损失曲线到对应的子图
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    ax.plot(epochs, losses)
    ax.set_title(f'Training Loss Curve of {label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

# 调整子图之间的间距
plt.tight_layout()

# 保存图像到文件中
plt.savefig('subplots_loss_curves.png', dpi=300, facecolor='white', transparent=False)

print("包含子图的损失曲线图已保存为 'subplots_loss_curves.png'")

# 显示图像
plt.show()