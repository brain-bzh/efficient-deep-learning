import matplotlib.pyplot as plt
import numpy as np


x = [6956298, 11171146, 11173962, 14728266]
y = [95.04, 95.11, 93.02, 92.64]
name = ["DenseNet121", "PreActResNet18", "ResNet18", "VGG16"]

# 一张图
fig = plt.figure()
plt.scatter(x, y)


# 限制x,y轴的范围，设置标
# plt.xticks(range(0, 16000, 2000))
plt.yticks(range(90, 96))

plt.xlabel("Number of model parameters")
plt.ylabel("Top 1 Accuracy(%)")

plt.title("Image Classification task on ImageNet dataset")

# 给散点加标签
for i in range(len(x)):
    plt.text(
        x[i],
        y[i],
        name[i],
        fontsize=10,
        color="r",
        style="italic",
        weight="light",
        verticalalignment="center",
        horizontalalignment="right",
        rotation=0,
    )

# plt.show()

fig.savefig("../TP1_report/accuracy.png")
