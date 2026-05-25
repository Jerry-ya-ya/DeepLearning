## fastai

fastai 是一個建立在 PyTorch 上的深度學習（Deep Learning）函式庫

它最大的特色是：
- 讓你用很少的程式碼
- 快速訓練 AI 模型

fastai 的設計理念

創辦人 Jeremy Howard 強調

讓更多人能使用 AI

AI 不是只屬於研究員或是在做 AI 專業領域的人

你可以把 fastai 想成

PyTorch 的高階簡化版

fastai 很受歡迎因為原本 PyTorch 要自己寫很多底層流程

例如：
- DataLoader
- Training Loop
- Optimizer
- Loss Function
- Validation
- GPU 管理

對初學者會很複雜

fastai 幫你包好了很多東西

例如你只需要：
- learn.fine_tune(5)

它就會自動：

- 訓練
- 驗證
- 使用 GPU
- 顯示 accuracy
- 幫你管理模型

fastai 最強的地方就是遷移式學習（Transfer Learning）

fastai 可以直接使用已經訓練好的大型模型

例如：
- ResNet（Microsoft Research）
- EfficientNet（Google Brain）
- ConvNeXtMeta（AI Research）

這些模型原本已經看過數百萬張圖片

## 模型介紹

### ResNet（Residual Network）

提出者 Microsoft Research

論文：https://arxiv.org/pdf/1512.03385

特色：Residual Block（殘差結構）

核心概念：讓神經網路可以變得非常深

以前：網路越深越難訓練

ResNet 利用 Skip Connection 解決了這件事

它最經典的是輸入可以直接跳過幾層

像這樣：

```bash
Input
 ├── Conv
 ├── Conv
 └── + 原始輸入
```

為什麼重要

因為後來很多模型：
- YOLO
- U-Net
- Transformer Vision
- ConvNeXt

其實都受到它影響

### EfficientNet

論文：https://arxiv.org/pdf/1905.11946

提出者：Google Brain

特色：效率超高

它想解決怎麼用更少資源達到更高準確率

一般模型放大方式

以前：
- 只加深
- 或只加寬
- 或只提高解析度

這樣會很混亂

EfficientNet 提出 Compound Scaling

一起平衡：
- 深度（Depth）
- 寬度（Width）
- 解析度（Resolution）

效果
- 同樣準確率
- 但模型更小更快

所以：
- 手機
- Jetson
- Edge AI

很常用

###　ConvNeXt

論文：https://arxiv.org/pdf/2201.03545

提出者：Meta AI Research

那時候 Vision Transformer 很紅

很多人覺得 CNN 已經輸了

他們的想法是如果把 CNN 用 Transformer 的設計思路重做呢？

結果他們發現 CNN 其實還是很強

特色：更現代化 CNN

例如：
- LayerNorm
- 大 Kernel
- 更好的訓練策略

它代表CNN 並沒有被 Transformer 淘汰

## 遷移式學習（Transfer Learning）

透過這些已經預訓練好的模型

你不用從零開始

只需要拿現成模型來重新學你的資料集

這就是遷移式學習（Transfer Learning）

fastai 最經典的圖片分類流程

Step 1：整理資料夾

```bash
dataset/
    happy/
    sad/
    angry/
```

Step 2：建立 DataLoader

```bash
dls = ImageDataLoaders.from_folder(path)
```

Step 3：建立模型

```bash
learn = vision_learner(dls, resnet34)
```

Step 4：開始訓練

```bash
learn.train(5)
```

就完成了一套訓練

