import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

random.seed(0); torch.manual_seed(0)

tfm = transforms.Compose([transforms.ToTensor()])
ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)

# campione
idxs = random.sample(range(len(ds)), 2000)
labels = [ds[i][1] for i in idxs]
cnt = Counter(labels)
print("Distribuzione classi (sample 2000):", cnt)

os.makedirs("outputs", exist_ok=True)

# griglia immagini
fig, axs = plt.subplots(5, 5, figsize=(6,6))
for ax in axs.ravel():
    i = random.choice(idxs)
    img, lab = ds[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(str(lab)); ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/eda_classes.png")
plt.close()

# 1. Bar chart percentuale
plt.figure(figsize=(6,4))
plt.bar(cnt.keys(), [v for v in cnt.values()], tick_label=[str(k) for k in cnt.keys()])
plt.title("Class distribution (sample)")
plt.xlabel("Classe")
plt.ylabel("Conteggio")
plt.savefig("outputs/eda_class_dist.png")
plt.close()

# 2. Histogram of pixel intensities
all_pixels = torch.cat([ds[i][0].view(-1) for i in idxs]).numpy()
plt.figure(figsize=(6,4))
plt.hist(all_pixels, bins=50)
plt.title("Pixel intensity distribution")
plt.xlabel("Valore pixel")
plt.ylabel("Frequenza")
plt.savefig("outputs/eda_pixel_hist.png")
plt.close()

# 3. PCA 2D scatter
X = torch.stack([ds[i][0].view(-1) for i in idxs]).numpy()
y = np.array(labels)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

pca = PCA(n_components=2)
X2 = pca.fit_transform(X_norm)

plt.figure(figsize=(6,6))
scatter = plt.scatter(X2[:,0], X2[:,1], c=y, cmap="tab10", alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("PCA 2D of sample")
plt.savefig("outputs/eda_pca2d.png")
plt.close()


print("EDA grafici salvati in outputs/")
