import random
import torch
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets, transforms

random.seed(0); torch.manual_seed(0)

tfm = transforms.Compose([transforms.ToTensor()])
ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)

# campione
idxs = random.sample(range(len(ds)), 2000)
labels = [ds[i][1] for i in idxs]
cnt = Counter(labels)
print("Distribuzione classi (sample 2000):", cnt)

# griglia immagini
fig, axs = plt.subplots(5, 5, figsize=(6,6))
for ax in axs.ravel():
    i = random.choice(idxs)
    img, lab = ds[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(str(lab)); ax.axis("off")
plt.tight_layout()
plt.show()
