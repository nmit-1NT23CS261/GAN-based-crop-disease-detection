import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= SETTINGS =================
image_size = 128
batch_size = 64
nz = 100
num_epochs = 285          # UPDATED TO 285
lr = 0.00015
beta1 = 0.5

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ================= DATASET =================
dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

# Train only Late_blight
dataset.samples = [s for s in dataset.samples if "Late_blight" in s[0]]
dataset.targets = [0] * len(dataset.samples)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Total Late Blight Images:", len(dataset))

# ================= WEIGHT INIT =================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ================= GENERATOR =================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ================= DISCRIMINATOR =================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)

# ================= INITIALIZE =================
netG = Generator().to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

real_label = 0.9
fake_label = 0.

os.makedirs("outputs/gan_images", exist_ok=True)

print("Starting GAN training...")

# ================= TRAIN LOOP =================
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):

        real_images = data[0].to(device)
        b_size = real_images.size(0)

        netD.zero_grad()
        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_images)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = netG(noise)

        label.fill_(fake_label)
        output = netD(fake_images.detach())
        lossD_fake = criterion(output, label)
        lossD_fake.backward()

        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake_images)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  LossD: {(lossD_real+lossD_fake).item():.4f}  LossG: {lossG.item():.4f}")

    with torch.no_grad():
        fake = netG(torch.randn(16, nz, 1, 1, device=device)).detach().cpu()
        vutils.save_image(
            fake,
            f"outputs/gan_images/epoch_{epoch+1}.png",
            normalize=True
        )

torch.save(netG.state_dict(), "model/tomato_late_blight_gan.pt")

print("GAN training complete!")