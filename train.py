import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import ImageClassificationModel
from model_wrapper import ModelWrapper



transform =transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)]
)

dataset = datasets.EuroSAT(
    root = "./datasets/eurosat/",
    transform=transform,
    download=True
)


train_set, test_set = random_split(
    dataset = dataset,
    lengths = [20000, 7000]
)

print(train_set)
print(test_set)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

model = ImageClassificationModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
model_save_dir = "./models/EuroSAT_CrossEntropy_Adam_4_Blocks/"

model_wrapper = ModelWrapper(
    train_data=train_set,
    test_data=test_set,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    model_save_dir=model_save_dir)

model_wrapper.train(25)