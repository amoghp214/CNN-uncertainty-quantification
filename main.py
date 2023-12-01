import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import ImageClassificationModel
from model_wrapper import ModelWrapper
from NLBN import NLBN
from uncertainty_quantification import calculate_dropout_ensemble_uncertainty as calc_deu



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

#model_wrapper.train(25)

trained_model = ImageClassificationModel()
trained_model.load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/model_20231201_142711_25"))

#dropout_variance = calc_deu(trained_model, test_set)
#print(dropout_variance)

r_train_set, r_test_set, _ = random_split(
    dataset = dataset,
    lengths = [4000, 1000, 22000]
)

optimizer = torch.optim.Adam(params=trained_model.parameters(), lr=0.001, weight_decay=0.0001)

nlbn = NLBN(
    model=trained_model,
    train_set=r_train_set,
    test_set=r_test_set,
    loss_fn=loss_fn,
    optimizer=optimizer,
    model_save_dir=model_save_dir
)

nlbn.construct_NLBN()
print(nlbn.r_models)
nlbn.create_rejection_dataset()
nlbn.train_NLBN()
nlbn.calculate_uncertainty()


