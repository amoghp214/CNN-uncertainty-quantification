import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import ImageClassificationModel
from model_wrapper import ModelWrapper
from NLBN import NLBN
from r_model import RejectionLayerModel
from uncertainty_quantification import calculate_dropout_ensemble_uncertainty as calc_deu

### THIS FILE IS USED FOR TESTING PURPOSES

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

nlbn.r_models[0] = RejectionLayerModel(nlbn.r_input_sizes[0])
nlbn.r_models[0].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_0_20231201_180844_10"))

nlbn.r_models[2] = RejectionLayerModel(nlbn.r_input_sizes[2])
nlbn.r_models[2].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_2_20231201_182146_10"))

nlbn.r_models[4] = RejectionLayerModel(nlbn.r_input_sizes[4])
nlbn.r_models[4].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_4_20231201_184245_10"))

nlbn.r_models[6] = RejectionLayerModel(nlbn.r_input_sizes[6])
nlbn.r_models[6].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_6_20231201_190126_10"))

nlbn.r_models[8] = RejectionLayerModel(nlbn.r_input_sizes[8])
nlbn.r_models[8].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_8_20231201_193046_10"))

nlbn.r_models[10] = RejectionLayerModel(nlbn.r_input_sizes[10])
nlbn.r_models[10].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_10_20231201_194500_10"))

nlbn.r_models[12] = RejectionLayerModel(nlbn.r_input_sizes[12])
nlbn.r_models[12].load_state_dict(torch.load("./models/EuroSAT_CrossEntropy_Adam_4_Blocks/r_model_12_20231201_200106_10"))

variance = nlbn.calculate_uncertainty()
for k, v in variance.items():
    print(k)
    print(v)


exit()

nlbn.create_rejection_dataset()
nlbn.train_NLBN()
nlbn.calculate_uncertainty()


