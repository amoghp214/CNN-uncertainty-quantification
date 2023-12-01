import sys
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import ImageClassificationModel
from model_wrapper import ModelWrapper
from NLBN import NLBN
from uncertainty_quantification import calculate_rejection_confidence_variance as calc_rcv

if __name__ == "__main__":
    model_path = str(sys.argv[1])

    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)]
    )

    dataset = datasets.EuroSAT(
        root = "./datasets/eurosat/",
        transform=transform,
        download=True
    )

    r_train_set, r_test_set, _ = random_split(
        dataset = dataset,
        lengths = [1000, 1000, 25000]
    )

    trained_model = ImageClassificationModel()
    trained_model.load_state_dict(torch.load(model_path))
    loss_fn = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(params=trained_model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = None
    model_save_dir = "./models/EuroSAT_CrossEntropy_Adam_4_Blocks/"

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

    rcv_variance = calc_rcv(nlbn)
    print("Rejection Confidence Variance: ", rcv_variance)