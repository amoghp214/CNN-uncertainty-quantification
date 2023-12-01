import sys
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import ImageClassificationModel
from model_wrapper import ModelWrapper
from NLBN import NLBN
from uncertainty_quantification import calculate_dropout_ensemble_uncertainty as calc_deu

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

    _, test_set = random_split(
        dataset = dataset,
        lengths = [20000, 7000]
    )

    trained_model = ImageClassificationModel()
    trained_model.load_state_dict(torch.load(model_path))

    deu_variance = calc_deu(trained_model, test_set)
    print("Dropout Ensemble: ")
    print(deu_variance)