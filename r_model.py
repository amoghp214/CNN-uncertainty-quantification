import torch

class RejectionLayerModel(torch.nn.Module):

    def __init__(self, dim):
        super(RejectionLayerModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(in_features=dim, out_features=10)
        self.tanh = torch.nn.Tanh()
        #self.layers = layers
    
    def forward(self, t):
        t = t
        t = self.flatten(t)
        t = self.fc(t)
        t = self.tanh(t)
        return t

        #for layer in self.layers:
        #    t = layer(t)
        #return t