import torch

class ImageClassificationModel(torch.nn.Module):

    def __init__(self):

        super(ImageClassificationModel, self).__init__()

        self.dropout = torch.nn.Dropout(0.1)

        self.conv1a = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4)
        self.conv1b = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.max_pool1c = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv2a = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv2b = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4)
        self.max_pool2c = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.flatten3a = torch.nn.Flatten()
        
        self.fc4a = torch.nn.Linear(in_features=12800, out_features=1024)
        self.fc4b = torch.nn.Linear(in_features=1024, out_features=512)
        self.out = torch.nn.Linear(in_features=512, out_features=10)
    
    
    
    def forward(self, t):

        # Block 1
        t = self.forward_conv1a(t)
        t = self.forward_conv1b(t)
        t = self.forward_max_pool1c(t)

        # Block 2
        t = self.forward_conv2a(t)
        t = self.forward_conv2b(t)
        t = self.forward_max_pool2c(t)

        # Block 3
        t = self.forward_flatten3a(t)

        # Block 4
        t = self.forward_fc4a(t)
        t = self.forward_fc4b(t)

        # Output
        t = self.out(t)
        t = torch.nn.functional.tanh(t)

        return t



    def forward_conv1a(self, t):
        t = self.conv1a(t)
        t = self.dropout(t)
        t = torch.nn.functional.relu(t)
        return t
    
    def forward_conv1b(self, t):
        t = self.conv1b(t)
        t = self.dropout(t)
        t = torch.nn.functional.relu(t)
        return t
    
    def forward_max_pool1c(self, t):
        t = self.max_pool1c(t)
        return t
    

    def forward_conv2a(self, t):
        t = self.conv2a(t)
        t = self.dropout(t)
        t = torch.nn.functional.relu(t)
        return t
    
    def forward_conv2b(self, t):
        t = self.conv2b(t)
        t = self.dropout(t)
        t = torch.nn.functional.relu(t)
        return t
    
    def forward_max_pool2c(self, t):
        t = self.max_pool2c(t)
        return t
    

    def forward_flatten3a(self, t):
        t = self.flatten3a(t)
        return t
    
    def forward_fc4a(self, t):
        t = self.fc4a(t)
        t = self.dropout(t)
        t = torch.nn.functional.tanh(t)
        return t
    
    def forward_fc4b(self, t):
        t = self.fc4b(t)
        t = self.dropout(t)
        t = torch.nn.functional.relu(t)
        return t
    
    def forward_old(self, t):
        
        ## INPUT
        t = t

        ### BLOCK 1
        # Conv 1a
        t = self.conv1a(t)
        t = torch.nn.functional.relu(t)
        # Conv 1b
        t = self.conv1b(t)
        t = torch.nn.functional.relu(t)
        # Max Pooling 1c
        t = self.max_pool1c(t)
        
        ### BLOCK 2
        # Conv 2a
        t = self.conv2a(t)
        t = torch.nn.functional.relu(t)
        # Conv 2b
        t = self.conv2b(t)
        t = torch.nn.functional.relu(t)
        # Max Pooling 2c
        t = self.max_pool2c(t)

        ### BLOCK 3
        # Flatten 3a
        t = self.flatten3a(t)

        ### BLOCK 4
        # Fully Connected 4a
        t = self.fc4a(t)
        t = torch.nn.functional.tanh(t)
        # Fully Connected 4b
        t = self.fc4b(t)
        t = torch.nn.functional.relu(t)

        ### OUT
        t = self.out(t)

        return t
    


    
    


'''
    def __init__(self):

        super(ImageClassificationModel, self).__init__()

        self.conv1a = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4)
        self.conv1b = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.max_pool1c = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv2a = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv2b = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4)
        self.max_pool2c = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv3a = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        self.conv3b = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4)
        self.max_pool3c = torch.nn.MaxPool2d(kernel_size=4, stride=2)

        self.flatten4a = torch.nn.Flatten()
        
        self.fc5a = torch.nn.Linear(in_features=512, out_features=4096)
        self.fc5b = torch.nn.Linear(in_features=4096, out_features=1024)
        self.out = torch.nn.Linear(in_features=1024, out_features=10)
    
    
    def forward(self, t):
        
        ## INPUT
        t = t

        ### BLOCK 1
        # Conv 1a
        t = self.conv1a(t)
        t = torch.nn.functional.relu(t)
        # Conv 1b
        t = self.conv1b(t)
        t = torch.nn.functional.relu(t)
        # Max Pooling 1c
        t = self.max_pool1c(t)
        
        ### BLOCK 2
        # Conv 2a
        t = self.conv2a(t)
        t = torch.nn.functional.relu(t)
        # Conv 2b
        t = self.conv2b(t)
        t = torch.nn.functional.relu(t)
        # Max Pooling 2c
        t = self.max_pool2c(t)

        ### BLOCK 3
        # Conv 3a
        t = self.conv3a(t)
        t = torch.nn.functional.relu(t)
        # Conv 3b
        t = self.conv3b(t)
        t = torch.nn.functional.relu(t)
        # Max Pooling 3c
        t = self.max_pool3c(t)

        ### BLOCK 4
        # Flatten 4a
        t = self.flatten4a(t)

        ### BLOCK 5
        # Fully Connected 5a
        t = self.fc5a(t)
        t = torch.nn.functional.tanh(t)
        # Fully Connected 5b
        t = self.fc5b(t)
        t = torch.nn.functional.relu(t)

        ### OUT
        t = self.out(t)

        return t

'''