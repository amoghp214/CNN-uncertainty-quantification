import torch
import datetime
from r_model import RejectionLayerModel
from r_model_dataset import RejectionModelDataset
from torch.utils.data import DataLoader
from model import ImageClassificationModel



class NLBN():
    def __init__(self, model: ImageClassificationModel, train_set, test_set, loss_fn, optimizer, model_save_dir):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_set = train_set
        self.test_set = test_set
        self.model_save_dir = model_save_dir
        self.train_dataloader = DataLoader(self.train_set, batch_size=len(train_set), shuffle=True)
        self.test_dataloader = DataLoader(self.test_set, batch_size=len(test_set), shuffle=True)

        #self.rejection_layers = {}
        self.r_models = {}
        self.r_train_datasets = {}
        self.r_test_datasets = {}
        self.rejection_layer_uncertainties = {}
        self.r_confs = {}
        self.r_thresholds = {
            0: 0.5, 
            2: 0.5, 
            4: 0.5, 
            6: 0.5, 
            8: 0.5, 
            10: 0.5, 
            12: 0.5
        }
        self.r_input_sizes = {
            0: 16*61*61, 
            2: 32*58*58, 
            4: 64*25*25, 
            6: 128*22*22, 
            8: 12800, 
            10: 1024, 
            12: 512
        }
        

    def train_rejection_layer(self, r_layer_num, r_train_dataloader, r_test_dataloader, epochs=10):
        # train one rejection layer for the given layer
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        r_model = self.r_models[r_layer_num]
        for p in r_model.parameters():
            print(p.size())
        self.optimizer = torch.optim.Adam(params=r_model.parameters(), lr=0.001, weight_decay=0.0001)
        
        best_test_loss = 1_000_000

        for epoch in range(1, epochs+1):

            print('EPOCH {}:'.format(epoch))

            r_model.train(True)
            avg_train_loss = self.train_epoch(epoch, r_layer_num, r_train_dataloader)

            r_model.eval()

            curr_test_loss = 0
            with torch.no_grad():
                for i, test_data in enumerate(r_test_dataloader):
                    test_x, test_y = test_data
                    test_y_h = r_model(test_x)
                    test_loss = self.loss_fn(test_y_h, test_y)
                    curr_test_loss += test_loss
            
            avg_test_loss = curr_test_loss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_train_loss, avg_test_loss))

            if avg_test_loss < best_test_loss:
                best_vloss = avg_test_loss
                model_path = self.model_save_dir + 'r_model_{}_{}_{}'.format(r_layer_num, timestamp, epoch)
                torch.save(r_model.state_dict(), model_path)

    def train_epoch(self, epoch, r_layer_num, r_train_dataloader):
        curr_loss = 0
        last_loss = 0
        batches = 0
        r_model = self.r_models[r_layer_num]
        #torch.autograd.set_detect_anomaly(True)

        for i, data in enumerate(r_train_dataloader):
            batches += 1
            x, y = data
            self.optimizer.zero_grad()

            y_h = r_model(x)
            #print("found predictions")
            loss = self.loss_fn(y_h, y)
            #print("found loss")
            loss.backward(retain_graph=True)
            #loss.backward()
            self.optimizer.step()
            
            curr_loss += loss.item()
            #print(curr_loss)

            if i % 1000 == 999:
                last_loss = curr_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch * len(self.training_loader) + i + 1
                curr_loss = 0

        return curr_loss / batches

    def train_NLBN(self):
        # train & create all rejection layers
        for i, r_model in self.r_models.items():
            print("Training Rejection Layer", i)
            r_train_dataloader = DataLoader(self.r_train_datasets[i], batch_size=64, shuffle=True)
            r_test_dataloader = DataLoader(self.r_test_datasets[i], batch_size=64, shuffle=True)
            self.train_rejection_layer(i, r_train_dataloader, r_test_dataloader, 10)


    def construct_NLBN(self):
        # create the rejection layers (untrained, just structure)
        for i, param in enumerate(self.model.parameters()):

            if (len(param.size()) > 1):
                #self.rejection_layers["r_layer_" + i] = []
                layers = []
                #fc_inp_size = 1

                if (len(param.size()) > 2):
                    
                    layers.append(torch.nn.Flatten())
                    
                    #self.rejection_layers["r_layer_" + i].append(torch.nn.Flatten())
                    #fc_inp_size = param.size()[0] * param.size()[1] * param.size()[2]

                #fc_inp_size = fc_inp_size * param.size()[-1]
                #layers.append(torch.nn.Linear(in_features=fc_inp_size, out_features=10))
                layers.append(torch.nn.Linear(in_features=self.r_input_sizes[i], out_features=10))
                #layers.append(torch.nn.Tanh())
                #self.rejection_layers["r_layer_" + i].append(torch.nn.Linear(in_features=fc_inp_size, out_features=10))
                
                #self.r_models[i] = RejectionLayerModel(layers)
                self.r_models[i] = RejectionLayerModel(self.r_input_sizes[i])
                print(i)
    
    def create_rejection_dataset(self):
        # Create inputs set (x) for the different rejection layers
        # pass through first layer, use this output as input for first rejection layer
        # then pass first outputs through second layer and use this output for second rejection layer inputs
        # repeat this process

        # for simplicity and in the interest of time, I have hardcoded the creation of the rejection dataset

        x_train = None
        y_train = None
        for data in self.train_dataloader:
            x_train, y_train = data
        
        print(type(x_train))
        print(type(y_train))

        x_train = self.model.forward_conv1a(x_train)
        #print(x_train)
        self.r_train_datasets[0] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_conv1b(x_train)
        self.r_train_datasets[2] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_max_pool1c(x_train)
        #self.r_train_datasets[4] = RejectionModelDataset(x_train, y_train)

        x_train = self.model.forward_conv2a(x_train)
        self.r_train_datasets[4] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_conv2b(x_train)
        self.r_train_datasets[6] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_max_pool2c(x_train)
        #self.r_train_datasets[8] = RejectionModelDataset(x_train, y_train)

        x_train = self.model.forward_flatten3a(x_train)
        self.r_train_datasets[8] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_fc4a(x_train)
        self.r_train_datasets[10] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        x_train = self.model.forward_fc4b(x_train)
        self.r_train_datasets[12] = RejectionModelDataset(x_train, y_train)
        print(x_train.size())

        ## For test data
        x_test = None
        y_test = None
        for data in self.test_dataloader:
            x_test, y_test = data


        x_test = self.model.forward_conv1a(x_test)
        self.r_test_datasets[0] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_conv1b(x_test)
        self.r_test_datasets[2] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_max_pool1c(x_test)
        #self.r_test_datasets[4] = RejectionModelDataset(x_test, y_test)

        x_test = self.model.forward_conv2a(x_test)
        self.r_test_datasets[4] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_conv2b(x_test)
        self.r_test_datasets[6] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_max_pool2c(x_test)
        #self.r_test_datasets[8] = RejectionModelDataset(x_train, y_test)

        x_test = self.model.forward_flatten3a(x_test)
        self.r_test_datasets[8] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_fc4a(x_test)
        self.r_test_datasets[10] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())

        x_test = self.model.forward_fc4b(x_test)
        self.r_test_datasets[12] = RejectionModelDataset(x_test, y_test)
        print(x_test.size())
    

    def forward(self, t):
        # forward propagatio on NLBN (with if statements)
        # for simplicity, this function has been hardcoded
        
        t = self.model.forward_conv1a(t)
        r = self.r_models[0](t)
        if (torch.max(r) > self.r_thresholds[0]):
            return r
        
        t = self.model.forward_conv1b(t)
        r = self.r_models[2](t)
        if (torch.max(r) > self.r_thresholds[2]):
            return r
        
        t = self.model.forward_max_pool1c(t)
        
        t = self.model.forward_conv2a(t)
        r = self.r_models[4](t)
        if (torch.max(r) > self.r_thresholds[4]):
            return r
        
        t = self.model.forward_conv2b(t)
        r = self.r_models[6](t)
        if (torch.max(r) > self.r_thresholds[6]):
            return r
        
        t = self.model.forward_max_pool2c(t)
        
        t = self.model.forward_flatten3a(t)
        r = self.r_models[8](t)
        if (torch.max(r) > self.r_thresholds[8]):
            return r
        
        t = self.model.forward_fc4a(t)
        r = self.r_models[10](t)
        if (torch.max(r) > self.r_thresholds[10]):
            return r
        
        t = self.model.forward_fc4b(t)
        r = self.r_models[12](t)
        if (torch.max(r) > self.r_thresholds[12]):
            return r
        
        t = self.model.out(t)
        t = torch.nn.functional.tanh(t)

        return t

    def forward_stochastic(self, t, r_conf_num=None):
        # forward propagatio on NLBN (with if statements)
        # for simplicity, this function has been hardcoded

        r_above_threshold = {
            0: False,
            2: False,
            4: False,
            6: False,
            8: False,
            10: False,
            12: False
        }

        r = {
            0: [],
            2: [],
            4: [],
            6: [],
            8: [],
            10: [],
            12: []
        }
        
        t = self.model.forward_conv1a(t)
        r[0] = self.r_models[0](t)
        if (torch.max(r) > self.r_thresholds[0]):
            r_above_threshold[0] = True
            #r_conf_num[0] += 1
            #for i in range(len(r)):
            #    self.r_confs[0][i] = r[i]
        
        t = self.model.forward_conv1b(t)
        r[2] = self.r_models[2](t)
        if (torch.max(r) > self.r_thresholds[2]):
            r_above_threshold[2] = True
            #r_conf_num[2] += 1
            #for i in range(len(r)):
            #    self.r_confs[2][i] = r[i]
        
        t = self.model.forward_max_pool1c(t)
        
        t = self.model.forward_conv2a(t)
        r[4] = self.r_models[4](t)
        if (torch.max(r) > self.r_thresholds[4]):
            r_above_threshold[4] = True
            #r_conf_num[4] += 1
            #for i in range(len(r)):
            #    self.r_confs[4][i] = r[i]
        
        t = self.model.forward_conv2b(t)
        r[6] = self.r_models[6](t)
        if (torch.max(r) > self.r_thresholds[6]):
            r_above_threshold[6] = True
            #r_conf_num[6] += 1
            #for i in range(len(r)):
            #    self.r_confs[6][i] = r[i]
        
        t = self.model.forward_max_pool2c(t)
        
        t = self.model.forward_flatten3a(t)
        r[8] = self.r_models[8](t)
        if (torch.max(r) > self.r_thresholds[8]):
            r_above_threshold[8] = True
            #r_conf_num[8] += 1
            #for i in range(len(r)):
            #    self.r_confs[8][i] = r[i]
        
        t = self.model.forward_fc4a(t)
        r[10] = self.r_models[10](t)
        if (torch.max(r) > self.r_thresholds[10]):
            r_above_threshold[10] = True
            #r_conf_num[10] += 1
            #for i in range(len(r)):
            #    self.r_confs[10][i] = r[i]
        
        t = self.model.forward_fc4b(t)
        r[12] = self.r_models[12](t)
        if (torch.max(r) > self.r_thresholds[12]):
            r_above_threshold[12] = True
            #r_conf_num[12] += 1
            #for i in range(len(r)):
            #    self.r_confs[12][i] = r[i]
        
        t = self.model.out(t)
        t = torch.nn.functional.tanh(t)


        if (r_above_threshold[0]):
            for i in range(len(t)):
                self.r_confs[0][i].append([r[0][i], t[i]])

        if (r_above_threshold[2]):
            for i in range(len(t)):
                self.r_confs[2][i].append([r[2][i], t[i]])

        if (r_above_threshold[4]):
            for i in range(len(t)):
                self.r_confs[4][i].append([r[4][i], t[i]])

        if (r_above_threshold[6]):
            for i in range(len(t)):
                self.r_confs[6][i].append([r[6][i], t[i]])

        if (r_above_threshold[8]):
            for i in range(len(t)):
                self.r_confs[8][i].append([r[8][i], t[i]])

        if (r_above_threshold[10]):
            for i in range(len(t)):
                self.r_confs[10][i].append([r[10][i], t[i]])

        if (r_above_threshold[12]):
            for i in range(len(t)):
                self.r_confs[12][i].append([r[12][i], t[i]])

        return t
    
    def calculate_uncertainty(self):
        # calculate the uncertainty for the rejection layers (variance with respect to y_h)
        r_conf_num = {
            -1: 0,
            0: 0,
            2: 0,
            4: 0,
            6: 0,
            8: 0,
            10: 0,
            12: 0
        }

        self.r_confs = {
            -1: [[] for _ in range(10)],
            0: [[] for _ in range(10)],
            2: [[] for _ in range(10)],
            4: [[] for _ in range(10)],
            6: [[] for _ in range(10)],
            8: [[] for _ in range(10)],
            10: [[] for _ in range(10)],
            12: [[] for _ in range(10)],
        }

        r_test_dataloader = DataLoader(self.test_set, batch_size=1, shuffle=True)
        
        for data in r_test_dataloader:
            t, _ = data
            self.forward_stochastic(t, r_conf_num)
        
        variance = [[[0] for _ in range(len(self.r_confs))] for _ in range(len(self.r_confs[0]))]

        for k, _ in self.r_confs.items():
            for i in range(len(self.r_confs[k])):
                variance[k][i] = self.calculate_moving_variance(self.r_confs[k][i])

        


    def set_r_thresholds(self, thresholds):
        self.r_thresholds = thresholds

    def calculate_moving_variance(self, confs):
        variance = 0
        for i in range(len(confs)):
            variance += (abs(confs[i][0] - confs[i][1])) ** 2
            
        variance /= len(confs)
        return variance

