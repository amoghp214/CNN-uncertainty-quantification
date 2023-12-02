import datetime
import torch
from torch.utils.data import DataLoader


class ModelWrapper():
    def __init__(self, train_data, test_data, model, loss_fn, optimizer, model_save_dir):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.model_save_dir = model_save_dir
        self.setup()
    
    def setup(self):
        self.train_dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=64, shuffle=True)

    def train_epoch(self, epoch_idx):
        curr_loss = 0
        last_loss = 0

        batches = 0

        for i, data in enumerate(self.train_dataloader):
            batches += 1
            x, y = data
            self.optimizer.zero_grad()

            y_h = self.model(x)
            loss = self.loss_fn(y_h, y)
            loss.backward()
            self.optimizer.step()
            
            curr_loss += loss.item()

            if i % 1000 == 999:
                last_loss = curr_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_idx * len(self.training_loader) + i + 1
                curr_loss = 0

        return curr_loss / batches



    def train(self, epochs=5):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        best_test_loss = 1_000_000

        for epoch in range(1, epochs+1):

            print('EPOCH {}:'.format(epoch))

            self.model.train(True)
            avg_train_loss = self.train_epoch(epoch_idx=epoch)

            self.model.eval()

            curr_test_loss = 0
            with torch.no_grad():
                for i, test_data in enumerate(self.test_dataloader):
                    test_x, test_y = test_data
                    test_y_h = self.model(test_x)
                    test_loss = self.loss_fn(test_y_h, test_y)
                    curr_test_loss += test_loss
            
            avg_test_loss = curr_test_loss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_train_loss, avg_test_loss))

            if avg_test_loss < best_test_loss:
                best_vloss = avg_test_loss
                model_path = self.model_save_dir + 'model_{}_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)
            


