import statistics
from torch.utils.data import DataLoader

def calculate_dropout_ensemble_uncertainty(model, test_set, num_fp = 10):
    y_h_conf = [[0 for i in range(10)] for j in range(test_set.__len__())]
    model.train(True)
    avg_conf_variance = [0 for i in range(10)]
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    for i, batch in enumerate(test_dataloader):
        x, _ = batch
        confs = [[] for j in range(0,10)]
        for t in range(0, num_fp):
            y_h = model.forward(x).squeeze().tolist()
            for a in range(len(confs)):
                confs[a].append(y_h[a])
        for a in range(len(confs)):
            conf_variance = statistics.variance(confs[a])
            avg_conf_variance[a] += conf_variance
    for i in range(len(avg_conf_variance)):
        avg_conf_variance[i] = avg_conf_variance[i] / test_set.__len__()
    model.eval()
    return avg_conf_variance
    

def calculate_rejection_confidence_variance(nlbn):
    rcv = nlbn.calculate_uncertainty()
    return rcv
    
    



