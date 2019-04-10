import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
class Trainer:
    def __init__(self,network,config,train=True):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=config['network_params']['lr'])
        self.init_rate = config['network_params']['lr']
        if train:
            self.network.train()
        else:
            self.network.eval()

        self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        self.epoch=0

    def train(self,inputs,targets):
        self.optimizer.zero_grad()
        inputs = torch.from_numpy(inputs).cuda().float()
        targets = torch.from_numpy(targets).cuda().float()
        y = self.network(inputs)
        loss = self.loss(y,targets)
        loss.backward()
        self.optimizer.step()
        lr = self.init_rate* (0.9 ** (self.epoch // 100))
        lr = max(0.001,lr)
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.epoch+=1
        return loss.item()

    def predict(self,inputs):
        self.network.eval()
        inputs = torch.from_numpy(inputs).cuda().float()
        return self.network(inputs).cpu().detach().numpy()

    def save(self,outputdir):
        torch.save(self.network.state_dict(),outputdir)
    
    def copy(self,model):
        self.network.load_state_dict(model.state_dict())

    def load(self,weightPath):
        self.network.load_state_dict(torch.load(weightPath))

class ImageTrainer:
    def __init__(self,network,config,train=True):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=config['network_params']['lr'])
        self.init_rate = config['network_params']['lr']
        if train:
            self.network.train()
        else:
            self.network.eval()

        self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        self.epoch=0

    def train(self,inputs,targets):
        self.optimizer.zero_grad()
        inputs = torch.from_numpy(inputs).cuda().float()
        if len(inputs.size()) < 4:
            inputs = torch.unsqueeze(inputs,0)
        targets = torch.from_numpy(targets).cuda().float()
        y = self.network(inputs)
        loss = self.loss(y,targets)
        loss.backward()
        self.optimizer.step()
        lr = self.init_rate* (0.9 ** (self.epoch // 100))
        lr = max(0.001,lr)
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.epoch+=1
        return loss.item()

    def predict(self,inputs):
        self.network.eval()
        inputs = torch.from_numpy(inputs).cuda().float()
        if len(inputs.size()) < 4:
            inputs = torch.unsqueeze(inputs,0)
        return self.network(inputs).cpu().detach().numpy()

    def save(self,outputdir):
        torch.save(self.network.state_dict(),outputdir)
    
    def copy(self,model):
        self.network.load_state_dict(model.state_dict())

    def load(self,weightPath):
        self.network.load_state_dict(torch.load(weightPath))