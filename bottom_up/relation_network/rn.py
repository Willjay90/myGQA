import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import yaml

class ResNet152FeatureModule(nn.Module):
    def __init__(self):
        super(ResNet152FeatureModule, self).__init__()
        model = models.resnet152(pretrained=True)
        modules = list(model.children())[:-3] # bottleneck layer
        print(modules)
        self.feature_module = nn.Sequential(*modules)
    def forward(self, x):
        return self.feature_module(x)

    
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3129)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class RelationNetwork(nn.Module):
    def __init__(self, args):
        super(RelationNetwork, self).__init__()
        # self.conv = ResNet152FeatureModule()

        ## (number of filters per object + coordinate of object)*2 + question vector
        self.g_fc1 = nn.Linear((2048+2)*2+14, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)
        
        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args['batch_size'], 2)
        self.coord_oj = torch.FloatTensor(args['batch_size'], 2)

        if args['cuda']:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)
        
        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args['batch_size'], 196, 2)
        
        if args['cuda']:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args['batch_size'], 196, 2))
        for i in range(196):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        
        self.fcout = FCOutputModel()
        

        
    def forward(self, img, qst):
        # qst: 64 x 14
        # x = self.conv(img) # batch_size * 2048 * 14 * 14

        # batch_size, n_channels, d, _ = x.size()

        # x_flat = (64 x  196 x 2048)
        # x_flat = x.view(batch_size, n_channels, d*d).permute(0, 2, 1)
        qst = qst.type(torch.FloatTensor)
        qst = qst.cuda()

        x_flat = img[0] # reading image features
        batch_size, d, n_channels  = x_flat.size()
        # add cordinates
        x_flat = torch.cat([x_flat, self.coord_tensor], 2)

        # add question everywhere
        qst = torch.unsqueeze(qst, 1) # 64 x 1 x 14
        qst = qst.repeat(1, d, 1) # 64 x 196 x 14
        qst = torch.unsqueeze(qst, 2) # 64 x 196 x 1 x 14

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1) # 64 x 1 x 196 x 2050  
        x_i = x_i.repeat(1, d, 1, 1) # 64 x 196 x 196 x 2050 
        x_j = torch.unsqueeze(x_flat, 2) # 64 x 196 x 1 x 2050
        x_j = torch.cat([x_j, qst], 3) # 64 x 196 x 1 x (2050+14)
        x_j = x_j.repeat(1, 1, d, 1) # 64 x 196 x 196 x (2050+14)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # 64 x 196 x 196 x 4114

        # reshape for passing through network
        x_ = x_full.view(batch_size*d*d, 4114)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(batch_size, d*d, 256)
        x_g = x_g.sum(1).squeeze()

        # fully connected
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)