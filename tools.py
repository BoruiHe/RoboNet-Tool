import torch
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def MaxAbsAxis(array,axis=None):
    return np.fabs(array,axis)

class ImgPredDataset(Dataset):
    def __init__(self, robot_dict, robot_name, cam=0):
        assert isinstance(robot_dict, dict), 'robot_dict should be of type dict'
        assert robot_name in robot_dict.keys(), 'robot_name should be one of {}, {}, {}, {}, {}, {}, {}'.format(*set(robot_dict.keys()))
        assert cam == 0, 'please only use the first camera'
        self.dictionary = robot_dict[robot_name]
        self.next_imgs, self.next_stas = [], []
        self.curr_imgs, self.curr_stas = [], []
        for d in self.dictionary:
            # only carmera #0, i.e. the first camera
            d['image'] = d['image'][:,0,:,:,:]
            d['image'] = d['image'].astype('float32')
            d['image'] = ((d['image']- 0) * 2 / 255) - 1
            # assume the value of attribute 'state' is degree of rotation
            # Q: why values of the fifth joint are often larger? In some cases, it could be 3 orders of magnitude larger than that of first 4 joints.
            # Normalize values of each column by dividing them by maximum absolute value 
            for i in range(d['state'].shape[1]):
                d['state'][:,i] = d['state'][:,i]/np.max(MaxAbsAxis(d['state']),0)[i]
            
            self.next_imgs.append(d['image'][1:,:,:,:])
            self.next_stas.append(d['state'][1:,:])
            self.curr_imgs.append(d['image'][:-1,:,:,:])
            self.curr_stas.append(d['state'][:-1,:])
            # extract images and states from only one file
            break    

        self.next_imgs = np.concatenate(self.next_imgs, axis=0)
        self.next_stas = np.concatenate(self.next_stas, axis=0)
        self.curr_imgs = np.concatenate(self.curr_imgs, axis=0)
        self.curr_stas = np.concatenate(self.curr_stas, axis=0)

    def __len__(self):
        return self.curr_imgs.shape[0]

    def __getitem__(self, idx):
        return self.curr_imgs[idx], self.curr_stas[idx], self.next_imgs[idx], self.next_stas[idx]

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def draw_scatter(horizontal_axis, ave, variance, std_deviation, path):
    plt.scatter(horizontal_axis, np.array(ave), label='average', color=(0.8,0.,0.))
    plt.scatter(horizontal_axis, np.array(std_deviation), label='std_deviation', color=(0.,0.5,0.))
    plt.xlabel('size of dataloader')
    plt.ylabel('training loss')
    plt.legend(loc='best')
    plt.savefig(path+'_ave.png')
    plt.close()
    plt.scatter(horizontal_axis, np.array(variance), label='variance', color=(0.8,0.,0.))
    plt.xlabel('size of dataloader')
    plt.ylabel('variance of training loss')
    plt.legend(loc='best')
    plt.savefig(path+'_var.png')
    plt.close()

def save_weight_of_layers(network, name):
    weight_of_layer = {}
    for i in range(len(network.main)):
        if isinstance(network.main[i],torch.nn.modules.conv.ConvTranspose2d):
            weight_of_layer[network.main[i]._get_name()+'_{}'.format(i)] = torch.sum(torch.square(network.main[i].weight)).item() + torch.sum(torch.square(network.main[i].bias)).item()
        elif isinstance(network.main[i],torch.nn.modules.batchnorm.BatchNorm2d):
            weight_of_layer[network.main[i]._get_name()+'_{}'.format(i)] = torch.sum(torch.square(network.main[i].weight)).item() + torch.sum(torch.square(network.main[i].bias)).item()
        else: # LeakyRelu
            weight_of_layer[network.main[i]._get_name()+'_{}'.format(i)] = 0.01*0.01
    with open(name+'.pkl', 'wb') as fp:
        pickle.dump(weight_of_layer, fp)

class My_loss(nn.Module):
    def __init__(self, img_shape):
        super(My_loss, self).__init__()
        self.img_shape = img_shape

    def forward(self, fake, real):
        assert fake.shape == self.img_shape, 'fakes shape may be wrong'
        assert real.shape == fake.shape, 'fake images and real images got different shape'
        square_sum = torch.sum((real-fake)**2)
        loss = square_sum/torch.prod(torch.tensor(fake.shape))
        loss_sqrt = torch.sqrt(square_sum)/torch.prod(torch.tensor(fake.shape))
        return loss, loss_sqrt