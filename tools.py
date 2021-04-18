import torch
import numpy as np
import torch.nn as nn
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
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # add more ConTranspose2d layers.
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 3 x 4 (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 6 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 12 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 24 x 32
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
            # state size. (nc) x 48 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class My_loss(nn.Module):
    def __init__(self, img_shape):
        super(My_loss, self).__init__()
        self.img_shape = img_shape

    def forward(self, fake, real):
        assert tuple(fake.shape[1:]) == self.img_shape, 'images shape may be wrong'
        assert real.permute(0,3,1,2).shape == fake.shape, 'fake images and real images got different shape'
        real = real.permute(0,3,1,2)
        loss = torch.sum((real-fake)**2)
        return loss