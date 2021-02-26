import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils


class ImgPredDataset(Dataset):
    def __init__(self, robot_dict, robot_name, cam=0):
        assert isinstance(robot_dict, dict), 'robot_dict should be of type dict'
        assert robot_name in robot_dict.keys(), 'robot_name should be one of {}, {}, {}, {}, {}, {}, {}'.format(*set(robot_dict.keys()))
        assert cam == 0, 'please only use the first camera'
        self.dictionary = robot_dict[robot_name]
        self.next_imgs, self.next_stas = [], []
        self.curr_imgs, self.curr_stas = [], []
        for d in self.dictionary:
            # only carmera NO.0, i.e. the first camera
            d['image'] = d['image'][:,0,:,:,:]
            d['image'] = d['image'].astype('float32')
            d['image'] = ((d['image']- 0) * 2 / 255) - 1
            
            self.next_imgs.append(d['image'][1:,:,:,:])
            self.next_stas.append(d['state'][1:,:])
            self.curr_imgs.append(d['image'][:-1,:,:,:])
            self.curr_stas.append(d['state'][:-1,:])        

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
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(3, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 3 x 4 (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 6 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 12 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 24 x 32
            nn.ConvTranspose2d( ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 48 x 64
        )

    def forward(self, input):
        return self.main(input)

class My_loss(nn.Module):
    def __init__(self, img_shape):
        super(My_loss, self).__init__()
        self.img_shape = img_shape
        return

    def forward(self, fake, real):
        assert tuple(fake.shape[1:]) == self.img_shape, 'images shape may be wrong'
        assert real.permute(0,3,1,2).shape == fake.shape, 'fake images and real images got different shape'
        real = real.permute(0,3,1,2)
        loss = 0
        # 2 lines -> 1 line code
        loss += torch.sum((real-fake)**2)
        return loss


if __name__ == "__main__":
    import pickle
    from loader import hdf5_loader
    from loader import get_metadata
    import matplotlib.pyplot as plt
    import os
    import glob
    import torch.optim as optim
    import adabound
    from adabelief_pytorch import AdaBelief
    import time


    # extract images/states/actions from *.hdf5 files and save them(Robonet_Database)
    ''' Robonet_Database contains 700 dictionaries. Each dictionary stores image/state/action extracted from one .hdf5 file. '''
    Robonet_Database = []
    hdf5_directory = 'C://Users//hbrch//OneDrive//Desktop//Robonet//hdf5' # you should substitute my path with your path.
    meta_data = get_metadata.load_metadata(hdf5_directory)
    
    for robot in set(meta_data.frame.robot):
        if not os.path.exists('./robots/{}.p'.format(robot)):
            Robonet_Database = []
            sub_meta_data = meta_data[meta_data['robot'] == robot]
            for h5file in list(sub_meta_data.index):
                data = {}
                hparams = hdf5_loader.default_loader_hparams()
                imgs, actions, states, labels = hdf5_loader.load_data(h5file, meta_data.get_file_metadata(h5file), hparams=hparams)
                data['file_name'] = h5file
                data['image'] = imgs
                data['action'] = actions
                data['state'] = states
                data['label'] = robot
                Robonet_Database.append(data)
            with open('./robots/{}.p'.format(robot), 'wb') as fp:
                pickle.dump(Robonet_Database, fp)
    
    # load dataset(Robonet_Database)
    favorite_robot = []
    for i in glob.glob('./robots/*.p'):
        with open(i, 'rb') as fp:
            robot = pickle.load(fp)
            for t in range(len(robot)):
                del robot[t]['action']
                del robot[t]['file_name']
            favorite_robot.append(robot)
    
    # build lists for each robot
    robot_dict = {}
    for i in favorite_robot:
        if not i[0]['label'] in robot_dict.keys():
            robot_dict[i[0]['label']] = []
        for dictionary in i:
            robot_dict[dictionary['label']].append(dictionary)
            del robot_dict[dictionary['label']][len(robot_dict[dictionary['label']])-1]['label']
    
    batch_size = 100
    nz, ngf, nc, training_range = 5, 64, 3, 100
    gpu_available = torch.cuda.is_available()
    baxter = ImgPredDataset(robot_dict, 'baxter')
    baxter_loader = DataLoader(baxter, batch_size=batch_size, shuffle=True) 

    netG = Generator()
    netG.apply(weights_init)
    criterion = My_loss(img_shape=(3, 48, 64))
    optimizer = AdaBelief(netG.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    if gpu_available:
        net = netG.cuda()
        criterion = criterion.cuda()
        
    loss_list = []
    for epoch in range(training_range):
        ep_loss = []
        for bn, (c_imgs, c_stats, n_imgs, n_stats) in enumerate(baxter_loader):
            if gpu_available:
                n_imgs = n_imgs.cuda()
            netG.zero_grad
            # the vector cannot be random!
            latent_V = torch.unsqueeze(torch.unsqueeze(n_stats,-1),-1).cuda()
            fake = netG(latent_V)
            loss = criterion(fake, n_imgs)
            loss.backward()
            optimizer.step
            if gpu_available:
                loss = loss.cpu()
            loss = loss/(64*48*100*3)
            ep_loss.append(loss)
            print('epcoh: {}, batch_num: {}, loss: {}'.format(epoch, bn, loss))
            if bn % 10 == 0:
                for i in [0, 49, 99]:
                    fig = plt.figure()
                    # draw fake image
                    ax = fig.add_subplot(1, 2, 1)
                    fake_p = 255 * (fake.cpu().detach().numpy()[i] + 1)/2
                    np.moveaxis(fake_p, 0, -1)
                    fake_p = np.around(np.moveaxis(fake_p, 0, -1), decimals=-1).astype(int)
                    imgplot = plt.imshow(fake_p)
                    ax.set_title('fake')
                    # draw real(next) image
                    ax = fig.add_subplot(1, 2, 2)
                    real_p = 255 * (n_imgs.cpu().detach().numpy()[i] + 1)/2
                    np.moveaxis(real_p, 0, -1)
                    real_p = np.around(real_p, decimals=-1).astype(int)
                    imgplot = plt.imshow(real_p)
                    ax.set_title('real')

                    plt.show()
        loss_list.append(torch.sum(torch.tensor(ep_loss))/len(ep_loss))
    
    x = range(0, training_range)
    y = loss
    fig = plt.subplots()
    plt.plot(x, loss_list)
    plt.xlabel('epoches')
    plt.ylabel('train loss')
    plt.show()
    timestamp = 'prediction[' + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()).replace(":", "-") + ']'
    plt.savefig(timestamp)

    pass