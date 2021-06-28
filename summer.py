import enum
from torch.utils.data import dataloader


if __name__ == '__main__':
    import pickle
    import numpy as np
    import os
    import glob
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    import tools
    import cv2
    import time


    bg = cv2.imread('faster_bg.png')
    # print('background shape: {}'.format(bg.shape))
    # cv2.imshow('bg', bg)
    # cv2.waitKey(0)
    imgs = []
    for i in range(31):
        imgs.append(cv2.imread('./M/{}.png'.format(i)))
    imgs = torch.from_numpy(np.array(imgs))
    small_dictionary = pickle.load(open("./loader/penn_baxter_left_traj1025.hdf5.pkl", 'rb'))
    states = torch.from_numpy(small_dictionary['states'])
    dataset = TensorDataset(imgs, states)
    summer_loader = DataLoader(dataset, batch_size=1)
    
    nz, ngf, nc, epoch_num = 5, 64, 3, 100
    gpu_available = torch.cuda.is_available()

    netG = tools.Generator(nz=nz, ngf=ngf, nc=nc)
    netG.apply(tools.weights_init)
    criterion = tools.My_loss(img_shape=(3, 480, 640))
    optimizer = optim.Adam(netG.parameters(), lr=0.1, betas=(0.5, 0.999))
    softplus = torch.nn.Softplus(beta=1, threshold=0)
    # if gpu_available:
    #     netG.cuda()
    
    for idx, (img, state) in enumerate(summer_loader):
        state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
        loss_list = []
        img = np.array(torch.squeeze(img))
        for epoch in range(epoch_num):
            ep_loss = []
            optimizer.zero_grad()
            fg = torch.squeeze(netG(state)).permute(0,2,1)
            
            # print('image shape: {}'.format(img.shape))
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.imwrite('img.png', img)

            # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            # bg=cv2.cvtColor(bg,cv2.COLOR_BGR2HSV)           

            # cv2.imshow('img',img)
            # cv2.imshow('bg',bg)
            # cv2.waitKey()

            # img,bg=img[:,:,0],bg[:,:,0]
            
            obj_fg = (img - bg) # (obj_fg >= 0).all() outputs FALSE
            obj_fg = torch.tensor(obj_fg, dtype=torch.float)
            obj_fg = softplus(obj_fg)
            obj_fg = torch.round(obj_fg)
            obj_fg = np.array(obj_fg)

            print('objective foreground shape: {}'.format(obj_fg.shape))
            cv2.imshow('img-bg', obj_fg)
            cv2.waitKey(0)
            a = obj_fg
            cv2.imwrite('obj_fg.png', a)

            obj_fg = torch.from_numpy(obj_fg)
            obj_fg = obj_fg.permute(2,0,1)
            loss = criterion(fg, obj_fg)
            loss.backward()
            optimizer.step()
            loss = loss/(640*480*3)
            ep_loss.append(loss)
            print('epcoh: {}, loss: {}'.format(epoch, loss))
            loss_list.append(torch.sum(torch.tensor(ep_loss))/len(ep_loss))
        
        x = range(0, epoch_num)
        y = loss
        fig = plt.subplots()
        plt.plot(x, loss_list)
        plt.xlabel('epoches')
        plt.ylabel('train loss')
        timestamp = 'prediction[' + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()).replace(":", "-") + '].png'
        plt.savefig('C:\\Users\\hbrch\\Desktop\\Robonet\\plots\\'+timestamp)
        plt.show()

    pass