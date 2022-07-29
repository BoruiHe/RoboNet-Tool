if __name__ == '__main__':
    import pickle
    import numpy as np
    import os
    import glob
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import tools
    import generator as ge
    import cv2
    import time


    # add learning rate scheduler
    # Add simple configuration in the future, which includes: episode choice, num of repetition, num of downsampling, criterion img_shape, epoch list choice, model choice......
    bg = cv2.imread('faster_2_bg.png')
    bg = cv2.pyrDown(bg)
    bg = torch.tensor(bg, dtype=torch.float)
    small_dictionary = pickle.load(open("./loader/berkeley_sawyer_traj19970.hdf5.pkl", 'rb'))

    # # episode 1
    # imgs_0 = []
    # for i in range(31):
    #     layer = cv2.imread('./M/n_{}.png'.format(i))
    #     layer = cv2.pyrDown(layer)
    #     layer = cv2.pyrDown(layer)
    #     layer = cv2.pyrDown(layer)
    #     imgs_0.append(layer)
    #     print(imgs_0[i].shape)
    # imgs_0 = torch.from_numpy(np.array(imgs_0)).float()
    # states_0 = torch.from_numpy(small_dictionary['states']).float()

    # episode 2
    imgs_1 = []
    for i in [0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30]:
        layer = cv2.imread('./M2_chosen/n_{}.png'.format(i))
        layer = cv2.pyrDown(layer)
        imgs_1.append(layer)
    imgs_1 = torch.from_numpy(np.array(imgs_1)).float()
    states_1 = torch.from_numpy(small_dictionary['states'][[0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30], :]).float()
    
    nz, ngf, nc = 5, 8, 3
    weight_decay = [0.1, 0.01]
    criterion = tools.My_loss(img_shape=(120, 160, 3))
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()).replace(":", "-")

    if os.path.exists('ls_epoch_M2.npy'):
        ls_epoch = np.load('ls_epoch_M2.npy')
    else:
        ls_epoch =[]
        for size in range(len(imgs_1)):
            dataset = TensorDataset(imgs_1[:size+1], states_1[:size+1])
            summer_loader = DataLoader(dataset, batch_size=1)
            netG = ge.Generator_120_160(nz=nz, ngf=ngf, nc=nc)
            netG.apply(tools.weights_init)
            optimizer = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
            prev_loss, curr_loss, epoch = 0, 0, 0
            while abs((curr_loss-prev_loss))>0.01 or prev_loss+curr_loss==0:
                ls_loss = []
                for _, (img, state) in enumerate(summer_loader):
                    state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
                    img = torch.squeeze(img)
                    optimizer.zero_grad()
                    pred_fg = torch.squeeze(netG(state))
                    pred_fg = pred_fg.permute(1,2,0)
                    pred_img = pred_fg*255 + bg
                    loss, _ = criterion(pred_img, img)
                    loss.backward()
                    optimizer.step()
                    ls_loss.append(loss)
                prev_loss = curr_loss
                curr_loss = sum(ls_loss)/len(ls_loss)
                epoch += 1
            ls_epoch.append(epoch)
        np.save('ls_epoch_M2.npy', np.array(ls_epoch))

    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    def copy_outputs(NN):
        for name in list(NN._modules.keys()):
            NN._modules[name].register_forward_hook(get_features(name))

    loss_of_each_size, l_variance, l_std_deviation = [], [], []
    l_sqrt_o_e_size, l_sqrt_variance, l_sqrt_std_deviation = [], [], []
    for size in range(len(imgs_1)):
        dataset = TensorDataset(imgs_1[:size+1], states_1[:size+1])
        summer_loader = DataLoader(dataset, batch_size=1)
        loss_list, l_sqrt_list = [], []
        for j in range(10): # repetitions
            netG = ge.Generator_toy(nz=nz, ngf=ngf, nc=nc)
            netG.apply(tools.weights_init)
            optimizer = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.999))
            for epoch in range(ls_epoch[size]):
                ep_loss, ep_loss_sqrt = [], []
                for _, (img, state) in enumerate(summer_loader):
                    state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
                    img = torch.squeeze(img)
                    optimizer.zero_grad()
                    pred_fg = netG(state)
                    pred_fg = torch.squeeze(pred_fg)
                    copy_outputs(netG) 
                    pred_fg = pred_fg.permute(1,2,0)
                    pred_img = pred_fg*255 + bg
                    loss, loss_sqrt = criterion(pred_img, img)
                    loss.backward()
                    optimizer.step()
                    ep_loss.append(loss)
                    ep_loss_sqrt.append(loss_sqrt)
                    print('epcoh: {}, loss: {}'.format(epoch, loss))
                if epoch == ls_epoch[size]-1:
                    loss_list.append(torch.sum(torch.tensor(ep_loss))/len(ep_loss))
                    l_sqrt_list.append(torch.sum(torch.tensor(ep_loss_sqrt))/len(ep_loss))
        # statistics -> loss
        l_ave = sum(loss_list)/len(loss_list)
        l_var = sum((i - l_ave) ** 2 for i in loss_list) / len(loss_list)
        loss_of_each_size.append(l_ave)
        l_variance.append(l_var)
        l_std_deviation.append(torch.sqrt(l_var))
        # statistics -> square root of loss
        l_sqrt_ave = sum(l_sqrt_list)/len(l_sqrt_list)
        l_sqrt_var = sum((i - l_sqrt_ave) ** 2 for i in l_sqrt_list) / len(l_sqrt_list)
        l_sqrt_o_e_size.append(l_sqrt_ave)
        l_sqrt_variance.append(l_sqrt_var)
        l_sqrt_std_deviation.append(torch.sqrt(l_sqrt_var))

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    torch.save(netG.state_dict(), 'C:\\Users\\hbrch\\Desktop\\Robonet\\netG_120_160.pth')
    x = range(1, len(imgs_1)+1)
    path = 'C:\\Users\\hbrch\\Desktop\\Robonet\\plots\\'
    suffix = ('sawyer_sq', 'sawyer_sqrt')
    tools.draw_scatter(x, loss_of_each_size, l_variance, l_std_deviation, path+suffix[0])
    tools.draw_scatter(x, l_sqrt_o_e_size, l_sqrt_variance, l_sqrt_std_deviation, path+suffix[1])

    # netG = ge.Generator_ss(nz=nz, ngf=ngf, nc=nc)
    # netG.load_state_dict(torch.load('C:\\Users\\hbrch\\Desktop\\Robonet\\netG.pth'))
    # netG.eval()

    for i, (v_img, state) in enumerate(summer_loader):
        state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
        v_fg = torch.squeeze(netG(state))
        v_fg = v_fg.permute(1,2,0)
        pred_img = v_fg*255 + bg
        # cv2.imshow("prediction", pred_img.to(dtype=torch.uint8).numpy())
        # cv2.waitKey()
        cv2.imwrite('./pred_img/berkeley_sawyer_cam2/pred_img_{}.png'.format(i), pred_img.to(dtype=torch.uint8).numpy())
    pass
