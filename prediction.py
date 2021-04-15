if __name__ == "__main__":
    import pickle
    from loader import hdf5_loader
    from loader import get_metadata
    import matplotlib.pyplot as plt
    import os
    import glob
    import torch
    import torch.optim as optim
    import adabound
    from adabelief_pytorch import AdaBelief
    from torch.utils.data import Dataset, DataLoader
    import time
    import tools

    # extract images/states/actions from *.hdf5 files and save them(Robonet_Database)
    ''' Robonet_Database contains 700 dictionaries. Each dictionary stores image/state/action extracted from one .hdf5 file. '''
    Robonet_Database = []
    hdf5_directory = 'C://Users//hbrch//Desktop//Robonet//hdf5' # you should substitute my path with your path.
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

    batch_size = 3
    nz, ngf, nc, epoch_num = 5, 64, 3, 30
    gpu_available = torch.cuda.is_available()
    baxter = tools.ImgPredDataset(robot_dict, 'baxter')
    baxter_loader = DataLoader(baxter, batch_size=batch_size, shuffle=True)

    if not os.path.exists('./baxter_left_traj1025'):
        for bn in range(30):
            real_p = 255 * (baxter.curr_imgs[bn] + 1)/2
            np.moveaxis(real_p, 0, -1)
            real_p = np.around(real_p, decimals=-1).astype(int)
            imgplot = plt.imshow(real_p)
            plt.savefig('baxter_left_traj1025/{}.png'.format(bn))
        real_p = 255 * (baxter.next_imgs[29] + 1)/2
        np.moveaxis(real_p, 0, -1)
        real_p = np.around(real_p, decimals=-1).astype(int)
        imgplot = plt.imshow(real_p)
        plt.savefig('baxter_left_traj1025/30.png')

    netG = tools.Generator(nz=nz, ngf=ngf, nc=nc)
    netG.apply(tools.weights_init)
    criterion = tools.My_loss(img_shape=(3, 48, 64))
    # optimizer = AdaBelief(netG.parameters(), lr=0.1, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    # optimizer = optim.SGD(netG.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if gpu_available:
        netG.cuda()
        
    loss_list = []
    for epoch in range(epoch_num):
        ep_loss = []
        for bn, (c_imgs, c_stats, n_imgs, n_stats) in enumerate(baxter_loader):
            if gpu_available:
                n_imgs = n_imgs.cuda()
            optimizer.zero_grad()
            latent_V = torch.unsqueeze(torch.unsqueeze(n_stats,-1),-1).cuda()
            fake = netG(latent_V)
            loss = criterion(fake, n_imgs)
            loss.backward()
            optimizer.step()
            if gpu_available:
                loss = loss.cpu()
            loss = loss/(64*48*3*batch_size)
            ep_loss.append(loss)
            print('epcoh: {}, batch_num: {}, loss: {}'.format(epoch, bn, loss))
            # if bn % 10 == 9:
            #     for i in [0, 1, 2]:
            #         fig = plt.figure()
            #         # draw fake image
            #         ax = fig.add_subplot(1, 2, 1)
            #         fake_p = 255 * (fake.cpu().detach().numpy()[i] + 1)/2
            #         np.moveaxis(fake_p, 0, -1)
            #         fake_p = np.around(np.moveaxis(fake_p, 0, -1), decimals=-1).astype(int)
            #         imgplot = plt.imshow(fake_p)
            #         ax.set_title('fake')
            #         # draw real(next) image
            #         ax = fig.add_subplot(1, 2, 2)
            #         real_p = 255 * (n_imgs.cpu().detach().numpy()[i] + 1)/2
            #         np.moveaxis(real_p, 0, -1)
            #         real_p = np.around(real_p, decimals=-1).astype(int)
            #         imgplot = plt.imshow(real_p)
            #         ax.set_title('real')
            #         plt.show()

        loss_list.append(torch.sum(torch.tensor(ep_loss))/len(ep_loss))
        # print('epoch:',epoch,netG.main[12].weight)
    
    x = range(0, epoch_num)
    y = loss
    fig = plt.subplots()
    plt.plot(x, loss_list)
    plt.xlabel('epoches')
    plt.ylabel('train loss')
    timestamp = 'prediction[' + time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()).replace(":", "-") + '].png'
    plt.savefig(timestamp)
    plt.show()
    
    pass           