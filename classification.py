import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, ipt_list):
        self.images = [i['image'] for i in ipt_list]
        self.images = np.concatenate(self.images)
        self.target = [i['label'] for i in ipt_list]
        L = []
        for z in ipt_list:
            plist = [z['label'] for _ in range(z['image'].shape[0])]
            L = L + plist
        self.target = L
        assert len(self.target) == len(self.images), 'wrong initiation'   


    def __getitem__(self, index):
        img = self.images[index]
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*9*13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = x.float()
        # original image: (batch, 48, 64, 3)
        x = x.permute(0, 3, 1, 2)
        # image: (batch, 3, 48, 64)
        x = self.pool(F.relu(self.conv1(x)))
        # image: (batch, 6, 44, 60)
        # image: (batch, 6, 22, 30)
        x = self.pool(F.relu(self.conv2(x)))
        # image: (batch, 16, 18, 26)
        # image: (batch, 16, 9, 13)
        x = x.reshape(-1, 16*9*13)
        # image: (batch, 1872)
        x = F.relu(self.fc1(x))
        # image: (batch, 120)
        x = F.relu(self.fc2(x))
        # image: (batch, 84)
        x = self.fc3(x)
        # image: (batch, 7)
        return x

if __name__ == "__main__":
    from loader import hdf5_loader
    from loader import get_metadata
    import matplotlib.pyplot as plt
    import os
    import glob
    import torch.optim as optim
    import adabound
    from adabelief_pytorch import AdaBelief


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
                del robot[t]['state']
                del robot[t]['file_name']
            favorite_robot.append(robot)

    # set a dictionary for changing labels
    robot_dict = {}
    for i in range(len(set(meta_data.frame.robot))):
        robot_dict[list(set(meta_data.frame.robot))[i]] = i

    # divide dataset into training and testing sets & change label from string to number(int)
    split_fraction = 0.75 # if there are 100 images, 75 images are used for training and the rest are used for testing 
    favorite_robot_train, favorite_robot_test = [], []
    for i in favorite_robot: # len(favorite_robot) = 7
        for dictionary in i:
            dictionary['label'] = robot_dict[dictionary['label']]
            dictionary['image'] = dictionary['image'].reshape(-1, *dictionary['image'].shape[-3:])
            cut_point = int(dictionary['image'].shape[0] * split_fraction)
            favorite_robot_train.append({'image': dictionary['image'][0:cut_point], 'label': dictionary['label']})
            favorite_robot_test.append({'image': dictionary['image'][cut_point:], 'label': dictionary['label']})

    # show divided sets and other Statistical data
    images_for_training, images_for_testing = 0, 0
    for LABEL in range(7):
        for name in robot_dict.keys():
                if robot_dict[name] == LABEL:            
                    print('___________________{}/{}___________________'.format(LABEL, name)) 
        img_count_1, img_count_2, img_count_3 = 0, 0, 0
        for i in favorite_robot_train:
            if i['label'] == LABEL:
                img_count_1 += i['image'].shape[0]
        for i in favorite_robot_test:
            if i['label'] == LABEL:
                img_count_2 += i['image'].shape[0]
        for i in favorite_robot:
            for dic in i:
                if dic['label'] == LABEL:
                    img_count_3 += dic['image'].shape[0]
        print('train set: {}, test set: {}, total: {}'.format(img_count_1, img_count_2, img_count_3))
        images_for_training += img_count_1
        images_for_testing += img_count_2
    print('images_for_training: {}, images_for_testing: {}'.format(images_for_training, images_for_testing))

    # trainloader and testloader
    train_set = dataset(favorite_robot_train)
    trainloader = DataLoader(train_set, batch_size=50, shuffle=True)
    test_set = dataset(favorite_robot_test)
    testloader = DataLoader(test_set, batch_size=300, shuffle=True)

    gpu_available = torch.cuda.is_available()
    net = Net()
    learning_rate = 0.0000001
    criterion = nn.CrossEntropyLoss()
    # optimizer_dict = {}
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-7)
    optimizer = adabound.AdaBound(net.parameters(), lr=learning_rate, eps=1e-7)
    # optimizer = AdaBelief(net.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)

    if gpu_available:
        net = net.cuda()
        criterion = criterion.cuda()

    # Training
    losslist = []
    for epoch in range(200):
        running_loss = 0
        if epoch % 5 == 4:
            learning_rate = learning_rate / 10.
            optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=1e-7)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) # if divedied by batch size??
            loss.backward()
            optimizer.step()
            if gpu_available:
                loss = loss.cpu()
            running_loss += loss.item()
        losslist.append((epoch+1, running_loss/884))
        with open('./result.txt', 'a') as f:
            print('[epoch: %d] loss: %.3f' % (epoch + 1, running_loss/884), file=f)

    # Save
    PATH = './simple_net.pth'
    torch.save(net.state_dict(), PATH)

    # Plot loss vs. epoch figure
    fig, ax = plt.subplots()
    x = np.array([i[0] for i in losslist])
    y = np.array([i[1] for i in losslist])
    ax.plot(x, y)
    ax.set(xlabel='epoch', ylabel='running loss')
    ax.grid()
    fig.savefig("test.png")
    plt.show()

    # Testing
    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    with open('./result.txt', 'a') as f:
        print('corrects: {}, total: {}, accuracy: {}'.format(correct, total, 100*correct/total), file=f)
    pass