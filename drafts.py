if __name__ == '__main__':
    from tools import save_weight_of_layers
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
    import generator as ge


    # fig = plt.figure(figsize=(7,12))
    # counter = 1
    # for i in [0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30]:
    #     image = plt.imread('./M2_chosen/n_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1

    # for i in [0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30]:
    #     image = plt.imread('./pred_img/r/r_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1
    # for i in range(12):
    #     image = plt.imread('./pred_img/r_pred/r_p_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1
    
    # for i in [0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30]:
    #     image = plt.imread('./pred_img/g/g_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1
    # for i in range(12):
    #     image = plt.imread('./pred_img/g_pred/g_p_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1

    # for i in [0, 1, 3, 4, 7, 8, 13, 14, 19, 21, 29, 30]:
    #     image = plt.imread('./pred_img/b/b_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1
    # for i in range(12):
    #     image = plt.imread('./pred_img/b_pred/b_p_{}.png'.format(i))
    #     plt.subplot(7,12,counter)
    #     plt.imshow(image)
    #     plt.axis('off')
    #     counter += 1         
    # fig.show()

    # netG = ge.Generator_120_160(nz=5, ngf=8, nc=3)
    # netG.load_state_dict(torch.load('C:\\Users\\hbrch\\Desktop\\Robonet\\netG_120_160.pth'))
    # netG.eval()

    a = torch.tensor([[[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[1,1,1],[3,3,3]],[[3,3,3],[2,2,2],[1,1,1]]])
    print(a.shape)
    z = a.shape[0]*a.shape[1]*a.shape[2]
    print(z)
    b = a.reshape(-1,a.shape[0]*a.shape[1]*a.shape[2])
    print(b.shape)
    b = b.reshape(-1, a.shape[0], a.shape[1])
    print(b)
    pass