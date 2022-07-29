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


    bg = torch.tensor(cv2.imread('faster_bg.png'), dtype=torch.float)
    imgs = []
    for i in range(31):
        imgs.append(cv2.imread('./M/n_{}.png'.format(i)))
    imgs = torch.from_numpy(np.array(imgs)).float() # Move all casting outside training loop
    small_dictionary = pickle.load(open("./loader/penn_baxter_left_traj1025.hdf5.pkl", 'rb'))
    states = torch.from_numpy(small_dictionary['states']).float() # Move all casting outside training loop
    dataset = TensorDataset(imgs[:2], states[:2])
    summer_loader = DataLoader(dataset, batch_size=1)
    
    nz, ngf, nc, epoch_num = 5, 8, 3, 100
    gpu_available = torch.cuda.is_available()
    netG = tools.Generator_s(nz=nz, ngf=ngf, nc=nc)
    criterion = tools.My_loss(img_shape=(240, 320, 3))
    optimizer = optim.Adam(netG.parameters(), lr=0.001, betas=(0.5, 0.999)) # try a lower lr!
    softplus = torch.nn.Softplus(beta=1, threshold=0)

    loss_list = []
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()).replace(":", "-")
    for j in range(3):
        netG.apply(tools.weights_init) # add another outer loop for different initial weights!
        for epoch in range(epoch_num):
            ep_loss = []
            for idx, (img, state) in enumerate(summer_loader): # instead of all 31 fg, try 2 or 3 first.
                state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
                img = torch.squeeze(img)
                optimizer.zero_grad()
                pred_fg = torch.squeeze(netG(state))
                # obj_fg = (img - bg).to(dtype=torch.float) # zero the bg!
                # obj_fg = softplus(obj_fg) # why softplus first and rescaling second?
                # obj_fg = obj_fg/255 # rescale to interval [0,1]
                pred_fg = pred_fg.permute(1,2,0)
                pred_img = pred_fg*255 + bg

                # cv2.imshow('obj_fg', np.array((obj_fg*255).to(dtype=torch.uint8)))
                # cv2.waitKey(0)
                
                # How could the difference be exactly 255.0?
                ##################################################
                ## A brief report:

                ## Conclusion: line 47 causes the trouble.

                ## Explanation:
                ## An image was read as a numpy array of uint8. An unsigned 8-bit int number must be in range [0, 255].
                ## Relating knowledge are included in textbooks of COMPUTER ARCHITECTURE.
                ## Doing subtraction without casting int numbers to float numbers may produce a wrong value. 
                ## It's a fatal mistake for tasks realting to image processing because negative and positive values represent different color in dispalying an image.

                ## EXAMPLE: Check the output of img[239,317,1],bg[239,317,1],obj_fg[239,317,1].
                ##################################################

                # if epoch % 10 == 0:
                #     fg_display = fg.permute(1,2,0).detach().numpy()
                #     print('foreground shape: {}'.format(obj_fg.shape))
                #     cv2.imshow('fg', fg_display)
                #     cv2.waitKey(0)

                # obj_fg = obj_fg.permute(2,0,1)
                # if not os.path.exists('./obj/obj_fg_{}.png'.format(idx)):
                #     obj_fg_s = (np.array(obj_fg.permute(1,2,0))).astype('uint8')
                #     cv2.imshow('obj_fg_s', obj_fg_s)
                #     cv2.waitKey(0)
                #     cv2.imwrite('./obj/obj_fg_{}.png'.format(idx), obj_fg_s)
                loss = criterion(pred_img, img)
                loss.backward()
                optimizer.step()
                ep_loss.append(loss)
                print('epcoh: {}, loss: {}'.format(epoch, loss))
            loss_list.append(torch.sum(torch.tensor(ep_loss))/len(ep_loss))
        torch.save(netG, './model/'+timestamp+'_{}.pth'.format(j))
            
    x = range(0, epoch_num)
    fig = plt.subplots()
    plt.plot(x, loss_list[:100], label='1')
    plt.plot(x, loss_list[100:200], label='2')
    plt.plot(x, loss_list[200:300], label='3')
    plt.xlabel('epoches')
    plt.ylabel('train loss')
    plt.legend()
    # plt.show()
    plt.savefig('C:\\Users\\hbrch\\Desktop\\Robonet\\plots\\'+timestamp+'.png')

    # (_, state) = next(iter(summer_loader))
    # state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
    # fg = torch.squeeze(netG(state)).permute(1,2,0)
    # fg = fg.detach().numpy()
    
    # cv2.imshow('generater_fg', fg)
    # cv2.waitKey()
    # cv2.imwrite('generated_fg_plus.png', fg*255)
    # np.save("generated_fg_plus.npy", fg*255) #should also save the corresponding tensor
    eval_ls = []
    for i, (v_img, state) in enumerate(summer_loader):
        state = torch.unsqueeze(torch.unsqueeze(state,-1),-1)
        v_img = torch.squeeze(v_img)
        v_fg = torch.squeeze(netG(state))
        v_fg = v_fg.permute(1,2,0)
        pred_img = v_fg*255 + bg
        eval_ls.append(criterion(pred_fg, v_img))
        # cv2.imshow('img', v_img.to(dtype=torch.uint8).numpy())
        # cv2.waitKey()
        # cv2.imshow('pred_img', pred_img.to(dtype=torch.uint8).numpy())
        # cv2.waitKey()
        diff = ((v_img - pred_img)/255).detach().numpy()
        # cv2.imshow("diff", diff)
        # cv2.waitKey()
        cv2.imwrite('./img/img_{}.png'.format(i), v_img.to(dtype=torch.uint8).numpy())
        cv2.imwrite('./pred_img/pred_img_{}.png'.format(i), pred_img.to(dtype=torch.uint8).numpy())
        cv2.imwrite('./diff=img-pred_img/diff_{}.png'.format(i), diff)
    print(eval_ls,len(eval_ls))
    pass