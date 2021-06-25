if __name__ == "__main__":
    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import glob
    import time
    import cv2
    import os


    begin = time.time()
    small_dictionary = pickle.load(open("./loader/penn_baxter_left_traj1025.hdf5.pkl", 'rb'))
    if not os.path.exists('./M'):
        i = 0
        for img in small_dictionary['imgs'][:,0,:,:,:]:
            plt.imshow(img)
            plt.savefig('./M/{}.png'.format(i))
            i += 1

    # target_size = 31 * 480 * 640 * 3
    target_img = []
    files = sorted(glob.glob('./M/*.png'))
    for ele in files:
        image = cv2.imread(ele)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        target_img.append(image)
    target_img = np.stack(target_img, axis=0)
    # target_sta = rd['baxter'][0]['image']
    # img = []
    # for idx in range(31):
    #     img.append((target_img[idx], idx+1))

    new_img = {}
    for col in range(3):
        new_img[col] = np.empty([480, 640], dtype=np.int16)
        for i in range(480):
            for j in range(640):
                a = list(target_img[:,i,j,col])
                new_img[col][i,j] = Counter(a).most_common(1)[0][0]

    bg = np.stack((new_img[2], new_img[1], new_img[0]), axis=-1)
    cv2.imshow('bg', bg)
    cv2.waitKey(0)
    cv2.imwrite('faster_bg.png', bg)
    print(time.time()-begin)
    pass