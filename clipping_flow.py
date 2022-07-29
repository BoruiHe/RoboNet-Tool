if __name__ == "__main__":
    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import glob
    import time
    import cv2
    import os


    # target_size = 31 * 240 * 320 * 3
    target_img = []
    files = sorted(glob.glob('./M2/*.png'))
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
        new_img[col] = np.empty([240, 320], dtype=np.int16)
        for i in range(240):
            for j in range(320):
                a = list(target_img[:,i,j,col])
                new_img[col][i,j] = Counter(a).most_common(1)[0][0]

    bg = np.stack((new_img[2], new_img[1], new_img[0]), axis=-1)
    cv2.imshow('bg', bg)
    cv2.waitKey(0)
    cv2.imwrite('faster_2_bg.png', bg)
    pass