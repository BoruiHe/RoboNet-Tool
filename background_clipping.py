def create_range():
    dic = {}
    for i in range(256):
        dic[i] = 0
    return dic

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle


    rd = pickle.load(open("M.pkl", 'rb'))
    # target_size = 31 * 48 * 64 * 3
    target_img = rd['baxter'][0]['image'][:,0,:,:,:]
    # target_sta = rd['baxter'][0]['image']
    img = []
    for idx in range(31):
        img.append((target_img[idx], idx+1))

    img_r, img_g, img_b = {}, {}, {}
    for i in range(48):
        for j in range(64):
            r_range = create_range()
            g_range = create_range()
            b_range = create_range()
            for z in range(31):
                r_range[target_img[z, i, j, 0]] += 1
                g_range[target_img[z, i, j, 1]] += 1
                b_range[target_img[z, i, j, 2]] += 1
            img_r[i,j] = list(r_range.values()).index(max(list(r_range.values())))
            img_g[i,j] = list(g_range.values()).index(max(list(g_range.values())))
            img_b[i,j] = list(b_range.values()).index(max(list(b_range.values())))
    re_r = np.empty([48, 64], dtype=np.int16)
    re_g = np.empty([48, 64], dtype=np.int16)
    re_b = np.empty([48, 64], dtype=np.int16)
    for coordinate in img_r.keys():
        re_r[coordinate[0], coordinate[1]] = img_r[coordinate]
        re_g[coordinate[0], coordinate[1]] = img_g[coordinate]
        re_b[coordinate[0], coordinate[1]] = img_b[coordinate]
    # such an image may never exist in dataset
    bg = np.stack((re_r, re_g, re_b), axis=-1)
    plt.imshow(bg)
    plt.show()
    pass