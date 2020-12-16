import h5py
import cv2
import imageio
import io
import numpy as np
import os
import random
import matplotlib.pyplot as pl


class ACTION_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


class STATE_MISMATCH:
    ERROR = 0
    PAD_ZERO = 1
    CLEAVE = 2


def default_loader_hparams():
    return {
            'target_adim': 4,
            'target_sdim': 5,
            'state_mismatch': STATE_MISMATCH.ERROR,     # TODO make better flag parsing
            'action_mismatch': ACTION_MISMATCH.ERROR,   # TODO make better flag parsing
            'img_size': [48, 64],
            'cams_to_load': [],
            'impute_autograsp_action': True,
            'load_annotations': False,
            'zero_if_missing_annotation': False, 
            # 'load_T': 0                               # TODO implement error checking here for jagged reading
            }


def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    target_height, target_width = target_dims
    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA
    
    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if start_time <= t < n_load + start_time]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1] 
                                for t in range(start_time, start_time + n_load)]
    else: 
        raise ValueError("encoding not supported")
    
    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)
    
    if image_format == 'RGB':
        return images
    elif image_format == 'BGR':
        return images[:, :, :, ::-1]
    raise NotImplementedError

def load_states(file_pointer, meta_data, hparams):
    s_T, sdim = meta_data['state_T'], meta_data['sdim']
    hparams['target_sdim'] = sdim
    if hparams['target_sdim'] == sdim:
        return file_pointer['env']['state'][:]

    elif sdim < hparams['target_sdim'] and hparams['state_mismatch'] & STATE_MISMATCH.PAD_ZERO:
        pad = np.zeros((s_T, hparams['target_sdim'] - sdim), dtype=np.float32)
        return np.concatenate((file_pointer['env']['state'][:], pad), axis=-1)

    elif sdim > hparams['target_sdim'] and hparams['state_mismatch'] & STATE_MISMATCH.CLEAVE:
        return file_pointer['env']['state'][:][:, :hparams['target_sdim']]

    else:
        raise ValueError("file sdim - {}, target sdim - {}, pad behavior - {}".format(sdim, hparams['target_sdim'], hparams['state_mismatch']))

def load_actions(file_pointer, meta_data, hparams):
    a_T, adim = meta_data['action_T'], meta_data['adim']
    hparams['target_adim'] = adim
    if hparams['target_adim'] == adim:
        return file_pointer['policy']['actions'][:]

    elif hparams['target_adim'] == adim + 1 and hparams['impute_autograsp_action'] and meta_data['primitives'] == 'autograsp':
        action_append, old_actions = np.zeros((a_T, 1)), file_pointer['policy']['actions'][:]
        next_state = file_pointer['env']['state'][:][1:, -1]
        
        high_val, low_val = meta_data['high_bound'][-1], meta_data['low_bound'][-1]
        midpoint = (high_val + low_val) / 2.0

        for t, s in enumerate(next_state):
            if s > midpoint:
                action_append[t, 0] = high_val
            else:
                action_append[t, 0] = low_val
        return np.concatenate((old_actions, action_append), axis=-1)

    elif adim < hparams['target_adim'] and hparams['action_mismatch'] & ACTION_MISMATCH.PAD_ZERO:
        pad = np.zeros((a_T, hparams['target_adim'] - adim), dtype=np.float32)
        return np.concatenate((file_pointer['policy']['actions'][:], pad), axis=-1)

    elif adim > hparams['target_adim'] and hparams['action_mismatch'] & ACTION_MISMATCH.CLEAVE:
        return file_pointer['policy']['actions'][:][:, :hparams['target_adim']]

    else:
        raise ValueError("file adim - {}, target adim - {}, pad behavior - {}".format(adim, hparams['target_adim'], hparams['action_mismatch']))

def load_annotations(file_pointer, metadata, hparams, cams_to_load):
    old_height, old_width = metadata['frame_dim']
    target_height, target_width = hparams.img_size
    scale_height, scale_width = target_height / float(old_height), target_width / float(old_width)
    annot = np.zeros((metadata['img_T'], len(cams_to_load), target_height, target_width, 2), dtype=np.float32)
    if metadata.get('contains_annotation', False) != True and hparams.zero_if_missing_annotation:
        return annot

    assert metadata['contains_annotation'], "no annotations to load!"
    point_mat = file_pointer['env']['bbox_annotations'][:].astype(np.int32)

    for t in range(metadata['img_T']):
        for n, chosen_cam in enumerate(cams_to_load):
            for obj in range(point_mat.shape[2]):
                h1, w1 = point_mat[t, chosen_cam, obj, 0] * [scale_height, scale_width] - 1
                h2, w2 = point_mat[t, chosen_cam, obj, 1] * [scale_height, scale_width] - 1
                h, w = int((h1 + h2) / 2), int((w1 + w2) / 2)
                annot[t, n, h, w, obj] = 1
    return annot

def load_data(f_name, file_metadata, hparams):
    # What higher parameters do I need? type: dictionary

    # rng = random.Random(rng)

    assert os.path.exists(f_name) and os.path.isfile(f_name), "invalid f_name"
    with open(f_name, 'rb') as f:
        buf = f.read()
    
    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor/!"

        ''' Keep every time step. So, hparams.load_T should be equal to the minimum time step in meta_data. See line 77 and 93. '''
        # # load_T indicates how many time steps I want to load. See line 161.
        # if 1 < hparams.load_T < n_states:
        #     start_time = rng.randint(0, n_states - hparams.load_T)
        #     n_states = hparams.load_T
        
        # cams_to_load indicates from which cameras I extract images
        # assert all([0 <= i < file_metadata['ncam'] for i in hparams.cams_to_load]), "cams_to_load out of bounds!"

        if not hparams['cams_to_load']:
            for i in range(file_metadata['ncam']):
                hparams['cams_to_load'].append(i)
        images, selected_cams = [], []
        for cam_index in hparams['cams_to_load']:
            # I should add an extra label called 'camera' that indicates from which camera I extract the image and return it as label
            img = load_camera_imgs(cam_index, hf, file_metadata, hparams['img_size'], start_time, n_states)[None]
            images.append(img)
            # Do not append cam_index, it should be attached as a lable. selected_cams is for load_annotations.
            selected_cams.append(cam_index)
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)
        
        actions = load_actions(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states-1]
        states = load_states(hf, file_metadata, hparams).astype(np.float32)[start_time:start_time + n_states]

        if hparams['load_annotations']:
            annotations = load_annotations(hf, file_metadata, hparams, selected_cams)[start_time:start_time + n_states]
            return images, actions, states, annotations

    return images, actions, states, selected_cams

def show_image(imgs=None, state=None, camera_index=None):
    assert (imgs != None).all(), "There is 'None' value in imgs matrix!"
    assert 0 <= state < imgs.shape[0], "You want the image at time step {} but there are only {} times steps".format(state, imgs.shape[0])
    assert 0 <= camera_index < imgs.shape[1], "You want the image from camera #{} but there are only {} cameras".format(camera_index, imgs.shape[1])
    pl.imshow(imgs)
    pl.savefig('demo_image.png')
    pl.show

if __name__ == '__main__':
    import get_metadata
    import random

    ''' Robonet_Database contains 700 dictionaries. Each dictionary stores image/state/action extracted from one .hdf5 file. '''
    Robonet_Database = []
    hdf5_directory = 'C://Users//hbrch//OneDrive//Desktop//Robonet//hdf5' # you should substitute my path with your path.
    meta_data = get_metadata.load_metadata(hdf5_directory)

    ''' The code from line 198 to 202 is a simple example, which is extracting image/state/action from a single .hdf5 file.'''
    h5file = 'hdf5\\berkeley_sawyer_traj11308.hdf5' # you should substitute my path with your path.
    hparams = default_loader_hparams()
    imgs, actions, states, labels = load_data(h5file, meta_data.get_file_metadata(h5file), hparams=hparams)
    #imgs(which state, which camera)
    show_image(imgs, 1, 10)

    '''The code below extracts image/state/action from every .hdf5 file in the hdf5_directory'''
    for h5file in list(meta_data.index):
        data = {}
        hparams = default_loader_hparams()
        imgs, actions, states, labels = load_data(h5file, meta_data.get_file_metadata(h5file), hparams=hparams)
        #imgs(which state, which camera)
        show_image(imgs[0,0])
        data['file_name'] = h5file
        data['image'] = imgs
        data['action'] = actions
        data['state'] = states
        data['label'] = labels
        Robonet_Database.append(data)
    pass