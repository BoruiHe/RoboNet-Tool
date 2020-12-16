import h5py
import os
import io
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class MetaDataContainer:
	def __init__(self, base_path, meta_data):
		self._meta_data = meta_data
		self._base_path = base_path

	def get_file_metadata(self, fname):
		fname = fname.split('/')[-1]
		return self._meta_data.loc[fname]

	def select_objects(self, obj_class_name):
		if isinstance(obj_class_name, str):
			return self._meta_data[[obj_class_name in x for x in self._meta_data['object_classes']]]
		return self._meta_data[[set(obj_class_name) == set(x) for x in self._meta_data['object_classes']]]

	@property
	def frame(self):
		return self._meta_data

	@property
	def files(self):
		return ['{}/{}'.format(self._base_path, f) for f in self.frame.index]

	def get_shuffled_files(self, rng=None):
		files = ['{}/{}'.format(self._base_path, f) for f in self.frame.index]
		if rng:
			rng.shuffle(files)
		else:
			random.shuffle(files)
		return files

	@property
	def index(self):
		return self._meta_data.index

	def base_path(self):
		return self._base_path

	def __getitem__(self, arg):
		return MetaDataContainer(self._base_path, self._meta_data[arg])

	def __contains__(self, item):
		return item in self._meta_data

	def __repr__(self):
		return repr(self._meta_data)

	def __str__(self):
		return str(self._meta_data)

	def __eq__(self, other):
		return self._meta_data == other

	def __ne__(self, other):
		return self._meta_data != other

	def __lt__(self, other):
		return self._meta_data < other

	def __le__(self, other):
		return self._meta_data <= other

	def __gt__(self, other):
		return self._meta_data > other

	def __ge__(self, other):
		return self._meta_data >= other

	def keys(self):
		return self._meta_data.keys()

	def __len__(self):
		return len(self._meta_data)
		
# def get_all_hdf5_files(path):
# 	assert isinstance(path, str), 'input path to directory named \'hdf5\''
# 	files = sorted(glob.glob('{}/*.hdf5'.format(path)))
# 	return files

def load_hdf5(fname):
	if not os.path.exists(fname) or not os.path.isfile(fname):
		raise IOError('cannot find {}'.format(fname))
	buf = open(fname, 'rb').read()

	metadata_dict = {}
	with h5py.File(io.BytesIO(buf)) as hf:
		
		# hf['env']['state'] is a 31*5 dataset
		metadata_dict['sdim'] = hf['env']['state'].shape[1] # 5
		metadata_dict['state_T'] = hf['env']['state'].shape[0] # 31

		# hf['policy']['actions'] is a 30*4 dataset
		metadata_dict['adim'] = hf['policy']['actions'].shape[1] # 4
		metadata_dict['action_T'] = hf['policy']['actions'].shape[0] # 30

		# number of cameras
		n_cams = hf['env'].attrs.get('n_cams', 0)
		assert n_cams > 0, 'the number of cameras is not found'
		metadata_dict['ncam'] = n_cams

		if hf['env'].attrs['cam_encoding'] == 'mp4':
			metadata_dict['img_encoding'] = 'mp4'
			metadata_dict['frame_dim'] = hf['env']['cam0_video']['frames'].attrs['shape'][:2]
			metadata_dict['img_T'] = hf['env']['cam0_video']['frames'].attrs['T']
			metadata_dict['image_format'] = hf['env']['cam0_video']['frames'].attrs['image_format']
		else:
			metadata_dict['img_encoding'] = 'jpg'
			metadata_dict['frame_dim'] = hf['env']['cam0_video']['frame0'].attrs['shape'][:2]
			metadata_dict['img_T'] = len(hf['env']['cam0_video'])
			metadata_dict['image_format'] = hf['env']['cam0_video']['frame0'].attrs['image_format']

		# misc
		for k in hf['misc'].keys():
			assert k not in metadata_dict, "key {} already present!".format(k)
			metadata_dict[k] = hf['misc'][k][()]
        
		# in metadata we can find 'robot', 'background' as labels
		for k in hf['metadata'].attrs.keys():
			assert k not in metadata_dict, "key {} already present!".format(k)
			metadata_dict[k] = hf['metadata'].attrs[k]
		
		if 'low_bound' not in metadata_dict and 'low_bound' in hf['env']:
			metadata_dict['low_bound'] = hf['env']['low_bound'][0]
        
		if 'high_bound' not in metadata_dict and 'high_bound' in hf['env']:
			metadata_dict['high_bound'] = hf['env']['high_bound'][0]
        
		return metadata_dict

def get_metadata_frame(files):
	assert isinstance(files, str), 'invalid path'

	files = sorted(glob.glob('{}/*.hdf5'.format(files)))
	if not files:
		raise ValueError('no hdf5 files found!')

	with Pool(cpu_count()) as p:
		meta_data = list(tqdm(p.imap(load_hdf5, files), total=len(files)))

	data_frame = pd.DataFrame(meta_data, index=[f.split('/')[-1] for f in files])

	return data_frame

def load_metadata(files):
	files = base_path = os.path.expanduser(files)
	return MetaDataContainer(base_path, get_metadata_frame(files))


if __name__ == "__main__":
	# # get all hdf5 files in the given directory 
	# hdf5_files = get_all_hdf5_files('C://Users//hbrch//OneDrive//Desktop//Robonet//hdf5')
	# # get dictionary for each file
	# metadata = list(map(load_hdf5, hdf5_files))

	# pd_frame = get_metadata_frame('C://Users//hbrch//OneDrive//Desktop//Robonet//hdf5')
	
	container = load_metadata('C://Users//hbrch//OneDrive//Desktop//Robonet//hdf5')
	sawyer = container[container['robot'] == 'sawyer']
	
	pass