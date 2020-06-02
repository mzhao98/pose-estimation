from dependencies import *



def load_train_data_no_pose():

	index_to_label = {
	    0: 'rowing',
	    1: 'badminton',
	    2: 'polo',
	    3: 'bocce',
	    4: 'snowboarding',
	    5: 'croquet',
	    6: 'sailing',
	    7: 'rockclimbing',
	}
	label_to_index = {v: k for k, v in index_to_label.items()}
	images_path = os.path.join('../data/')
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	image_dict = {}
	label_dict = {}

	index = 0
	for i in range(len(image_files)-100):
	# for i in range(100):
	    im = image_files[i]
	    if 'Thumb' in im or '.DS' in im:
	        continue
	    if 'rar' in im:
	        continue
	#     print(im)
	    label_str = im.split('_')[2].lower()
	    image_dict[index] = im
	    label_dict[index] = label_to_index[label_str]
	    index += 1

	return label_dict, image_dict



def load_test_data_no_pose():

	index_to_label = {
	    0: 'rowing',
	    1: 'badminton',
	    2: 'polo',
	    3: 'bocce',
	    4: 'snowboarding',
	    5: 'croquet',
	    6: 'sailing',
	    7: 'rockclimbing',
	}
	label_to_index = {v: k for k, v in index_to_label.items()}
	images_path = os.path.join('../data/')
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	image_dict = {}
	label_dict = {}

	index = 0
	for i in range(100, len(image_files)):
	# for i in range(100):
	    im = image_files[i]
	    if 'Thumb' in im or '.DS' in im:
	        continue
	    if 'rar' in im:
	        continue
	#     print(im)
	    label_str = im.split('_')[2].lower()
	    image_dict[index] = im
	    label_dict[index] = label_to_index[label_str]
	    index += 1

	return label_dict, image_dict



def load_train_data():

	index_to_label = {
	    0: 'rowing',
	    1: 'badminton',
	    2: 'polo',
	    3: 'bocce',
	    4: 'snowboarding',
	    5: 'croquet',
	    6: 'sailing',
	    7: 'rockclimbing',
	}
	label_to_index = {v: k for k, v in index_to_label.items()}
	images_path = os.path.join('../data/')
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	image_dict = {}
	label_dict = {}

	index = 0
	for i in range(len(image_files)-100):
	# for i in range(100):
	    im = image_files[i]
	    if 'Thumb' in im or '.DS' in im:
	        continue
	    if 'rar' in im:
	        continue
	#     print(im)
	    label_str = im.split('_')[2].lower()
	    image_dict[index] = im
	    label_dict[index] = label_to_index[label_str]
	    index += 1

	return label_dict, image_dict

def load_test_data():

	index_to_label = {
	    0: 'rowing',
	    1: 'badminton',
	    2: 'polo',
	    3: 'bocce',
	    4: 'snowboarding',
	    5: 'croquet',
	    6: 'sailing',
	    7: 'rockclimbing',
	}
	label_to_index = {v: k for k, v in index_to_label.items()}
	images_path = os.path.join('../data/')
	image_files = [f for f in listdir(images_path) if isfile(join(images_path, f))]

	image_dict = {}
	label_dict = {}

	index = 0
	for i in range(100, len(image_files)):
	# for i in range(100):
	    im = image_files[i]
	    if 'Thumb' in im or '.DS' in im:
	        continue
	    if 'rar' in im:
	        continue
	#     print(im)
	    label_str = im.split('_')[2].lower()
	    image_dict[index] = im
	    label_dict[index] = label_to_index[label_str]
	    index += 1

	return label_dict, image_dict









class UIUC_w_Pose_Dataset(data.Dataset):
#       '''Characterizes a dataset for PyTorch'''
    def __init__(self, labels, images, images_path, original_train_dataset, pose_net1):
        '''Initialization'''
        self.labels = labels
        self.images = images
        self.original_train_dataset = original_train_dataset
        self.net = pose_net1
        
        self.images_path = images_path
        
        self.transform = transforms.Compose(
                [
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
#                     transforms.CenterCrop(10),
                 
                 transforms.Normalize((0.5, 0.5, 0.5), 
                                      (0.5, 0.5, 0.5))])
        self.mask_transform = transforms.Compose(
                [
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                ])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.labels)

    def get_mask(self, pose_net1, train_dataset, stop_idx, image_arr):

        batch_data, batch_labels = train_dataset.__getitem__(stop_idx)

        batch_data = batch_data.unsqueeze(0)
        batch_labels = batch_labels

        batch_data = batch_data.double()
        batch_labels = batch_labels

        predicted_output = pose_net1(batch_data)

        Ry=0.53038674
        Rx=0.35294117

        idx = stop_idx

        img_mask = np.zeros((image_arr.shape[0], image_arr.shape[1]))

        predicted_output_idx = predicted_output[0]


        target_output_reshaped = predicted_output_idx.reshape((14,2))

        for (p1, p2) in target_output_reshaped:
            p1, p2 = np.round(p1.item(), 2)/Rx, np.round(p2.item(), 2)/Ry
            p1, p2 = int(p1), int(p2)
            
#             print(p1)
#             print(p2)
#             print(image_arr.shape[0])
            
            p1 = min(p1, image_arr.shape[1]-1)
            p2 = min(p2, image_arr.shape[0]-1)
            p1 = max(p1, 0)
            p2 = max(p2, 0)
            
            img_mask[p2, p1] = 1


        return img_mask


    
    
    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        image_filename = self.images[index]
        path_to_image = os.path.join(self.images_path, image_filename)

        # Load data and get label
        original_image = Image.open(path_to_image)
        img_mask = self.get_mask(self.net, self.original_train_dataset, index, np.array(original_image))
        
        image = self.transform(original_image).float()
        
        x = image
        
        img_mask = Image.fromarray(img_mask, mode='L')
        img_mask_transformed = self.mask_transform(img_mask).float()
        
#         print('img_mask_transformed.shape', img_mask_transformed.shape)
        
        
        x = np.concatenate((x, img_mask_transformed), axis=0)
#         print("x.shape", x.shape)
        
        y = torch.tensor(np.array(self.labels[index])).float()

        return x, y


        
	
class UIUC_Actions_Dataset(data.Dataset):
#       '''Characterizes a dataset for PyTorch'''
    def __init__(self, labels, images, images_path):
        '''Initialization'''
        self.labels = labels
        self.images = images
        
        self.images_path = images_path
        
        self.transform = transforms.Compose(
                [
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
#                     transforms.CenterCrop(10),
                 
                 transforms.Normalize((0.5, 0.5, 0.5), 
                                      (0.5, 0.5, 0.5))])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.labels)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        image_filename = self.images[index]
        path_to_image = os.path.join(self.images_path, image_filename)

        # Load data and get label
        image = Image.open(path_to_image)
        image = self.transform(image).float()
        x = image
#         y = torch.tensor(np.array(self.labels[index])).float()

#         print(y)
        
        y = int(self.labels[index])

        return x, y
		