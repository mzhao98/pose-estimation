from dependencies import *

def load_train_data():

	replacements = [(0,1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 4),
	            (6, 7), (7, 8), (8, 12), (9, 12), (10, 11),
	            (11, 10), (12, 13), (13, 12)]

	leeds_path = '../data/Leeds/lspet_dataset'
	images_path = os.path.join(leeds_path, 'images')
	annot_file = os.path.join(leeds_path, 'joints.mat')
	joints_contents = sio.loadmat(annot_file)
	joints = joints_contents['joints']

	joints = np.swapaxes(joints, 2, 0)
	joints = np.swapaxes(joints, 1, 2)

	image_files = [f for f in listdir(images_path)]
	image_files.sort()

	train_labels_dict = {}
	train_images_dict = {}

	Ry=0.53038674
	Rx=0.35294117

	for index in range(9500):

	    curr_img = image_files[index]
	    curr_joints = joints[index, :, :2]
	    visible_joints = joints[index, :, 2]

	#     curr_img = os.path.join(images_path, test_img)
	    
	    for j in range(len(curr_joints)):
	        new_x = Rx * curr_joints[j][0]
	        new_y = Ry * curr_joints[j][1]
	        curr_joints[j][0] = new_x
	        curr_joints[j][1] = new_y
	    
	    for i in range(len(visible_joints)):
	        if visible_joints[i] == 0:
	            replacement_index = replacements[i][1]
	            if visible_joints[replacement_index] == 0:
	                new_xy = np.array([np.mean(curr_joints[:,0]), 
	                                   np.mean(curr_joints[:,1])])
	                curr_joints[i] = new_xy
	            else:
	                curr_joints[i] = curr_joints[replacement_index]
	    
	    curr_joints = curr_joints.flatten()
	    
	    
	    
	    train_labels_dict[index] = curr_joints
	    train_images_dict[index] = curr_img

	return train_labels_dict, train_images_dict

class Leeds_Dataset(data.Dataset):
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
        y = torch.tensor(np.array(self.labels[index])).float()

        return x, y
		