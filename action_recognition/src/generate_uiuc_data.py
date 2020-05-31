from dependencies import *

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
	for i in range(len(image_files)):
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
		