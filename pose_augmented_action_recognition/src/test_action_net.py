from dependencies import *
from plot_utils import *
from generate_uiuc_data import *
from action_net import *


def eval_ActionNet_w_pose_on_train():
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

	label_dict, image_dict = load_train_data()

	# Load Data
	PATH = '../../Exp2_pose_on_leeds/saved_models/leeds1_pose_network_1_final_finished.pkl'
	pose_net1 = Pose_AlexNet(num_classes=28).double()
	pose_net1.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
	pose_net1.eval()

	original_train_dataset = UIUC_Actions_Dataset(label_dict,
												  image_dict, images_path)

	train_dataset = UIUC_w_Pose_Dataset(label_dict,
										image_dict, images_path, original_train_dataset, pose_net1)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
													batch_size=256,
													shuffle=True,
													)
	# Parameters
	max_epochs = 3000
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	action_net1 = ActionNet_w_Pose().double()
	action_PATH = '../saved_models/uiuc_action_network_5_final.pkl'
	action_net1.load_state_dict(torch.load(action_PATH, map_location=torch.device('cpu')))
	action_net1.eval()


	# Generators
	# training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.NLLLoss()
	# loss_fn = nn.CrossEntropyLoss()

	# Loop over epochs
	print("Beginning Testing..................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data = batch_data.double()
		batch_labels = batch_labels

		predicted_output = action_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels.long()

		# print(predicted_output)
		# print()
		# print(target_output)

		loss = loss_fn(predicted_output, target_output)
		#         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

		total_epoch_loss += loss.item()

	return total_epoch_loss, total_epoch_loss/len(train_dataset)


def eval_ActionNet_w_pose_on_test():
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

	label_dict, image_dict = load_test_data()

	# Load Data
	PATH = '../../Exp2_pose_on_leeds/saved_models/leeds1_pose_network_1_final_finished.pkl'
	pose_net1 = Pose_AlexNet(num_classes=28).double()
	pose_net1.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
	pose_net1.eval()

	original_train_dataset = UIUC_Actions_Dataset(label_dict,
												  image_dict, images_path)

	train_dataset = UIUC_w_Pose_Dataset(label_dict,
										image_dict, images_path, original_train_dataset, pose_net1)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
													batch_size=256,
													shuffle=True,
													)
	# Parameters
	max_epochs = 3000
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	action_net1 = ActionNet_w_Pose().double()
	action_PATH = '../saved_models/uiuc_action_network_5_final.pkl'
	action_net1.load_state_dict(torch.load(action_PATH, map_location=torch.device('cpu')))
	action_net1.eval()



	# Generators
	# training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.NLLLoss()
	# loss_fn = nn.CrossEntropyLoss()

	# Loop over epochs
	print("Beginning Testing..................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data = batch_data.double()
		batch_labels = batch_labels

		predicted_output = action_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels.long()

		# print(predicted_output)
		# print()
		# print(target_output)

		loss = loss_fn(predicted_output, target_output)
		#         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

		total_epoch_loss += loss.item()

	return total_epoch_loss, total_epoch_loss/len(train_dataset)

def eval_ActionNet_without_pose_on_train():
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

	label_dict, image_dict = load_train_data_no_pose()

	# Load Data
	train_dataset = UIUC_Actions_Dataset(label_dict,
												  image_dict, images_path)


	train_data_loader = torch.utils.data.DataLoader(train_dataset,
													batch_size=256,
													shuffle=True,
													)
	# Parameters

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	action_net1 = ActionNet().double()
	action_PATH = '../saved_models/uiuc_action_network_3_final.pkl'
	action_net1.load_state_dict(torch.load(action_PATH, map_location=torch.device('cpu')))
	action_net1.eval()


	# Generators
	# training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.NLLLoss()
	# loss_fn = nn.CrossEntropyLoss()

	# Loop over epochs
	print("Beginning Testing..................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data = batch_data.double()
		batch_labels = batch_labels

		predicted_output = action_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels.long()

		# print(predicted_output)
		# print()
		# print(target_output)

		loss = loss_fn(predicted_output, target_output)
		#         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

		total_epoch_loss += loss.item()

	return total_epoch_loss, total_epoch_loss/len(train_dataset)


def eval_ActionNet_without_pose_on_test():
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

	label_dict, image_dict = load_test_data_no_pose()

	# Load Data
	train_dataset = UIUC_Actions_Dataset(label_dict,
												  image_dict, images_path)


	train_data_loader = torch.utils.data.DataLoader(train_dataset,
													batch_size=256,
													shuffle=True,
													)
	# Parameters

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	action_net1 = ActionNet().double()
	action_PATH = '../saved_models/uiuc_action_network_3_final.pkl'
	action_net1.load_state_dict(torch.load(action_PATH, map_location=torch.device('cpu')))
	action_net1.eval()



	# Generators
	# training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.NLLLoss()
	# loss_fn = nn.CrossEntropyLoss()

	# Loop over epochs
	print("Beginning Testing..................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data = batch_data.double()
		batch_labels = batch_labels

		predicted_output = action_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels.long()

		# print(predicted_output)
		# print()
		# print(target_output)

		loss = loss_fn(predicted_output, target_output)
		#         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

		total_epoch_loss += loss.item()

	return total_epoch_loss, total_epoch_loss/len(train_dataset)



def train_ActionNet():
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


	label_dict, image_dict = load_train_data()

	# Load Data
	original_train_dataset = UIUC_Actions_Dataset(label_dict, 
                              image_dict, images_path)

	# original_train_data_loader = torch.utils.data.DataLoader(original_train_dataset,
	#                                           batch_size=256,
	#                                           shuffle=True,
	#                                          )

	PATH = '../../leeds/saved_models/leeds1_pose_network_1.pkl'
	pose_net1 = AlexNet(28).double()
	pose_net1.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
	pose_net1.eval()


	train_dataset = UIUC_w_Pose_Dataset(label_dict, 
	                              image_dict, images_path, original_train_dataset, pose_net1)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
	                                          batch_size=256,
	                                          shuffle=True,
	                                         )


	

	# Parameters
	max_epochs = 3000
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)


	action_net1 = ActionNet_w_Pose().double()



	# Try different optimzers here [Adam, SGD, RMSprop]
	optimizer = optim.SGD(action_net1.parameters(), lr=lr, momentum=momentum)


	training_losses = []

	# Generators
	training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.NLLLoss()
	# loss_fn = nn.CrossEntropyLoss()

	action_net1.train()

	# Loop over epochs
	print("Beginning Training..................")
	for epoch in range(max_epochs):
	    # print("epoch: ", epoch)
	    # Training
	    total_epoch_loss = 0
	    for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):
	        
	        batch_data = batch_data.double()
	        batch_labels = batch_labels
	        
	        predicted_output = action_net1(batch_data)
	                                        
	        predicted_output = predicted_output.double()                                
	        target_output = batch_labels.long()
	        
	        # print(predicted_output)
	        # print()
	        # print(target_output)
	        
	        
	        
	        loss = loss_fn(predicted_output, target_output)
	#         loss = F.nll_loss(predicted_output, target_output)   # Compute loss

	        optimizer.zero_grad()
	        loss.backward()
	        
	        optimizer.step()  
	        
	        total_epoch_loss += loss.item()
	    
	        if batch_idx % 25 == 0:
	            print('Train Epoch: {} \tLoss: {:.6f}'.format(
	                epoch, total_epoch_loss))
	    
	    if epoch % 100 == 0:
	        with open('../saved_models/uiuc_action_network_4.pkl', 'wb') as f:
	            torch.save(action_net1.state_dict(), f)
	            
	    training_losses.append(total_epoch_loss)
	    
	with open('../saved_models/uiuc_action_network_4_final.pkl', 'wb') as f:
	    torch.save(action_net1.state_dict(), f)
	    
	with open('../saved_models/uiuc_training_losses_4.npy', 'wb') as f:
	    np.save(f, np.array(training_losses))



if __name__ == '__main__':
	test_loss, avg_test_loss = eval_ActionNet_without_pose_on_test()
	train_loss, avg_train_loss = eval_ActionNet_without_pose_on_train()
	print("Action UIUC without Pose NLL Test loss = ", test_loss)
	print("Action UIUC without Pose NLL Avg Test loss = ", avg_test_loss)

	print("Action UIUC without Pose NLL Train loss = ", train_loss)
	print("Action UIUC without Pose NLL Avg Train loss = ", avg_train_loss)






