from dependencies import *
from plot_utils import *
from generate_leeds_data import *
from pose_net import *
def eval_PoseNet_on_train():
	# Load Data
	leeds_path = '../../data/Leeds/lspet_dataset'
	images_path = os.path.join(leeds_path, 'images')
	annot_file = os.path.join(leeds_path, 'joints.mat')

	train_labels_dict, train_images_dict = load_train_data()
	train_dataset = Leeds_Dataset(train_labels_dict, train_images_dict, images_path)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=256,
                                          shuffle=True,
                                         )
	# Parameters
	max_epochs = 1
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	pose_net1 = Pose_AlexNet(num_classes=28).double()

	model_file = '../saved_models/leeds1_pose_network_1_final_finished.pkl'
	pose_net1.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
	pose_net1.eval()

	# Generators
	training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.MSELoss()

	# Loop over epochs
	print("Beginning Testing.......................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

		batch_data = batch_data.double()
		batch_labels = batch_labels.double()

		predicted_output = pose_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels


		loss = loss_fn(predicted_output, target_output)

		total_epoch_loss += loss.item()

	return total_epoch_loss





def eval_PoseNet_on_test():
	# Load Data
	leeds_path = '../../data/Leeds/lspet_dataset'
	images_path = os.path.join(leeds_path, 'images')
	annot_file = os.path.join(leeds_path, 'joints.mat')

	train_labels_dict, train_images_dict = load_test_data()
	train_dataset = Leeds_Dataset(train_labels_dict, train_images_dict, images_path)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=256,
                                          shuffle=True,
                                         )
	# Parameters
	max_epochs = 1
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	pose_net1 = Pose_AlexNet(num_classes=28).double()

	model_file = '../saved_models/leeds1_pose_network_1_final_finished.pkl'
	pose_net1.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
	pose_net1.eval()

	# Generators
	training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.MSELoss()

	# Loop over epochs
	print("Beginning Testing.......................")

	total_epoch_loss = 0
	for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):

		batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

		batch_data = batch_data.double()
		batch_labels = batch_labels.double()

		predicted_output = pose_net1(batch_data)

		predicted_output = predicted_output.double()
		target_output = batch_labels


		loss = loss_fn(predicted_output, target_output)

		total_epoch_loss += loss.item()

	return total_epoch_loss


if __name__ == '__main__':
	test_loss = eval_PoseNet_on_test()
	print("Leeds MSE Test loss = ", test_loss / 500)

	train_loss = eval_PoseNet_on_train()
	print("Leeds MSE Train loss = ", train_loss/9500)






