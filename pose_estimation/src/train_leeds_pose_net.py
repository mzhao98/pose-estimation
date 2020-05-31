from dependencies import *
from plot_utils import *
from generate_leeds_data import *
from pose_net import *

def train_PoseNet():
	# Load Data
	leeds_path = '../data/Leeds/lspet_dataset'
	images_path = os.path.join(leeds_path, 'images')
	annot_file = os.path.join(leeds_path, 'joints.mat')

	train_labels_dict, train_images_dict = load_train_data()
	train_dataset = Leeds_Dataset(train_labels_dict, train_images_dict, images_path)

	train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=256,
                                          shuffle=True,
                                         )
	# Parameters
	max_epochs = 5000
	lr = 0.01
	momentum = 0.9

	# CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print("cuda device = ", device)

	pose_net1 = Pose_AlexNet(num_classes=28).double().to(device)
	
	# Try different optimzers here [Adam, SGD, RMSprop]
	# optimizer = optim.RMSprop(pose_net1.parameters(), lr=lr, momentum=momentum)
	optimizer = optim.Adagrad(pose_net1.parameters(), lr=lr)

	training_losses = []

	# Generators
	training_set = train_dataset
	training_generator = train_data_loader

	loss_fn = torch.nn.MSELoss()

	# Loop over epochs
	print("Beginning Training.......................")
	for epoch in range(max_epochs):
	    # Training
	    total_epoch_loss = 0
	    for batch_idx, (batch_data, batch_labels) in enumerate(training_generator):
	        
	        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

	        batch_data = batch_data.double()
	        batch_labels = batch_labels.double()
	        
	        predicted_output = pose_net1(batch_data)
	                                        
	        predicted_output = predicted_output.double()                                
	        target_output = batch_labels
	        
	       
	        loss = loss_fn(predicted_output, target_output)

	        optimizer.zero_grad()
	        loss.backward()
	        
	        optimizer.step()  
	        
	        total_epoch_loss += loss.item()
	    
	        if batch_idx % 25 == 0:
	            print('Train Epoch: {} \tLoss: {:.6f}'.format(
	                epoch, total_epoch_loss))
	    
	    if epoch % 100 == 0:
	        with open('../saved_models/leeds1_pose_network_1.pkl', 'wb') as f:
	            torch.save(pose_net1.state_dict(), f)
	            
	    training_losses.append(total_epoch_loss)
	    
	with open('../saved_models/leeds1_pose_network_1_final.pkl', 'wb') as f:
	    torch.save(pose_net1.state_dict(), f)
	    
	with open('../saved_models/leeds1_training_losses_1.npy', 'wb') as f:
	    np.save(f, np.array(training_losses))



if __name__ == '__main__':
	train_PoseNet()






