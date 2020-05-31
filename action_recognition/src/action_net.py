from dependencies import *


class ActionNet_4Conv(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
        # torch.Size([256, 3, 96, 96])
        # 3 input image channel (RGB), #6 output channels, 4x4 kernel 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 96, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(96, 256, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv4 = nn.Conv2d(256, 384, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')

        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(55296, 4096)
        self.fc2 = nn.Linear(4096, 1028)
        self.fc3 = nn.Linear(1028, 397)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.norm1(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.norm2(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
#         print(x.shape)
        
        x = F.relu(self.conv4(x))
        
        x = torch.flatten(x, 1)
#         print(x.shape)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)
#         x = self.drop1(x)
        
#         output = x
        output = F.log_softmax(x, dim=1)
        return output
        
        



class ActionNet(nn.Module):
    def __init__(self):
        super(ActionNet, self).__init__()
        # torch.Size([256, 3, 96, 96])
        # 3 input image channel (RGB), #6 output channels, 4x4 kernel 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv2 = nn.Conv2d(32, 96, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        self.conv3 = nn.Conv2d(96, 256, kernel_size=(3,3), stride=1, 
                               padding=1, dilation=1, groups=1, 
                               bias=True, padding_mode='reflect')
        
        
        self.drop1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm([48, 48])
        self.norm2 = nn.LayerNorm([24, 24])
        
        self.fc1 = nn.Linear(36864, 4096)
        self.fc2 = nn.Linear(4096, 1028)
        self.fc3 = nn.Linear(1028, 8)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.norm1(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.norm2(x)
#         print(x.shape)
        
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
#         print(x.shape)

        x = torch.flatten(x, 1)
#         print(x.shape)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        x = F.relu(x)
#         x = self.drop1(x)
        
#         output = x
        output = F.log_softmax(x, dim=1)
        return output
        
        

