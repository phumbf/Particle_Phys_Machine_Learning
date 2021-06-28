"""
A convolutional neural network project to determine the species of particle tracks passing through a liquid argon time 
projection chamber. This script was worked on as part of a project at the SLAC 2019 summer school.

The datasets can be found from https://www.nevis.columbia.edu/~kazuhiro/test_20k.h5 and
https://www.nevis.columbia.edu/~kazuhiro/train_60k.h5

These are public datasets where electrons, muons, protons, photons and charged pion tracks are displayed in 
256x256 pixel images and generated via simulation. Algorithm trained using a stochasic gradient descent.
"""

from __future__ import print_function
import os
import torch
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

#Simple deep residual network (https://arxiv.org/abs/1512.03385).
class ResidualModule(torch.nn.Module):

    #num_input = number of input features, same for output features
    #stride = number of pixels to skip for kernel
    #momentum = batch normalisation parameter
    def __init__(self, num_input, num_output, stride=1, momentum=0.9 ):
        super(ResidualModule, self).__init__()

        # residual path
        self._features = torch.nn.Sequential(
          torch.nn.Conv2d(num_input, num_output, kernel_size=3, stride=stride, padding=1, bias=False),
          torch.nn.BatchNorm2d(num_output,momentum=momentum),
          torch.nn.ReLU(inplace=True),
          torch.nn.Conv2d(num_output, num_output, kernel_size=3, padding=1, bias=False),
          torch.nn.BatchNorm2d(num_output,momentum=momentum)
        )

        # if stride > 1 subsamble the input
        self._shortcut = None
        if stride>1 or not num_input==num_output:
            self._shortcut = torch.nn.Conv2d(num_input,num_output,kernel_size=1,stride=stride,bias=False)

    def forward(self, x):
        bypass = x if self._shortcut is None else self._shortcut(x)
        residual = self._features(x)
        return torch.nn.ReLU(inplace=True)(bypass + residual)

class ResidualLayer(torch.nn.Module):

    def __init__(self, num_input, num_output, num_modules, stride=1, momentum=0.9):
        super(ResidualLayer,self).__init__()

        ops = [ ResidualModule(num_input, num_output, stride=stride, momentum=momentum) ]

        for i in range(num_modules-1):

            ops.append( ResidualModule(num_output, num_output, stride=1, momentum=momentum) )

        self._layer = torch.nn.Sequential(*ops)

    def forward(self,x):
        return self._layer(x)
    
class ResNet(torch.nn.Module):

    #num_output_base = number of filters in the first layer
    #num_class = number of filters in the last layer
    #num_input = number of input data channels
    def __init__(self, num_class, num_input, num_output_base, blocks, bn_momentum=0.9):
        super(ResNet, self).__init__()
        
        self._ops = []        
        num_output = num_output_base
        for block_index, num_modules in enumerate(blocks):
            stride = 2 if block_index > 0 else 1
            self._ops.append( ResidualLayer(num_input, num_output, num_modules, stride=stride, momentum=bn_momentum) )            
            #For the next layer, increase channel count by 2
            num_input  = num_output
            num_output = num_output * 2
            
        self._features = torch.nn.Sequential(*self._ops)
        self._classifier = torch.nn.Linear(num_input, num_class)

        def weights_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    torch.nn.init.xavier_uniform_(m.bias)
        self._features.apply(weights_init)

    def forward(self,x):        
        tensor = self._features(torch.unsqueeze(x,1))
        tensor = F.max_pool2d(tensor, kernel_size=tensor.size()[2:])
        tensor = tensor.view(-1,np.prod(tensor.size()[1:]))
        tensor = self._classifier(tensor)
        return tensor

#Define Blob class to keep objects together
class BLOB:
    pass
blob=BLOB()
blob.net       = ResNet(5,1,16,[2,2,2,3,3]).cuda() # construct Lenet, use GPU
#Define the loss function
blob.criterion = torch.nn.CrossEntropyLoss() # use softmax loss to define an error
blob.optimizer = torch.optim.Adam(blob.net.parameters(),weight_decay=0.001) # use Adam optimizer algorithm
blob.softmax   = torch.nn.Softmax(dim=1) # not for training, but softmax score for each class
blob.iteration = 0    # integer count for the number of train steps
blob.data      = None # data for training/analysis
blob.label     = None # label for training/analysis

#Class to make the dataset handling easier
class dataset:
  
    def __init__(self,fname):
        
        self._fname = fname
        self._file = h5.File(self._fname,'r')
        self._labels = np.array(self._file['label']).astype(np.int32)
        self._classes={}
        for index, pdgcode in enumerate(np.unique(self._file['label'])):
            self._classes[int(pdgcode)] = index
            
    def __del__(self):
        self._file.close()
   
   #Function to return a dict to map Particle Data Group code to class index
   #e.g. PDG code of 11 for an electron etc.
    def classes(self):
        return self._classes
   
   #Function to return array of labels for all samples
    def labels(self):
        return self._labels
    
    def __len__(self):
        return len(self._labels)
  
    def __getitem__(self,index):
        image = np.array(self._file['image0'][index])
        label = self._classes[self._labels[index]]
        return image,label

#Computes the network output by returning dict of predicted labels, softmax, loss, accuracy
def forward(blob,train=True):
    
    with torch.set_grad_enabled(train):
        # Prediction
        data = blob.data.cuda()
        prediction = blob.net(data)
        # Training
        loss,acc=-1,-1
        if blob.label is not None:
            label = blob.label.cuda()
            loss = blob.criterion(prediction,label)
        blob.loss = loss
        
        softmax    = blob.softmax(prediction).cpu().detach().numpy()
        prediction = torch.argmax(prediction,dim=-1)
        accuracy   = (prediction == label).sum().item() / float(prediction.nelement())        
        prediction = prediction.cpu().detach().numpy()
        
        return {'prediction' : prediction,
                'softmax'    : softmax,
                'loss'       : loss.cpu().detach().item(),
                'accuracy'   : accuracy}

#Determine the loss fn value and propagate gradients for weight update
def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()

#Training loop which calls forward and backward
def train_loop(blob,train_loader,num_iteration):
    # Set the network to training mode
    blob.net.train()
    # Let's record the loss at each iteration and return
    train_loss=[]
    # Loop over data samples and into the network forward function
    while blob.iteration < num_iteration:
        for data in train_loader:
            if blob.iteration >= num_iteration:
                break
            blob.iteration += 1
            # data and label
            blob.data, blob.label = data
            # call forward
            res = forward(blob,True)
            # Record loss
            train_loss.append(res['loss'])
            # Print occasionally!
            if blob.iteration == 0 or (blob.iteration+1)%100 == 0:
                print('Iteration',blob.iteration,'... Loss',res['loss'],'... Accuracy',res['accuracy'])
            backward(blob)
    return np.array(train_loss)

# For calculating the moving average
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#Function to save the state of the networj (i.e. the weights)
def save_state(blob, prefix='./snapshot'):
    # Output file name
    filename = '%s-%d.ckpt' % (prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
        }, filename)
    return filename

#Load the network state
def restore_state(blob, weight_file):
    # Open a file in read-binary mode
    with open(weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None:
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        blob.iteration = checkpoint['global_step']

#Returns accuracy per batch, label and prediction per image to test network on test sample
def inference_loop(blob,data_loader,num_iterations=300):
    # set the network to test (non-train) mode
    blob.net.eval()
    # create the result holder
    accuracy, label, prediction, softmax = [], [], [], []
    for i,data in enumerate(data_loader):
        if (i+1) == num_iterations:
            break
        blob.data, blob.label = data
        res = forward(blob,False)
        accuracy.append(res['accuracy'])
        prediction.append(res['prediction'])
        label.append(blob.label)
        softmax.append(res['softmax'])
    
    # organize the return values
    accuracy   = np.hstack(accuracy)
    prediction = np.hstack(prediction)
    label      = np.hstack(label)
    softmax    = np.vstack(softmax)
    return accuracy, label, prediction, softmax
 
#Confusion matrix useful for displaying number of correct/incorrect predictions
def plot_confusion_matrix(label,prediction,class_names):
    
    fig, ax = plt.subplots(figsize=(12,8),facecolor='w')
    num_labels = len(class_names)
    max_value  = np.max([np.max(np.unique(label)),np.max(np.unique(label))])
    assert max_value < num_labels
    mat,_,_,im = ax.hist2d(prediction,label,
                           bins=(num_labels,num_labels),
                           range=((-0.5,num_labels-0.5),(-0.5,num_labels-0.5)),cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(class_names,fontsize=16)
    ax.set_yticklabels(class_names,fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_xlabel('Prediction',fontsize=20)
    ax.set_ylabel('True Label',fontsize=20)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i,j, str(mat[i, j]),
                    ha="center", va="center", fontsize=16,
                    color="white" if mat[i,j] > (0.5*mat.max()) else "black")
    fig.tight_layout()
    plt.show()

#Initial GPU options
os.environ['CUDA_VISIBLE_DEVICES']='0'
#-1 to not run on a GPU
torch.cuda.set_device(-1)

#Random seed
SEED=123
_=np.random.seed(SEED)
_=torch.manual_seed(SEED)

#Create instances of the dataset class 
train_data=dataset('train_60k.h5')
test_data =dataset('test_20k.h5')

#Good to sanity check size of datasets
print('Size of the train dataset:', len(train_data))
print('Size of the test dataset:', len(test_data))
for pdg,label in train_data.classes().items():
    print('PDG',pdg,'has a label',label)

pdg,counts = np.unique(train_data.labels(),return_counts=True)
for i, data in enumerate(pdg):
    print('TRAIN set PDG',data,'...',counts[i],'events')

#Choose a random dataset for visualisation
pdg_list = train_data.classes().keys()
for pdg in pdg_list:
    labels = train_data.labels()
    where  = np.where(train_data.labels()==pdg)[0]
    np.random.shuffle(where)
    image,label = train_data[where[0]]
    
    print('PDG %d (label=%d)' % (pdg,label))
    plt.imshow(image,cmap='jet')
    plt.show()

#Use the DataLoader class to form a random subset of the data
train_loader = DataLoader(train_data,batch_size=64,shuffle=True,num_workers=1)
test_loader  = DataLoader(test_data, batch_size=64,shuffle=False,num_workers=1)
it = iter(train_loader)

#Create list of images/labels
#Throughout, batch refers to a subset of the data
batch_data = next(it)
image_batch = batch_data[0]
label_batch = batch_data[1]

print('Image batch shape:',image_batch.shape)
print('Label batch shape:',label_batch.shape)

for data in train_loader:
    print('Image batch shape:',image_batch.shape)
    print('Label batch shape:',label_batch.shape)
    break

loss = train_loop(blob,train_loader,num_iteration=6000)
#Plot the loss value 
fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
plt.plot(range(len(loss)),loss,marker="",linewidth=2,color='blue')

plt.plot(moving_average(range(len(loss)),30),moving_average(loss,30),marker="",linewidth=2,color='red')
plt.yscale("log")
plt.show()

#Save the network state
weight_file = save_state(blob)

#Compare the performance pre training to after the training (i.e. loaded state)
# Recreate the network (i.e. initialize)
blob.net=ResNet(5,1,16,[2,2,2,3,3]).cuda()
# Get one batch of data to test
blob.data, blob.label = next(iter(train_loader))
# Run forward function
res = forward(blob,True)
# Report
print('Accuracy:',res['accuracy'])
# Restore the state
restore_state(blob,'./snapshot-6000.ckpt')
# Run the forward function
res = forward(blob,True)
print('Accuracy',res['accuracy'])

#Run inference for train and test samples and compare for overtraining
# For the Train set
accuracy, label, prediction, softmax = inference_loop(blob,train_loader,300)
print("Train set accuracy mean",accuracy.mean(),"std",accuracy.std())
plot_confusion_matrix(label,prediction,train_data.classes().keys())

# For the Test set
accuracy, label, prediction, softmax = inference_loop(blob,test_loader,300)
print("Test set accuracy mean",accuracy.mean(),"std",accuracy.std())
plot_confusion_matrix(label,prediction,test_data.classes().keys())
