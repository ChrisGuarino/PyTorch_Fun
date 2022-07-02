#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from random import shuffle
import torch  
import torchvision  
import torchvision.transforms as transforms

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    batch_size = 4 

    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,batch_size,shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck') 

    import matplotlib.pyplot as plt 
    import numpy as np 

    #Function to Visualize the Data
    def imshow(img): 
        img = img /2+0.5 #This unnormalies the data
        npimg = img.numpy() 
        plt.imshow(np.transpose(npimg,(1,2,0))) 
        plt.show() 

    #Get some Random Training Images 
    detailer = iter(trainloader)
    images,labels = detailer.next() 

    #Show Images 
    imshow(torchvision.utils.make_grid(images))

    #Print Labels 
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

