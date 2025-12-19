###################### Entire Gradient Descent Algo from Scratch ######################
#Creating our Data and Forward Pass

import torch 

#Creating the DATA, to follow line y=2x+1 
# (y=Wx+b, formula is simply a line with slope 2 and y-intercept at +1)
#Our batch of data will have 10 data points 
N=10 
#Each data point has 1 input feature and 1 output value 
D_in=1 
D_out=1 

#Create our input data X 
X=torch.randn(N,D_in)
print(f"\nGenerated Tensor:\n{X}\n")

#Create our true target labels y by using the "true" W and b
#The "true" W is 2.0, the "true" b is 1.0 
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0) 
y_true = X @ true_W + true_b + torch.randn(N,D_out) * 0.1 #0.1 adds alittle noise
print(f"\nTarget W and b values:\n{y_true}\n")

#General Model's initial hypothesis - WRONG but a starting place
#Initialize our parameters with random values. 
#Shapes must be correct for matrix multiplication! 
#BTW these are the "knobs" that are being adjusted during training. 
W=torch.randn(D_in,D_out,requires_grad=True)  
b=torch.randn(1,requires_grad=True)

print(f"\nInitial Weight W:\n{W}")
print(f"\nInitial Bias b:\n{b}") 

#Forward Pass - Simple Linear Regression, 
# X is out input data, W is the weight, and b is the bias
# y_hat is our model's prediction, or guess
# Goal is to find the perfect values for W and b to get y_hat as close to y (expected value)
y_hat = X @ W + b

print(y_hat, y_true) 

###################### STEP 2 ###################### 
# Backward Pass or Backward Propagation
# Which direction to increase or decrease the value of the Weights and Bias...
# We need an Error/Loss Function to do this. Something that calculates how off we are. 
# Using Mean Squared Error Loss Function becasue this si simple regression

#MSE Define 
error = y_hat - y_true # Error is the difference between the target and the predicted y.
squared_error = error ** 2 # Then simply square it
loss = squared_error.mean() # Then take the mean. 
# This is the Mean Squared Error Loss Function ^

# WE WANT THE LOSS AS SMALL AS POSSIBLE 
print(f"Loss (single scorecard number): {loss}")

# Compute Gradients
loss.backward() # Very critical! This populated the .grad attribute for our W and b tensors. 

# Now the gradients are stored in the ".grad" attributes of our paramters W and b 
print(f"\nGradient for W (partial derivative):\n{W.grad}")
print(f"\nGradient for b (partial derivative):\n{b.grad}")
# These gradients tell us which direction we need to nudge the W and b parameters. 
# So a NEGATIVE sign/gradient tells us that increasing the parameter decreases the loss. 
# Remember. Trying to get the loss as close to 0 as possible.  

#Gradient Descent and Training
print("TRAINING LOOP")
#Hyperparameters 
learning_rate, epochs = 0.01, 100 

#Re-initialize parameters, be sure requires_grad = True, this is what tracks the derivatives
W = torch.randn(1,1,requires_grad=True)
b = torch.randn(1,requires_grad=True)

#Training Loop 
for epoch in range(epochs): 
    
    #Forward pass and loss - Makes guess and calculates loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat-y_true)**2)

    #Backward pass - Calculates the gradients given the loss for the parameters and 
    # their derivates. 
    loss.backward() 

    #Update parameters - Gradient Descent step where we are nudging the parameters 
    # in the direct to reduce loss. in the next interation
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    
    #Zero out Gradients for next iteration
    W.grad.zero_()
    b.grad.zero_()
    
    #Lets print out to see whats happening 
    if epoch % 10 == 0: 
        print(f"Epoch {epoch:02d}: Loss: {loss.item():.4f}, W= {W.item():.3f}, b= {b.item():.3f}")

print(f"\nFinal Parameters:  W= {W.item():.3f}, b= {b.item():.3f}")
print(f"True Parameters: W=2.000, b=1.000") 

#OUTPUT - it works. Notice how the values for W and b are approching the target values of 
# true_W = 2.0 and true_b = 1 AS the loss decreases. This is how gradient descent works. 
'''Gradient for W (partial derivative):
tensor([[-6.9960]])

Gradient for b (partial derivative):
tensor([-2.4000])
TRAINING LOOP
Epoch 00: Loss: 2.3759, W= 1.427, b= -0.171
Epoch 10: Loss: 1.2599, W= 1.650, b= 0.064
Epoch 20: Loss: 0.6916, W= 1.797, b= 0.243
Epoch 30: Loss: 0.3958, W= 1.893, b= 0.380
Epoch 40: Loss: 0.2375, W= 1.954, b= 0.486
Epoch 50: Loss: 0.1499, W= 1.992, b= 0.569
Epoch 60: Loss: 0.0995, W= 2.014, b= 0.635
Epoch 70: Loss: 0.0693, W= 2.025, b= 0.688
Epoch 80: Loss: 0.0505, W= 2.031, b= 0.730
Epoch 90: Loss: 0.0383, W= 2.032, b= 0.764

Final Parameters:  W= 2.031, b= 0.790
True Parameters: W=2.000, b=1.000''' 

 #The input has 1 feature and the output has 1 value - Good place to start.
D_in = 1
D_out = 1
#Create the Linear Layer - simply a linear regression
linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)
#Lets take a look inside the parameters it has created
print((f"Layer's Weight (W): {linear_layer.weight}\n"))
print((f"Layer's Bias (b): {linear_layer.bias}\n"))
#This is the forward pass. You can use it just like a function.
#(Assume X is a tensor of shape [10,1] from previous chapters)
y_hat_nn = linear_layer(X)
print(f"Output of nn.Linear (first 3 rows):\n {y_hat_nn[:3]}")
#Activation Functions - These decided which neurons turn on or off. 
# If you just hadd a bunhc nof linear layers stacked on top of each other you would end 
# up with just a large linear regression. Activation functions add non-linearity between 
# linear layers. 

# In this case we will use a Rectified Linear Unit (RELU) Activation Function
relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0,-0.5,0.0,0.5,2.0])
activated_data = relu(sample_data)

print(f"Original Data: {sample_data}\n")
print(f"Data after ReLu: {activated_data}\n")
'''Output: 
Original Data: tensor([-2.0000, -0.5000,  0.0000,  0.5000,  2.0000])
Data after ReLu: tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])'''
#ReLu will flatten any data below 0 by default and activate (slope) above 0. 

#GeLu, Gaussian Error Linear Unit - this will not just flatten data below 0 to 0. It 
# will heavily push the data there before and after 0 (pivot point) a smother less binary 
# version of ReLu 
gelu = torch.nn.GELU()
sample_data = torch.tensor([-2.0,-0.5,0.0,0.5,2.0])
activated_data = gelu(sample_data)

print(f"Original Data: {sample_data}\n")
print(f"Data after GeLu: {activated_data}\n") 
'''Output: 
Original Data: tensor([-2.0000, -0.5000,  0.0000,  0.5000,  2.0000])
Data after GeLu: tensor([-0.0455, -0.1543,  0.0000,  0.3457,  1.9545])''' 

#Softmax - Used on the final output layer normally for classification tasks 
# Converts logits (raw parameter outputs) into a probability distribution. 
# What this means is that all outputs are forced to be between 0 and 1 and all sum to 1. 
softmax = torch.nn.Softmax(dim=-1)
#Simulated raw data
logits = torch.tensor([[1.0,3.0,0.5,1.5],[-1.0,2.0,1.0,0.0]])
probabilities = softmax(logits)

print(f"Output Probabilities: {probabilities}\n")
print(f"Sum of probabilities: {probabilities[0].sum()}\n")