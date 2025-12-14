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