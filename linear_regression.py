# We train the model which predict the expected salary of a employee on the base of theri experience
# it's a linear regression model

#Import all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

#Read the data
df = pd.read_csv("C:/Users/MasteR/Desktop/dataset.csv", header=None)

#Data pre-processing
df = df.drop_duplicates()

#Data Splitting
X = df.iloc[:,0] # X is the input feature
Y = df.iloc[:,1] # Y is the actual output

#Data Normalization
meanX = np.mean(X)
meanY = np.mean(Y)
stdX = np.std(X)
stdY = np.std(Y)

X = (X - meanX)/stdX
Y = (Y - meanY)/stdY

#Data Visualization
plt.scatter(X,Y)
plt.show()
plt.close()

#Making rank 2 arrays
X = X.to_numpy()[:,np.newaxis]
Y = Y.to_numpy()[:,np.newaxis]
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
m,col = X.shape
ones = np.ones((m,1))
X = np.hstack((ones,X))
theta = np.zeros((2,1))
alpha = 0.01
iterations = 7000

#Cost function or Loss function of linear regression (having single feature)
def cost_function(X,Y,theta):
    error = (np.dot(X,theta))-Y
    sqr_err = np.power(error,2)
    sum_sqrError = np.sum(sqr_err)
    j = (1/2*m)*sum_sqrError
    return j

#Gradient decent algorithm
def gradient_decent(X,Y,theta,alpha,m,iterations):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        error = (np.dot(X,theta))-Y
        loss = (np.dot(X.T,error))*alpha/m
        theta = theta - loss
        history[i] = cost_function(X,Y,theta)
    return(history,theta)

(h,theta) = gradient_decent(X,Y,theta,alpha,m,iterations)

#Calling the function getting best fit line
(h,theta) = gradient_decent(X,Y,theta,alpha,m,iterations)
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],np.dot(X,theta),color = 'g')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
plt.close()

#Get prediction
def get_prediction(x,theta):
    x = (x - meanX)/stdX
    y = theta[0,0]+theta[1,0]*x
    y = (y+stdY+meanY) #Normalize the block

    return y

#Take the experience in months from user and model predict the salary base on their experience

experience = float(input("Enter your experience in months (e.g 12 or 12.34)"))
print(f"The expected salary is {get_prediction(experience,theta):.3f}$")