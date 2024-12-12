import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


#Read the data
df = pd.read_csv("C:/Users/MasteR/Desktop/dataset.csv", header=None)

#Data pre-processing
df = df.drop_duplicates()
#Data Splitting
X = df.iloc[:,0].values.reshape(-1,1) #input features
Y = df.iloc[:,1] #Actual output

#Data visualization
plt.scatter(X,Y)
plt.show()
plt.close()

model = LinearRegression()

#fit the model
model.fit(X,Y)

y_pred = model.predict(X)

#Getting best fit line
plt.scatter(X,Y,color='blue')
plt.plot(X,y_pred,color='red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()
plt.close()

#Make predictions

def predict_salary(experience):
    experience_reshape = np.array([[experience]])
    prediction = model.predict(experience_reshape)
    return prediction[0]

experience = float(input("Enter your experience in months: "))
print(f"The expected salary is {predict_salary(experience):.3f}$")