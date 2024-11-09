
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('https://raw.githubusercontent.com/Simeon-Dhinakaran/AI-and-ML/refs/heads/main/Budget_For_Advertising.csv')
inputx = dataset.iloc[:, 0:3].values
outputy = dataset.iloc[:, 3].values


input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/4, random_state = 0)


model = LinearRegression()
print("\nThe parameters of the model are\n\n",model.get_params())
print("\nThe model we are using is ", model.fit(input_train, output_train))

from operator import ne

tvBudget = float(input("\nGive the TV Budget ($)  "))
radioBudget = float(input("\nGive the Radio Budget ($)  "))
newspaperBudget = float(input("\nGive the Newspaper Budget ($)  "))
testinput = [[tvBudget,radioBudget,newspaperBudget]]
predicted_output = model.predict(testinput)
print('\nThe test input is as follows ',testinput) 
print('\nThe predicted sales is ',predicted_output) 
yes = input("\nCan I proceed\n")


print("\nThe test input (TV Budget, Radio Budget and Newspaper Budget) is as follows \n")
print(input_test)

predicted_output = model.predict(input_test)
print("\nThe predicted sales for the test input is as follows \n")
print(predicted_output)
