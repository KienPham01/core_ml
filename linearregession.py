import pandas as pd
import  matplotlib.pyplot as plt

dataframe =  pd.read_csv('Advertising.csv')
x =  dataframe.values[:,2]
y = dataframe.values[:,4]
# print(y)

plt.scatter(x,y, marker='o')
# plt.show()

def predict(new_radio,weight,bias):
    return weight*new_radio + bias

def cost_funtion(x,y,weight,bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i]-(weight*x[i]+bias))**2
    return sum_error/n


def update_weight(x,y,weight,bias,learning_rate):
    n = len(x)
    weight_temporary = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temporary += -2*x[i]*(y[i] - (x[i]*weight+bias))
        bias_temp += -2*(y[i] - (x[i]*weight+bias))

    weight -=  (weight_temporary/n)*learning_rate
    bias -= (bias/n)*learning_rate

    return weight,bias

def train(x,y,weight,bias,learning_rate,inter):
    cost_history = []
    for i in range(inter):
        weight,bias = update_weight(x,y,weight,bias,learning_rate)
        cost =  cost_funtion(x,y,weight,bias)
        cost_history.append(cost)

    return  weight,bias,cost_history

weight,bias,cost =  train(x,y,0.03,0.0014,0.001,60)

print("result is")
print(weight)
print(bias)
print(cost)

print('predict is:')

print(predict(19,weight,bias))

numberofiterations = [i for i in range(60)]

plt.plot(cost,numberofiterations)
plt.show()


