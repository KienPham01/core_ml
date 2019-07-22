import pandas  as pd
import  numpy as np
import  matplotlib.pyplot as plt

data = pd.read_csv('data_classification.csv',header=None)

print(data.values)

true_x = []
true_y = []
false_x = []
false_y = []



for item in data.values:
    if item[2] == 1.:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])

plt.scatter(true_x,true_y,marker='o',c='b')
plt.scatter(false_x,false_y,marker='s',c='r')

# plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def devided(p):
    if p >= 0.5:
        return 1
    else:
        return 0
def predict(features,weight):
    z = np.dot(features,weight)
    return  sigmoid(z)

def cost_function(features,labels,weights):

    """
    :param features:(100 * 3)
    :param labels:(100*1)
    :param weights:(3*1)
    :return:
    """

    n = len(labels)
    prediction = predict(features,weights)

    cost_class1 = -labels*np.log(prediction)
    cost_class2 = -(1-labels)*np.log(1-prediction)

    cost = cost_class1 + cost_class2

    return  cost.sum()/n



