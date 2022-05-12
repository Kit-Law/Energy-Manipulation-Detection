import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os

dirname = os.path.dirname(__file__)

#Initilize the metadata for the graph
titles = [
    "Normal Curves",
    "Abnormal Curves",
    "Testing Curves",
    "Training Vs Testing",
    "Normal Vs Abnormal Vs Testing"
]

color_list = ['plum','lightpink','lightsteelblue','paleturquoise','lightskyblue']
hours = [str(x) for x in range(0, 24)]

def plotGraph(data, i):
    #Disply settings for the graph
    plt.xlabel('Hour Of The Day')
    plt.ylabel('Energy Usage (kWH)')
    plt.title(titles[i])
    plt.gca().set_facecolor('lavender')
    if len(data) == 3:
        plt.legend([Line2D([0], [0], color=color_list[0], lw=4),
                    Line2D([0], [0], color=color_list[1], lw=4),
                    Line2D([0], [0], color=color_list[2], lw=4)],
                    ["Normal", "Abnormal", "Testing"], loc=0)
        
        #Plot the data
        if i == 3:
            for j,plot in enumerate(data):
                for curve in plot:
                    plt.plot(hours, curve, color=color_list[j], linestyle=":")
        else:
            for j,plot in enumerate(data):
                for k in range(10):
                    plt.plot(hours, plot[k], color=color_list[j])
    else:
        #Plot the data
        for curve in data:
            plt.plot(hours, curve, color=color_list[i])

    #Save the plot
    plt.savefig(os.path.join(dirname, ('../../Data/Plots/'+str(titles[i])+'.png')), dpi=1000)
    plt.clf()

#Read the training data
data = pd.read_csv(os.path.join(dirname, '../../Data/TrainingData.txt'), header=None)
data = np.array(data)
#Split the traning data into the two classes, then combine the data so classes are in the order {0,1,0,1,...,0,1}
noraml, abnormal = [data[(data[:,24]==0)], data[(data[:,24]==1)]]

#Drop classifer labels
noraml = np.delete(noraml, 24, 1)
abnormal = np.delete(abnormal, 24, 1)

#Read the testing data
testing = pd.read_csv(os.path.join(dirname, '../../Data/TestingData.txt'), header=None)
testing = np.array(testing)

#Plot the graphs
plotGraph(noraml, 0)
plotGraph(abnormal, 1)
plotGraph(testing, 2)
plotGraph([noraml, abnormal, testing], 3)
plotGraph([noraml, abnormal, testing], 4)