from lib2to3.pgen2.token import EQUAL
from tkinter import Variable
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

curveNumber = 1

#Read the labeled priceing curves
priceCurves = pd.read_csv('..\..\Output\TestingResults.txt', header=None)
priceCurves = np.array(priceCurves)

#Remove all of the non-abnormal price curves
priceCurves = priceCurves[(priceCurves[:,24] == 1)]
#Remove the labeling
priceCurves = np.delete(priceCurves, 24, 1)

#Read the user task inputs
userTasks = pd.read_excel('..\..\Input\COMP3217CW2Input.xlsx', sheet_name='User & Task ID',
    dtype={'User & Task ID': str, 'Ready Time': int, 'Deadline':int, 'Maximum scheduled energy per hour':int, 'Energy Demand':int})

# Define the model
model = LpProblem(name="scheduling-allocation", sense=LpMinimize)

allVariables = []
objectiveFunc = []
#Create linear constraints
for index, task in userTasks.iterrows():
    variables = []
    for hour in range(0, task["Ready Time"]):
        variables.append(LpVariable(name=(task["User & Task ID"] + "_hour" + str(hour)), 
            lowBound= 0, upBound = 0, cat='Integer'))


    for hour in range(task["Ready Time"], task["Deadline"] + 1):
        variable = LpVariable(name=(task["User & Task ID"] + "_hour" + str(hour)), 
            lowBound= 0, upBound = task["Maximum scheduled energy per hour"], cat='Integer')
        objectiveFunc.append(priceCurves[0][hour] * variable)
        variables.append(variable)

    model += (lpSum(variables[task["Ready Time"] : task["Deadline"] + 1]) == task["Energy Demand"], task["User & Task ID"])

    for hour in range(task["Deadline"] + 1, 24):
        variables.append(LpVariable(name=(task["User & Task ID"] + "_hour" + str(hour)), lowBound= 0, upBound = 0, cat='Integer'))

    allVariables.append(variables)

model += lpSum(objectiveFunc)

# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

#for var in [item for sublist in allVariables for item in sublist]:
#    print(f"{var.name}: {var.value()}")

#for name, constraint in model.constraints.items():
#    print(f"{name}: {constraint.value()}")

hours = [str(x) for x in range(0, 24)]
pos = np.arange(len(hours))
users = ['user1', 'user2', 'user3', 'user4', 'user5']
color_list = ['plum','lightpink','lightsteelblue','paleturquoise','lightskyblue']
tallestHieghts = np.full(24, 0)

def labelTasks(barPlot, label):
    for i,rect in enumerate(barPlot):
        if rect.get_height() != 0:
            height = rect.get_height()
            plt.gca().text(rect.get_x() + 0.05 + rect.get_width()/2., tallestHieghts[i] + 0.1,
                    label, ha='center', va='bottom', rotation=90)

for tasks in allVariables:
    heights = []
    for task in tasks:
        if task.value() == None:
            heights.append(0)
        else:
            heights.append(int(task.value()))
    
    bar = plt.bar(pos, heights, color=color_list[int(tasks[0].name[4]) - 1], edgecolor='black',bottom=tallestHieghts)
    labelTasks(bar, tasks[0].name.split('_')[1][4:6])

    tallestHieghts += np.array(heights)

plt.xticks(pos, hours)
plt.xlabel('Hour Of The Day')
plt.ylabel('Energy Usage (kWH)')
plt.title('Optimal Energy Scheduling For Curve %i'%curveNumber)
plt.gca().set_facecolor('lavender')
plt.legend([Line2D([0], [0], color=color_list[0], lw=4),
            Line2D([0], [0], color=color_list[1], lw=4),
            Line2D([0], [0], color=color_list[2], lw=4),
            Line2D([0], [0], color=color_list[3], lw=4),
            Line2D([0], [0], color=color_list[4], lw=4)],
            ["User 1", "User 2", "User 3", "User 4", "User 5"], loc=0)
#plt.gca().set_facecolor('gainsboro')
plt.show()
