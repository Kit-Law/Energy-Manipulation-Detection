from lib2to3.pgen2.token import EQUAL
from tkinter import Variable
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#findOptimalSchedule crates a linear program to find the most optimal solution
#for what time each user's tasks should be run, acording to the early time,
#late time, requierd energy for the task and the cost of energy at each hour
def findOptimalSchedule(priceCurves, userTasks, curveNumber):
    # Define the lp model (minimize to find the ceapest energy cost)
    model = LpProblem(name="scheduling-allocation", sense=LpMinimize)

    allVariables = []
    objectiveFunc = []
    #Create linear constraints
    for index, task in userTasks.iterrows():
        variables = []
        
        for hour in range(task["Ready Time"], task["Deadline"] + 1):
            #Define variables for all possible running times for the task
            variable = LpVariable(name=(task["User & Task ID"] + "_hour" + str(hour)), 
                lowBound= 0, upBound = task["Maximum scheduled energy per hour"], cat='Integer')
            #Add the cost of this hour of the task to the objective function buffer
            objectiveFunc.append(priceCurves[hour] * variable)
            variables.append(variable)

        #Add the constraint to make the sum of the variables equal the energy demand
        model += (lpSum(variables) == task["Energy Demand"], task["User & Task ID"])
        allVariables.append(variables)

    #Add the objective function
    model += lpSum(objectiveFunc)

    # Solve the optimization problem
    status = model.solve()

    # Get the results
    print(f"status: {model.status}, {LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")

    #Commented code shows the varible values
    #for var in model.variables():
    #    if var.value() != 0.0:
    #        print(f"{var.name}: {var.value()}")

    #for var in [item for sublist in allVariables for item in sublist]:
    #    print(f"{var.name}: {var.value()}")

    #for name, constraint in model.constraints.items():
    #    print(f"{name}: {constraint.value()}")

    plotGraph(model, allVariables, curveNumber)

#labelTasks labels each bar with it's task id
def labelTasks(barPlot, tallestHieghts, label):
    for i,rect in enumerate(barPlot):
        if rect.get_height() != 0:
            plt.gca().text(rect.get_x() + 0.05 + rect.get_width()/2., tallestHieghts[i] + 0.1,
                    label, ha='center', va='bottom', rotation=90)

#plotGraph plots the optimial times to run each task for the user
#The graph is a bar chat that has stacked bars, with labels on each
#for the task id.
def plotGraph(model, allVariables, curveNumber):
    #Initilize the metadata for the graph
    hours = [str(x) for x in range(0, 24)]
    pos = np.arange(len(hours))
    color_list = ['plum','lightpink','lightsteelblue','paleturquoise','lightskyblue']
    tallestHieghts = np.full(24, 0)

    #Seperate the non zero values for the varibles into a 2D array for the bar heights
    for tasks in allVariables:
        heights = [0] * 24
        for i, task in enumerate(tasks):
            if task.value() != None:
                heights[i] = int(task.value())
        
        #Plot the task heights along with task id label
        bar = plt.bar(pos, heights, color=color_list[int(tasks[0].name[4]) - 1], edgecolor='black',bottom=tallestHieghts)
        labelTasks(bar, tallestHieghts, tasks[0].name.split('_')[1][4:6])

        tallestHieghts += np.array(heights)

    #Disply settings for the graph
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

    plt.savefig('..\..\Output\Plots\\'+str(curveNumber)+'.png')
    plt.clf()

#Main Program

#Read the labeled priceing curves
priceCurves = pd.read_csv('..\..\Output\TestingResults.txt', header=None)
priceCurves = np.array(priceCurves)

#Read the user task inputs
userTasks = pd.read_excel('..\..\Input\COMP3217CW2Input.xlsx', sheet_name='User & Task ID',
    dtype={'User & Task ID': str, 'Ready Time': int, 'Deadline':int, 'Maximum scheduled energy per hour':int, 'Energy Demand':int})

#Find the optimal solution for each curve
for curveNumber in range(len(priceCurves)):
    if priceCurves[curveNumber][24] == 1:
        findOptimalSchedule(priceCurves[curveNumber][:-1], userTasks, curveNumber)