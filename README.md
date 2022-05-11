# Energy-Manipulation-Detection

This project schedules the best times to run tasks depending on a given energy price curve.
This is the coursework for COMP3217, titled: Detection of Manipulated Pricing in Smart 
Energy CPS Scheduling

## Linux

On linux the run.sh bash script can be run to preform the following steps.

## Setup

requierments.txt contains a lit of python modules that can be installed with the 
following command:

```BASH
pip install -r requirements.txt 
```

### Requirements:
- numpy
- sklearn
- pandas
- pulp
- matplotlib
- openpyxl

## Execution

To label the testing data found in 'Data/TestingData.txt' run the classifier 
script like so:

```BASH
python "./Src/Classifier/Classifier.py"
```

Now the testing data has been labeled and stored at 'Output/TestingResults.txt' an 
optimal solutions for each task set can be found in 'Input/COMP3217CW2Input.xlsx'
by runing the TaskSolver script like so:

```BASH
python "./Src/Task Planer/TaskSolver.py"
```

## Testing

To compare the acuraccy of each classifier the ClassifierComparison scrpit
can be run like so:

```BASH
python "./Src/Classifier/ClassifierComparison.py"
```