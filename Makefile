PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(PIP) install -r requirements.txt 
 $(PYTHON) ./Src/Classifier/Classifier.py
 $(PYTHON) ./Src/Task\ Planer/TaskSolver.py