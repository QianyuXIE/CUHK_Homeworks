'''
DDA 5002 Optimization Fall 2025
Homework 1, Problem 2

Author: Qianyu Xie
Email: 225040249@link.cuhk.edu.cn
2025/9/15 21:04
'''

'''
The problem to solve:

    Mainimize:

    Subject to:

    Where:

'''

import coptpy as cp
from coptpy import COPT

# Create COPT environment
env = cp.Envr()

# Create COPT model
model = env.createModel("Problem2") 

# Add variables: 
distance = [60, 55, 75, 80, 64]
power = [10, 8, 13, 15, 9]
m = model.addVar(lb=0, name="m")
b = model.addVar(lb=0, name="b")

# Add constraints:
model.addConstr()

# Set objective function:
model.setObjective(
    (for i in range (5):
        
), sense=COPT.MINIMIZE
)