from cvxstoc import NormalRandomVariable
from cvxstoc.expectation import expectation
from cvxstoc import prob
from cvxpy import Maximize, Problem
import cvxpy as c
from cvxpy.expressions.variable import Variable
import numpy as np

# Create problem data.
n = 10
mu = np.arange(n)
Sigma = np.arange(n)[::-1] * np.eye(n)
p = NormalRandomVariable(mu, Sigma)
alpha = -1
beta = 0.05

# Create and solve stochastic optimization problem.
x = Variable(n)

constraints = []

constraints.append(x >= 0)
constraints.append(c.sum(x) == 1)
constraints.append(prob(x.T * p <= alpha, num_samples=100) <= beta)

prob = Problem(Maximize(expectation(x.T * p, num_samples=100)), constraints)
prob.solve()

if np.any(x.value is None):
    print('Solver failed with status: {}'.format(prob.status))

# print('----- Constraints -----')
# for constraint in constraints:
#     print(constraint)

print('----- Solution -----')
print('x: {}'.format(x.value))
print('expected value of p: {}'.format(p.value))
