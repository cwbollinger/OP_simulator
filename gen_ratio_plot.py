import matplotlib.pyplot as plt

data = {1: 0.5, 2: 0.4, 3: 0.4, 4: 0.6, 5: 0.6}

x_data = [1, 2, 3, 4, 5]
y_data = [0.5, 0.4, 0.4, 0.6, 0.6]
plt.bar(x_data, y_data)
plt.title('Successful plan generation for varying constraints')
plt.xlabel('Ratio of "free" nodes to time windowed nodes')
plt.ylabel('Fraction of successfully generated plans')
plt.show()
