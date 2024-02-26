#!/usr/bin/env python3
""" plot a stacked bar chart for the amount and types of fruit people have """
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# Colors
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
persons = ['Farrah', 'Fred', 'Felicia']

fig, ax = plt.subplots()
width = 0.5
bottom = np.zeros(len(persons))
for i in range(len(fruits)):
    ax.bar(persons, fruit[i], width, bottom=bottom, label=fruits[i], color=colors[i])
    bottom += fruit[i]

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()

plt.show()