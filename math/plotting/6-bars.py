#!/usr/bin/env python3
"""Module that plots a stacked bar graph of fruit per person."""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]
width = 0.5

plt.bar(names, apples, width, color='red', label='apples')
plt.bar(names, bananas, width, bottom=apples, color='yellow',
        label='bananas')
plt.bar(names, oranges, width, bottom=apples + bananas, color='#ff8000',
        label='oranges')
plt.bar(names, peaches, width, bottom=apples + bananas + oranges,
        color='#ffe5b4', label='peaches')

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(range(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
