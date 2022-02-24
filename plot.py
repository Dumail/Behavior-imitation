#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2021/10/20 上午5:36
# @Author  : Chaofan Pan
# @Email   : pan.chaofan@foxmail.com
# @File    : plot.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
x_index = ['100', '200', '500', '1000', '2000', '3000']
bot1 = np.arange(6) / 10
plt.plot(x, bot1, color='green', label='bot1')
plt.plot(x, bot1 + 0.1, color='red', label='bot2')
plt.plot(x, bot1 - 0.1, color='blue', label='bot3')
plt.plot(x, bot1 + 0.2, color='gray', label='bot4')
plt.plot(x, bot1 - 0.2, color='yellow', label='bot5')
plt.text(2.5, 0.25, 'TODO', fontsize=30)
plt.legend()
plt.xlabel("Number of gameplay records")
plt.ylabel("Actions similarity")
plt.xticks(x, x_index)
plt.show()
