# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:04:49 2020

Plot the results from run.py

@author: Ningyu Wang
"""

import numpy as np
import matplotlib.pyplot as plt

# path 
path = 'E:/20201111HeleShaw/'
# list of path
path_list = [
    'N35-4',
    'N45-4',
    'N45-6',
    'N45-8',
    'N52-4',
    'N52-6',
    'N52-8'
    ]

# list of x coordinate of ellipse center
x_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]
# list of y coordinate of ellipse center
y_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]
# list of semi long axis
a_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]

# list of semi short axis
b_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]

# list of tilt angle
angle_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]

# list of eccentricity
e_list = [
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    np.array([]),
    ]


fig_x = plt.figure()
plt.title('x')
fig_y = plt.figure()
plt.title('y')
fig_a = plt.figure()
plt.title('a')
fig_b = plt.figure()
plt.title('b')
fig_angle = plt.figure()
plt.title('angle')
fig_e = plt.figure()
plt.title('e')

for ite0 in range(len(path_list)):
    readname = path + path_list[ite0] + '_x.txt'
    x_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_x.number)
    plt.plot(x_list[ite0],label=path_list[ite0])
    
    readname =  path + path_list[ite0]+'_y.txt'
    y_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_y.number)
    plt.plot(y_list[ite0],label=path_list[ite0])
    
    readname =  path + path_list[ite0]+'_a.txt'
    a_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_a.number)
    plt.plot(a_list[ite0],label=path_list[ite0])
    
    readname =  path + path_list[ite0]+'_b.txt'
    b_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_b.number)
    plt.plot(b_list[ite0],label=path_list[ite0])
    
    readname =  path + path_list[ite0]+'_angle.txt'
    angle_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_angle.number)
    plt.plot(angle_list[ite0],label=path_list[ite0])
    
    readname =  path + path_list[ite0]+'_e.txt'
    e_list[ite0] = np.loadtxt(readname)
    plt.figure(fig_e.number)
    plt.plot(e_list[ite0],label=path_list[ite0])
    