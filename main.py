# -*- coding: utf-8 -*-
"""
This program is to find the curvature radii along the oil blob perimeter.
It studies only the largest oil blob.
If there is a certain oil blob that you want to track, please cut the image properly.

Created on Wed Nov 11 2020

@author: ningyu
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# This implies a code from internet. Reference see inside.
from findCurvatureRadii2 import findCurvatureR

# show image
from imshow import imshow

# Determine if the points on the contour are in clockwise or counter- order.
def isCounterClockwise(points,step=3,FLAG_GLOBAL=True):
    # counter-clockwise
    FLAG_CLDIR = 1;
    if FLAG_GLOBAL:
        # center of contour
        cx = np.mean(points[:,0])
        cy = np.mean(points[:,1])

        # phase angle of points
        pa1 = atan(points[0,0],points[0,1],cx,cy)
        pa2 = atan(points[step,0],points[step,1],cx,cy)

        if pa1 > pa2:
            # clockwise
            FLAG_CLDIR = -1;
    else:
        m,n = points.shape;
        hm = int(m/2)
        r1 = points[hm,:] - points[0,:]
        r2 = points[-1,:] - points[hm,:]

        if np.cross(r1,r2) < 0:
            FLAG_CLDIR = -1;

    return(FLAG_CLDIR)

# Remove straight section. Straight lines have infinite curvature radius.
def removeConcave(points,tol=1e-13):
    m,n = points.shape;

    # Check each three adjacent points from the last in the list.
    # If it is a concaving section, remove the middle point.
    # In this way, when a point is removed, the two remaining points from the
    # last step enters the next iteration.

    FLAG_CLDIR = isCounterClockwise(points);

    # remove the concave points
    for ite in range(m-1,1,-1):
        r1 = points[ite-2,:] - points[ite-1,:]
        r2 = points[ite-1,:] - points[ite,:]

        # Use cross product to determine if the vector between points
        # is rotating in the clockwise direction or the opposite.
        # If it is different from the overall direction,
        # remove the middle point.
        if np.cross(r1,r2) * FLAG_CLDIR < 0:
            #print(ite,r1,r2)
            points = np.delete(points,ite-1,0)


    # At last, check the two sections containing the first and last points.
    r1 = points[1,:] - points[0,:]
    r2 = points[0,:] - points[-1,:]
    if 1-abs( np.dot(r1,r2) / np.linalg.norm(r1) / np.linalg.norm(r2) ) < tol:
        points = np.delete(points,0,0)
    r1 = points[0,:] - points[-1,:]
    r2 = points[-1,:] - points[-2,:]
    if 1-abs( np.dot(r1,r2) / np.linalg.norm(r1) / np.linalg.norm(r2) ) < tol:
        points = np.delete(points,-1,0)

    return(points)


# Subsample the regions of high point density
def subSample(points,distancetol=5):
    m,n = points.shape;
    '''
    # if three points are adjacent, remove the middle point
    for ite in range(m-1,1,-1):
        r = points[ite,:] - points[ite-2,:]
        if np.linalg.norm(r,2) < distancetol:
            np.delete(points,ite-1,0)
    '''
    # if two points are adjacent, remove the earlier point
    for ite in range(m-1,0,-1):
        r = np.linalg.norm(points[ite,:] - points[ite-1,:],2)
        #print(r)
        if r < distancetol:
            #print('delete',ite)
            points = np.delete(points,ite-1,0)

    return(points)



# Interpolate the contour for long straight sections
def interpolateContour(points,insert=1,distancetol=5):
    points = np.asarray(points)
    n = len(points[:,0])
    '''
    # first, deal with the gap between the first and last points
    d = np.linalg.norm(points[0,:]-points[-1,:]);
    if d > distancetol:
        a = points[-1,:];
        b = points[0,:]
        for j in range(insert,0,-1):
            x = a[0] * (insert-j) / insert + b[0] * j / insert
            y = a[1] * (insert-j) / insert + b[1] * j / insert
            points = np.insert(points,n,[x,y],axis=0)
    '''

    # then, deal with other points
    for i in range(n-1,0,-1):
        d = np.linalg.norm(points[i,:]-points[i-1,:]);
        if d > distancetol:
            a = points[i-1,:]
            b = points[i,:]
            for j in range(insert,0,-1):
                x = a[0] * (insert-j) / insert + b[0] * j / insert
                y = a[1] * (insert-j) / insert + b[1] * j / insert
                points = np.insert(points,i,[x,y],axis=0)
    return(points)



# Get the oil blob perimeter
# Returns the grayscale image and a 2D array of xy coordinates
# of points on the contour.
def findOilBlob(filename,threshold=127,color='gray',iterations=[2,8,6],showresult=False):
    #filename = "test.jpg"
    print("Reading image : ", filename)
    imReference = cv2.imread(filename, cv2.IMREAD_COLOR)

    '''
    # filter image
    imReference = cv2.bilateralFilter(imReference,5,40,40)
    '''


    # convert to grayscale
    if color in ['blue','BLUE','Blue','B','b']:
        imgray = imReference[:,:,0]
    elif color in ['green','GREEN','Green','G','g']:
        imgray = imReference[:,:,1]
    elif color in ['red','RED','Red','R','r']:
        imgray = imReference[:,:,2]
    else:
        imgray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    imgrayorig = imgray.copy()
    #imgSPLIT = cv2.split(imReference);
    #imgray = imgSPLIT[1];
    imshow(imgray,showresult,name='grayscale')

    # filter image
    imgray = cv2.bilateralFilter(imgray,5,40,40)


    # cut image edges
    imReference = imReference[2:-2,2:-2,:]


    # denoise
    imgray = cv2.fastNlMeansDenoising(imgray,None,10,7,21);

    imshow(imgray,showresult,name='denoise')

    # threshold
    ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_OTSU)
    imshow(thresh,showresult,name='threshold')


    # erode and dilate
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = iterations[0])
    dilation = cv2.dilate(erosion,kernel,iterations = iterations[1])
    imshow(dilation,showresult,name='dilation');
    erosion = cv2.erode(dilation,kernel,iterations = iterations[2])
    imshow(erosion,showresult,name='erosion');



    # get the contours
    '''
    # The cv2.findContours() function removed the first output in a newer version.
    tmpim, contours, hierarchy = cv2.findContours(thresh, method=cv2.RETR_TREE, \
                                              mode=cv2.CHAIN_APPROX_SIMPLE)
    '''
    contours, hierarchy = cv2.findContours(erosion, method=cv2.RETR_TREE, \
                                       mode=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgray, contours, contourIdx = -1, color=(255,0,0), \
                     thickness=4)
    imshow(imgray,showresult,name='contours');

    # First, the real interfaces have a lot of points.
    # Abandon the contours with only several points.
    # Keep only the longest contour.
    contours = sorted(contours,key=len,reverse=True);
    contoursorig = np.copy(contours)
    contours = contours[0];


    return (imgray,contours,imgrayorig,contoursorig)

# Function of a circle
def funcCircle(para,x,y):
    return( (x-para[0])**2 + (y-para[1])**2 - para[2]**2)
    #return( np.sqrt((x-para[0])**2 + (y-para[1])**2) - para[2])


# from a set of x y values, fit a circle
def findCircle(points):
    try:
        x = points[:,0];
        y = points[:,1];
        x_guess = (max(x)+min(x))/2;
        y_guess = (max(y)+min(y))/2;
        R_guess = ( max(y)-min(y) ) / 2;
        guess = np.array([x_guess,y_guess,R_guess]);
        #print(guess)
        lower_bounds = [min(x),min(y),0]
        upper_bounds = [max(x),max(y),max(max(x),max(y))/2]
        bounds = (lower_bounds,upper_bounds)
    except:
        raise Exception(points.shape)
    return(optimize.least_squares(fun=funcCircle,x0=guess, \
                                  xtol = 1e-12,bounds=bounds,args=(x,y)).x)

# find Curvature
# return x,y,r of the inscribed circle at each point
# NW modify on May22-2020 to locate the window around the point
# instead of after it.
def findCurvatureRadius(points,window=5):
    # number of points
    N = points.shape[0];

    # half window length used to fit the circle
    n = int(window/2);

    # counter-clockwise direction of the points
    FLAG_CLDIR = isCounterClockwise(points)

    #print(N)
    # fitted circle
    result = np.zeros([N,3]);

    try:
        for i in range(N):
            if i < n+1:
                indx = np.concatenate((np.arange(N+i-n,N), np.arange(0,i+n+1)))
            elif i + n >= N:
                indx = np.concatenate((np.arange(i-n,N), np.arange(0,i+n-N)))
            else:
                indx = np.arange((i-n),(i+n+1))

            #result[i,:] = findCircle(points[indx,:]);
            x = points[indx,0]
            y = points[indx,1]
            result[i,:] = findCurvatureR(x,y)

            # find the local clock direction
            FLAG_CLDIRindx = isCounterClockwise(points[indx,:],step=min(N,3),FLAG_GLOBAL=False)
            # If the local clock direction is different from the global value
            # return negative curvature radius.
            result[i,:] *= FLAG_CLDIRindx * FLAG_CLDIR;

    except:
        print('Error finding curvature radius at point ', i)
        '''
        print(i,n,N)
        print(N+i-n,i+n+1)
        print(i-n,i+n-N)
        '''

    return( result )



# Function of a ellipse
# 
# para, list:
#   0: x_c
#   1: y_c    
#   2: a
#   3: b
#   4: tilt angle
def funcEllipse(para,x,y):
    #x_hat = (x-para[0])*np.cos(para[4]) - (y-para[1])*np.sin(para[4])
    #y_hat = (x-para[0])*np.sin(para[4]) + (y-para[1])*np.cos(para[4])
    x_hat =  (x-para[0])*np.cos(para[4]) + (y-para[1])*np.sin(para[4])
    y_hat = -(x-para[0])*np.sin(para[4]) + (y-para[1])*np.cos(para[4])
    return( (x_hat/para[2])**2 + (y_hat/para[3])**2 - 1 )
    #return( np.sqrt((x-para[0])**2 + (y-para[1])**2) - para[2])

'''
# from a set of x y values, fit a ellipse
# nonlinear least square fit

input:
    points: n x 2, np.array
output:
        : list:
            x0:     x coordinate of ellipse
            y0:     y coordinate of ellipse
            a:      semi-major axis
            b:      semi-minor axis
            theta:  tilt angle
'''
def findEllipse(points):
    try:
        x = points[:,0];
        y = points[:,1];
        x_guess = (max(x)+min(x))/2;
        y_guess = (max(y)+min(y))/2;
        R_guess = ( max(y)-min(y) ) / 2;
        guess = np.array([x_guess,y_guess,R_guess,R_guess,0]);
        #print(guess)
        lower_bounds = [min(x),min(y),0,0,-2*np.pi]
        upper_bounds = [max(x),max(y),max(max(x),max(y)),max(max(x),max(y)),2*np.pi]
        bounds = (lower_bounds,upper_bounds)
    except:
        raise Exception(points.shape)
    return(optimize.least_squares(fun=funcEllipse,x0=guess, \
                                  xtol = 1e-15,bounds=bounds,args=(x,y)).x)


'''
# from a set of x y values, fit a ellipse
# least square fit
# Sr = f(x,y) = Ax^2 + Cy^2 + Bxy + Dx + Ey + 1 = 0

input:
    points: n x 2, np.array
output:
        : list:
            x0:     x coordinate of ellipse
            y0:     y coordinate of ellipse
            a:      semi-major axis
            b:      semi-minor axis
            theta:  tilt angle
'''
def findEllipse2(points):
    x2 = points[:,0]**2
    x3 = points[:,0]**3
    x4 = points[:,0]**4
    y2 = points[:,1]**2
    y3 = points[:,1]**3
    y4 = points[:,1]**4
    xy = points[:,0] * points[:,1]
    x2y2 = xy**2
    x2y = x2 * points[:,1]
    xy2 = y2 * points[:,0]
    x3y = x3 * points[:,1]
    xy3 = y3 * points[:,0]
    
    
    x2 = np.sum(x2)
    x3 = np.sum(x3)
    x4 = np.sum(x4)
    y2 = np.sum(y2)
    y3 = np.sum(y3)
    y4 = np.sum(y4)
    xy = np.sum(xy)
    x2y2 = np.sum(x2y2)
    x2y = np.sum(x2y)
    xy2 = np.sum(xy2)
    x3y = np.sum(x3y)
    xy3 = np.sum(xy3)
    
    
    # sequence
    # x2,y2,xy,x,y
    Amatrix = np.array([[x4,   x2y2,x3y, x3, x2y], \
                        [x2y2, y4,  xy3, xy2,y3], \
                        [x3y,  xy3, x2y2,x2y,xy2], \
                        [x3,   xy2, x2y, x2, xy], \
                        [x2y,  y3,  xy2, xy, y2]])
    bvec = np.array([-x2,-y2,-xy,-np.sum(points[:,0]),-np.sum(points[1])])
    
    # get A,C,B,D,E
    res = np.linalg.solve(Amatrix,bvec)
    
    print(res)
    
    # from wikipedia: https://en.wikipedia.org/wiki/Ellipse
    # notice the definition of B and C
    A = res[0]
    C = res[1]
    B = res[2]
    D = res[3]
    E = res[4]
    
    
    
    x0 = ( 2*C*D - B*E ) / ( B**2 - 4*A*C )
    y0 = ( 2*A*E - B*D ) / ( B**2 - 4*A*C )
    if B == 0:
        if A <= C:
            theta = 0;
        else:
            theta = np.pi/2;
    else:
        theta = atan(1, ( C-A - np.sqrt( (A-C)**2 + B**2) ) / B )
    
    tmpab = 2 * ( A*E**2 + C*D**2 - B*D*E + (B**2-4*A*C)*1 )
    
    print(tmpab)
    
    a = -np.sqrt( tmpab * ( (A+C) + np.sqrt( (A-C)**2 + B**2 )) ) / (B**2 - 4*A*C)
    print((A+C) - np.sqrt( (A-C)**2 + B**2 )) 
    b = -np.sqrt( tmpab * ( (A+C) - np.sqrt( (A-C)**2 + B**2 )) ) / (B**2 - 4*A*C)
    return(x0,y0,a,b,theta,A,C,B,D,E)



# generate a circle
def generateCircle(xyr,n=100):
    theta = np.linspace(0,2*np.pi,n);
    x = xyr[0] + xyr[2] * np.cos(theta)
    y = xyr[1] + xyr[2] * np.sin(theta)
    return(x,y)

# compute the angle (arctangent) from xy coordinates
def atan(x,y,x0=0,y0=0):
    x = x-x0;
    y = y-y0;
    try:
        n = len(x);
        if n != len(y):
            raise('The length of x and length of y should be the same!')

        theta = np.zeros(n);
        for i in range(n):
            if x[i] == 0 and y[i] == 0:
                theta[i] = 0
            elif x[i] == 0 and y[i] > 0:
                theta[i] = np.pi/2
            elif x[i] == 0 and y[i] < 0:
                theta[i] = np.pi*3/2
            elif y[i] == 0 and x[i] > 0:
                theta[i] = 0
            elif y[i] == 0 and x[i] < 0:
                theta[i] = np.pi

            elif x[i] >= 0 and y[i] >= 0:
                theta[i] = np.arctan(y[i]/x[i]);
            elif x[i] >= 0 and y[i] < 0:
                theta[i] = 2 * np.pi - np.arctan(abs(y[i]/x[i]));
            elif x[i] < 0 and y[i] >= 0:
                theta[i] = np.pi - np.arctan(abs(y[i]/x[i]))
            elif x[i] < 0 and y[i] < 0:
                theta[i] = np.pi + np.arctan(abs(y[i]/x[i]))

    except TypeError:
        if x == 0 and y == 0:
            theta = 0
        elif x == 0 and y > 0:
            theta = np.pi/2
        elif x == 0 and y < 0:
            theta = np.pi*3/2
        elif y == 0 and x > 0:
            theta = 0
        elif y == 0 and x < 0:
            theta = np.pi

        elif x >= 0 and y >= 0:
            theta = np.arctan(y/x);
        elif x >= 0 and y < 0:
            theta = 2 * np.pi - np.arctan(abs(y/x));
        elif x < 0 and y >= 0:
            theta = np.pi - np.arctan(abs(y/x))
        elif x < 0 and y < 0:
            theta = np.pi + np.arctan(abs(y/x))

    return(theta)


# low-pass filter
from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# averaging filter
def avgFilter(data,window):
    data1 = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window-1:
            data1[i] = (sum(data[(i-window+1):])+sum(data[0:i+1]))/window
            #print(i)
            #print(data[(i-window+1):],data[0:i+1])

        else:
            data1[i] = np.mean(data[(i-window):(i)])
            #print(i-window,i)
            #print(data[i-window:i])
    return(data1)






# test
def test1():
    #workpath = '../test';
    workpath = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_1fps';
    #filename = 'ts3_0000010081.tif';
    #filename = 'ts3_0000012692.tif';
    #filename = 'ts3_0000012718.tif';
    #filename = 'ts3_0000010072.tif';
    #filename = 'ts3_0000010158.tif';
    #filename = 'ts3_0000012622.tif';
    filename = 'ts3_001005.tif';
    thre = 127;
    iteration = [1,5,4]
    showresult = False;


    # average filter:
    # average this number of points to generate a point
    freqratio = 3;

    # number of points used to fit the circle
    #window = [18,20,25]
    window = [20]
    #window = [15]


    filename = workpath +"/" + filename
    cha1,contours,__ = findOilBlob(filename,iterations=iteration,showresult=showresult);
    #cv2.drawContours(cha1, contours, contourIdx = -1, color=(255,0,0), \
    #                 thickness=4)
    #imshow(cha1,showresult)

    # shrink unwanted dimension
    oilcontour = contours[:,0,:].astype(float);

    oilcontour0 = np.copy(oilcontour)

    # save a copy of the original image
    oilorg = np.copy(oilcontour)

    # interpolate straight section
    #oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=30)
    oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=10)


    plt.figure()
    plt.plot(oilcontour[:,0],oilcontour[:,1],'bs-')
    plt.plot(oilorg[:,0],oilorg[:,1],'r+')


    '''
    # low pass filter
    oilcontour[:,0] = butter_lowpass_filter(oilcontour[:,0], cutoff, fs, order)
    oilcontour[:,1] = butter_lowpass_filter(oilcontour[:,1], cutoff, fs, order)
    '''

    # filter and subsample
    # averaging filter
    oilcontour[:,0] = avgFilter(oilcontour[:,0], int(freqratio))
    oilcontour[:,1] = avgFilter(oilcontour[:,1], int(freqratio))
    # subsample
    ind = np.arange(0,int(len(oilcontour)/freqratio));
    oilcontour = oilcontour[ind*freqratio]


    # plot the oil blob
    plt.figure()
    plt.plot(oilcontour0[:,0],oilcontour0[:,1],'rs',label='oil contour')
    plt.plot(oilcontour[:,0],oilcontour[:,1],'s',label='subsampled oil contour')
    #plt.axis('square')
    #plt.title('oil blob')

    # find the ecliptical fit
    ellps = findEllipse(oilcontour)

    theta = np.linspace(0,2*np.pi,len(oilcontour[:,0]))
    X = ellps[2]*np.cos(theta);
    Y = ellps[3]*np.sin(theta);
    x = ellps[0] + X*np.cos(ellps[4])-Y*np.sin(ellps[4])
    y = ellps[1] + X*np.sin(ellps[4])+Y*np.cos(ellps[4])

    plt.plot(x,y,'r-',label='fitted ellipse')
    plt.axis('square')
    plt.title('oil blob',fontsize=12)
    plt.legend(prop={'size': 12})
    plt.xlabel('pixels',fontsize=12)
    plt.ylabel('pixels',fontsize=12)

    eclpsxy = np.array([x,y]).transpose()


    for lp0 in window:

        curvatureradius = findCurvatureRadius(oilcontour,window=lp0);
        curvatureeclps = findCurvatureRadius(eclpsxy,window=lp0);

        tmpcha = np.copy(cha1)
        for i in range(len(curvatureradius[:,0])):
            #if np.mod(i,10)!=0:
            #    continue
            x,y = generateCircle(curvatureradius[i,:],300)
            try:
                tmpcha[y.astype(int),x.astype(int)] = 0;
            except:
                pass
        plt.imsave(workpath+'/pore'+str(lp0)+'.jpg',tmpcha)

        fig = plt.figure()
        plt.plot(eclpsxy[:,0],eclpsxy[:,1])
        for i in range(len(curvatureeclps[:,0])):
            #if np.mod(i,10)!=0:
            #    continue
            x,y = generateCircle(curvatureeclps[i,:],300)
            try:
                #tmpcha[y.astype(int),x.astype(int)] = 0;
                plt.plot(x,y)
            except:
                pass
        plt.axis('square')
        plt.savefig(workpath+'/ellipse'+str(lp0)+'.jpg')






        # compute the phase angle of each point on the oil-water interface
        theta = atan(oilcontour[:,0],oilcontour[:,1], \
                     (max(oilcontour[:,0])+min(oilcontour[:,0]))/2, \
                     (max(oilcontour[:,1])+min(oilcontour[:,1]))/2)

        # compute the phase angle of each point on the oil-water interface
        thetaeclps = atan(eclpsxy[:,0],eclpsxy[:,1], \
                     (max(eclpsxy[:,0])+min(eclpsxy[:,0]))/2, \
                     (max(eclpsxy[:,1])+min(eclpsxy[:,1]))/2)

        '''
        plt.figure()
        plt.plot(curvatureradius[:,2],'rs-')
        #plt.plot(y,'rs-')
        #plt.plot(yavg,'rs-')
        plt.xlabel('index')
        plt.ylabel('curvature radius')
        plt.title('radius')
        '''

        plt.figure()
        thetaeclps1 = np.roll(thetaeclps,int((freqratio+lp0)/2))
        plt.plot(theta*180/np.pi,curvatureradius[:,2],'rs',label='oil blob')
        plt.plot(thetaeclps1*180/np.pi,curvatureeclps[:,2],'b+',label='fitted ellipse')
        #kappa = 2/((ellps[2]*np.sin(theta+ellps[4]))**2+(ellps[3]*np.sin(theta+ellps[4]))**2)**(3/2)
        R = ((ellps[2]*np.sin(theta+ellps[4]))**2+(ellps[3]*np.cos(theta+ellps[4]))**2)**(3/2) / ellps[2] / ellps[3]
        theta1 = np.copy(theta)
        theta1 = np.roll(theta1,int((freqratio-lp0)/2))
        #plt.plot(theta1*180/np.pi, R,'b-',label='analitycal ellipse')

        plt.xlabel('phase angle')
        plt.ylabel('curvature radius')
        plt.title('radius')
        plt.legend()

        '''
        plt.figure()
        plt.hist(curvatureradius[:,2],bins=10)
        plt.title('radius')
        '''


        plt.figure()
        # semimajor axis coordinate
        x=(oilcontour[:,0]-ellps[0])*np.cos(ellps[4]) - (oilcontour[:,1]-ellps[1])*np.sin(ellps[4]);
        x1= np.roll(x,int((-freqratio+lp0)/2))
        plt.plot(x1,curvatureradius[:,2],'bs-',label='curvature radius')
        plt.plot(x1,theta*180/np.pi,label='phase angle')
        plt.xlabel('semi-major axis coordinate')
        #plt.ylabel()
        plt.legend()

        '''
        plt.figure()
        # semimajor axis coordinate
        x=oilcontour[:,0]*np.cos(ellps[4]) - oilcontour[:,1]*np.sin(ellps[4]);
        plt.loglog((1+((ellps[2]/ellps[3])**2-1)*x**2),curvatureradius[:,2],'bs-',label='curvature radius')
        plt.xlabel('semi-major axis coordinate')
        plt.legend()
        '''

        plt.figure()
        x = (oilcontour[:,0]-ellps[0])*np.cos(ellps[4]) - (oilcontour[:,1]-ellps[1])*np.sin(ellps[4]);
        y = (oilcontour[:,0]-ellps[0])*np.sin(ellps[4]) + (oilcontour[:,1]-ellps[1])*np.cos(ellps[4]);
        plt.plot(x,y,'s-')


if __name__ == '__main__':
    
    '''
    cut the image, use only the left hand side half
    '''
    # working directory
    sourcepath_cut = 'E:/20201111HeleShaw/N45-4';
    sourcepath_cut = 0
    targetpath_cut = sourcepath_cut +'/cut';

    #os.mkdir(targetpath_cut)

    from cut import cutall
    #cutall(x=[0,1000],y=[400,1000],FlagPercent=False,sourcepath=sourcepath_cut,targetpath=targetpath_cut)


    # working directory
    #sourcepath = 'F:/ferrofluid_experiment/postprocessing/noflow_rotateMag/ts3_1fps/cut';
    sourcepath = targetpath_cut
    targetpath = sourcepath +'/../result';

    try:
        os.mkdir(targetpath_cut)
    except FileExistsError:
        pass

    # parameters
    thre = 127;
    iteration = [1,5,4]
    showresult = False;
    dotsize = 3;
    subsampledistance = 9;


    # average filter:
    # average this number of points to generate a point
    freqratio = 2;

    # number of points used to fit the circle
    #window = [19]
    window = [19]

    # load the files in the source directory
    # @Vaibhav
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    from os.path import isfile, join
    files = [f for f in os.listdir(sourcepath) if isfile(join(sourcepath, f))]

    # prepare the output directory
    try:
        os.mkdir(targetpath)
    except FileExistsError:
        pass


    # record the max and min of curvature radius
    maxradiustotal = 0;
    minradiustotal = 0;


    print('Processing...')
    # This is for debuging. Run only certain number of images.
    count = 0;
    for ite in enumerate(files):
        print(' '+ite[1])

        filename = sourcepath+'/'+ite[1];

        # get the raw contour
        cha1,contours,imorig,__ = findOilBlob(filename,iterations=iteration,showresult=showresult);

        # shrink unwanted dimension
        oilcontour = contours[:,0,:].astype(float);
        '''
        # interpolate straight section
        oilcontour = interpolateContour(oilcontour,insert=freqratio-1,distancetol=10)
        '''
        # subsample
        '''
        ind = np.arange(0,int(len(oilcontour)/freqratio));
        oilcontour = oilcontour[ind*freqratio]
        '''
        print('size before subsample: ',oilcontour.shape)
        oilcontour = subSample(oilcontour,distancetol=subsampledistance)
        print('size after subsample: ',oilcontour.shape)
        # remove concave sections
        oilcontour = removeConcave(oilcontour)
        print('size after removal of concave points: ',oilcontour.shape)


        '''
        # averaging filter
        oilcontour[:,0] = avgFilter(oilcontour[:,0], int(freqratio))
        oilcontour[:,1] = avgFilter(oilcontour[:,1], int(freqratio))
        '''


        # compute curvature
        curvatureradius = findCurvatureRadius(oilcontour,window=window[0]);

        # recover the three channels of original figure
        imorig = cv2.cvtColor(imorig,cv2.COLOR_GRAY2BGR)


        # write points into image

        #maxradius = max(curvatureradius[:,2])
        #minradius = min(curvatureradius[:,2])

        maxradius = 190;
        minradius = 130;

        # See what is the range of the cavature radius
        if maxradiustotal < maxradius:
            maxradiustotal = np.copy(maxradius)
        if minradiustotal > minradius:
            minradiustotal = np.copy(minradius)

        numcontourpoint = curvatureradius.shape[0]
        curvatureradiusplot = (curvatureradius[:,2]-minradius) / (maxradius-minradius) * 256
        Nx,Ny,__ = imorig.shape
        for lp1 in range(numcontourpoint):
            # coordinate index
            indy = int(oilcontour[lp1,0])
            indx = int(oilcontour[lp1,1])
            # color
            imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,[0,1]] = 0
            imorig[indx-dotsize:indx+dotsize+1,indy-dotsize:indy+dotsize+1,2] = curvatureradiusplot[lp1]

        imshow(imorig,showresult)
        savename = targetpath+'/'+ite[1]
        cv2.imwrite(savename,imorig)

        '''
        count = count+1
        if count > 5:
            break
        '''


    print(maxradiustotal,minradiustotal)

    # plot the curvature radius of the last image
    plt.figure()
    plt.plot(curvatureradius[:,2],'sr-')

    # plot the color bar for the curvature radius
    tmp = np.linspace(0, 1, 256)
    fig = plt.figure()
    z = np.zeros(tmp.shape)
    img = np.zeros((len(tmp),20,3))
    for ite in range(20):
        img[:,ite,0] = tmp
    h = plt.imshow(img)

    plt.ylim(0,256)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    t1 = round(minradius*220/1280*2);
    t3 = round(maxradius*220/1280*2);
    t2 = round((t1+t3)/2);
    plt.yticks(ticks=[0,128,256],labels=[t1,t2,t3])
    plt.title(r'$\mu m$')
