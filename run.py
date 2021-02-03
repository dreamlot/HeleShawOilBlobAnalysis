# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 03:46:14 2020

@author: Ningyu Wang
"""
from main import *

# list of path
path_list = [
    'E:/20201111HeleShaw/N35-8',
    'E:/20201111HeleShaw/N45-4',
    'E:/20201111HeleShaw/N45-6',
    'E:/20201111HeleShaw/N45-8',
    'E:/20201111HeleShaw/N52-4',
    'E:/20201111HeleShaw/N52-6',
    'E:/20201111HeleShaw/N52-8'
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


for ite0 in range(len(path_list)):
    
    if ite0!=1:
        continue
    
    sourcepath_cut = path_list[ite0]
    '''
    cut the image, use only the left hand side half
    '''
    # working directory
    #sourcepath_cut = 'E:/20201111HeleShaw/N45-4';
    targetpath_cut = sourcepath_cut +'/cut';
    
    #os.mkdir(targetpath_cut)
    
    '''
    from cut import cutall
    cutall(x=[0,1000],y=[400,1000],FlagPercent=False,sourcepath=sourcepath_cut,targetpath=targetpath_cut)
    '''
    
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
        count = count+1
        
        # Uncomment this blob if need to accelerate the computation.
        #if count % 10 > 0:
        #    continue
        #if count < 380:
        #    continue
        #if count > 382:
        #    break
        
        print(' '+ite[1])
        
        # output the specific data for plotting
        #if ite[1] != 'n45-40380.tif':
        #    continue
        
    
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
        
        ind = np.arange(0,int(len(oilcontour)/freqratio));
        oilcontour = oilcontour[ind*freqratio]
        
        
        
        # comment the following codes for quick evaluation of a,b,x,y,e
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
    
        '''
        # comment the following codes  for quick evaluation of a,b,x,y,e
        # to save time, skip this par, directly fit the ellipse
        
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
        
        # output the specific data for plotting
        #np.savetxt('E:\\20201111HeleShaw\\N45-4\\result\\contour.txt',oilcontour);
        #np.savetxt('E:\\20201111HeleShaw\\N45-4\\result\\curvatureradius.txt',curvatureradius);
        
    
        '''
        if count > 5:
            break
        '''
        # find the ecliptical fit
        ellps = findEllipse(oilcontour)
        # tmpa = max(a_list[ite0],b_list[ite0])
        # tmpb = min(a_list[ite0],b_list[ite0])
        
        x_list[ite0] = np.append(x_list[ite0],ellps[0])
        y_list[ite0] = np.append(y_list[ite0],ellps[1])
        
        if ellps[2] > ellps[3]:
            
            a_list[ite0] = np.append(a_list[ite0],ellps[2])
            b_list[ite0] = np.append(b_list[ite0],ellps[3])
            
            
            angle_list[ite0] = np.append(angle_list[ite0],ellps[4])
            e_list[ite0] = np.append(e_list[ite0],np.sqrt(1-(ellps[3]/ellps[2])**2))
        else:
            
            a_list[ite0] = np.append(a_list[ite0],ellps[3])
            b_list[ite0] = np.append(b_list[ite0],ellps[2])
            
            
            angle_list[ite0] = np.append(angle_list[ite0],ellps[4]+np.pi/2)
            e_list[ite0] = np.append(e_list[ite0],np.sqrt(1-(ellps[2]/ellps[3])**2))
            
        
    
    savename = path_list[ite0]+'_x.txt'
    np.savetxt(savename,x_list[ite0])    
    savename = path_list[ite0]+'_y.txt'
    np.savetxt(savename,y_list[ite0])    
    savename = path_list[ite0]+'_a.txt'
    np.savetxt(savename,a_list[ite0])    
    savename = path_list[ite0]+'_b.txt'
    np.savetxt(savename,b_list[ite0])    
    savename = path_list[ite0]+'_angle.txt'
    np.savetxt(savename,angle_list[ite0])
    savename = path_list[ite0]+'_e.txt'
    np.savetxt(savename,e_list[ite0])    
    
    print(maxradiustotal,minradiustotal)
    
