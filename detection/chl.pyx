
cimport cython
#from libcpp cimport bool
#from libc.string cimport memcpy
#import time
#import numpy as np
#cimport numpy as np
#from libc.stdlib cimport malloc
from libc.math cimport sqrt
from libc.string cimport memset
#import 




cdef float camera_factor = 1
cdef float camera_cx = 325.5
cdef float camera_cy = 253.5
cdef float camera_fx = 518.0
cdef float camera_fy = 519

cython: cdivision=True

#cdef list globe = []


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float GetDist(unsigned short [:,::1] image, int* points):
    cdef float x1 = 0
    cdef float y1 = 0
    cdef float z1 = 0
    cdef float x2 = 0
    cdef float y2 = 0
    cdef float z2 = 0

    cdef int p1x = points[0]
    cdef int p1y = points[1]
    cdef int p2x = points[2]
    cdef int p2y = points[3]


    cdef unsigned short* Mi1 = &image[points[1],0]
    cdef unsigned short* Mi2 = &image[points[3],0]
    cdef float d1 = Mi1[points[0]]
    cdef float d2 = Mi2[points[2]]

    if (d1 > 800 and d1 < 10000):
        z1 = d1 / camera_factor
        x1 = (points[0] - camera_cx) * z1 / camera_fx
        y1 = (points[1] - camera_cy) * z1 / camera_fy

    if (d2 > 800 and d2 < 10000):
        z2 = d2 / camera_factor
        x2 = (points[2] - camera_cx) * z2 / camera_fx
        y2 = (points[3] - camera_cy) * z2 / camera_fy

    cdef float dist = 0
    dist = sqrt(((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)))
    return dist


#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef list findz(unsigned short [:,::1] rawDepth):
#    return process(rawDepth)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list findz(unsigned short [:,::1] rawDepth):
    #cdef unsigned short [:,::1] rawDepth = dImage1
    cdef int quei[310000]
    cdef int quej[310000]
    cdef int seti[310000]
    cdef int setj[310000]
    cdef int setnum[310000]

    cdef int width = 640
    cdef int height = 480
    #print("test")

    cdef int flag[480][640]# = np.zeros((480,640))
    #print("initialized")
    memset(flag, 0, sizeof(flag[0][0]) * width * height);
    #print("set")
    #print(flag[0][0])
    #cdef double[:, :] flag# = flagg
    #flag[:,] = 0
    #flag[1,:] = 0

    cpdef list return_list = []
    cdef int setcount = 0
    cdef int j,i
    cdef int tempi,tempj
    cdef float d,dist

    cdef int pixelNum = 120
    cdef list times = []
    cdef int points[4]
    cdef unsigned short* Mi
    for j in range(0,height):
        Mi = &rawDepth[j, 0]
        #if(j == 311):
            #print(flag[100,100])
        #print(j)
        for i in range(0,width):
            d = Mi[i]
            #print(d)
            #if (flag[(j*height) + i] == 0 and d >800 and d<8000):
            #if(j == 311):
            #    print(flag[j,i],d) 
            #if(j == 311 and i == 128):
            #    print("before",j,i,flag[j,i],d)
            if (flag[j][i] == 0 and d >800 and d<8000):
                #print("one worked")
                #if(j == 311):
                #    print("after",j,i,flag[j,i],d)
                setcount += 1
                seti[setcount] = i
                setj[setcount] = j
                setnum[setcount] = 1
                flag[j][i] = setcount

                h = 1
                t = 1
                quei[1] = i
                quej[1] = j
                while (h <= t):
                    tempi = quei[h]
                    tempj = quej[h]
                       
                    if (tempi>0 and flag[tempj][tempi-1] == 0):
                        points[:] = [tempi, tempj, tempi - 1, tempj]
                        dist = GetDist(rawDepth, &points[0])
                        #if(j == 311):
                            #print(dist)
                        if(dist < pixelNum):
                            t += 1
                            quei[t] = tempi - 1
                            quej[t] = tempj
                            flag[tempj][tempi-1] = setcount
                            setnum[setcount] += 1
                        
                    if (tempj < 479 and flag[tempj+1][tempi] == 0):
                        points[:] = [tempi, tempj, tempi, tempj + 1]
                        dist = GetDist(rawDepth, &points[0])
                        if (dist < pixelNum):
                            t += 1
                            quei[t] = tempi
                            quej[t] = tempj + 1
                            flag[tempj+1][tempi] = setcount
                            setnum[setcount] += 1
                        
                    if (tempi < 639 and flag[tempj][tempi+1] ==0):
                        points[:] = [tempi, tempj, tempi + 1, tempj]
                        dist = GetDist(rawDepth, &points[0])
                        if (dist < pixelNum):
                            t += 1
                            quei[t] = tempi + 1
                            quej[t] = tempj
                            flag[tempj][tempi+1] = setcount
                            setnum[setcount] += 1
                    h += 1

    #cdef int ccount = 0
    #cdef int ncount = 0
    cdef int k
    cdef int fi
    cdef int fj
    #print(seti)
    #print(setj)
    #cdef unsigned short* Mi
    #dImage = dImage1.copy()
    #cv2.normalize(dImage,  dImage, 0, 255, cv2.NORM_MINMAX)
    #print(dImage)
    #d_n_image = d_n_image.reshape(d_n_image.shape[0],d_n_image.shape[1],1)
    #rgbImage = rgbImage[...,::-1]
    #cdef np.ndarray combined_image = np.concatenate((d_n_image, rgbImage), axis=2)
    #combined_image = (combined_image.astype('float32') - 127.5)/128
    #cdef int xmin = 0,ymin=0,xmax=0,ymax=0
    #cdef np.ndarray roi_combined_image
    #cdef np.ndarray roi_detect
    for k in range (1,setcount):
        fi = seti[k]
        fj = setj[k]
        #print(setj[k])
        Mi = &rawDepth[fj, 0]
        if(setnum[k] > 400):
            #return_list.append([ccount,fi,fj,Mi[fi]])
            box = get_bounding_box_WASSIMEA(fi,fj,Mi[fi])
            #print(boxes)
            #xmin = box[0]
            #ymin = box[1]
            #xmax = box[2]
            #ymax = box[3]
            #roi_combined_image = combined_image[ymin:ymax, xmin:xmax]
            #width_original = roi_combined_image.shape[1]
            #roi_detect = cv2.resize(roi_combined_image, (48,48))
            #roi_detect = roi_detect.reshape(1,48,48,4)
            #print(box)
            return_list.append(box)
            #return_list.append([roi_detect,width_original,box])

            #return_list.append()

    #memset(quei, 0, sizeof(quei))        
    #memset(quej, 0, sizeof(quej))
    #memset(seti, 0, sizeof(seti))
    #memset(setj, 0, sizeof(setj))
    #memset(setnum, 0, sizeof(setnum))
    #print("at memset")
    #memset(flagg, 0, sizeof(flagg[0][0]) * width * height);
    return return_list

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list get_bounding_box_WASSIMEA(int x,int y,float z):
    cdef float factor = 1600 / z 
    cdef int xmin = <int>(x - (120 * factor / 2))
    cdef int ymin = <int>(y - (120 * (factor / 4)))
    cdef int xmax = <int>(xmin + (150 * factor))
    cdef int ymax = <int>(ymin + (150 * factor))
    #print(x,y,z)
    if(xmin < 0):
        xmin = 0
    if(ymin < 0):
        ymin = 0
    if(xmax > 640):
        xmax = 640
    if(ymax > 480):
        ymax = 480
    cdef int width = xmax - xmin
    cdef int height = ymax - ymin
    while (height != width):
        if(width > height):
            ymax = ymax + 1
            height = ymax - ymin
        elif(height > width):
            xmax = xmax + 1
            width = xmax - xmin
    return [xmin,ymin,xmax,ymax]





