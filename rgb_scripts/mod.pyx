cimport cython
import cv2
import numpy as np
cimport numpy as np
from libc.string cimport memset
from libc.math cimport sqrt,atan,cos,sin
#import time

cdef float camera_factor = 1
cdef float camera_cx = 325.5
cdef float camera_cy = 253.5
cdef float camera_fx = 518.0   
cdef float camera_fy = 519.0


cython: cdivision=True

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray get_channels(unsigned short [:,:] rawDepth, headpoint):
    cdef int width = rawDepth.shape[1]
    cdef int height = rawDepth.shape[0]
    cdef unsigned short [:,:] c2 = np.zeros((height,width)).astype(np.ushort)
    cdef unsigned short [:,:] c3 = np.zeros((height,width)).astype(np.ushort)

    cdef unsigned short [:,:] d1 = np.zeros((height,width)).astype(np.ushort)
    cdef unsigned short [:,:] d2 = np.zeros((height,width)).astype(np.ushort)

    #memset(c2, 0, sizeof(c2[0,0]) * width * height)
    #memset(c3, 0, sizeof(c3[0,0]) * width * height)
    #memset(d1, 0, sizeof(d1[0,0]) * width * height)
    #memset(d2, 0, sizeof(d2[0,0]) * width * height)

    cpdef np.ndarray mod = np.zeros((height,width,3), np.int32)

    d1[0,0] = 0
    d1[0,width - 1] = 0
    d1[height - 1,0] = 0
    d1[height - 1,width - 1] = 0

    c2[0,0] = 0
    c2[0,width - 1] = 0
    c2[height - 1,0] = 0
    c2[height - 1,width - 1] = 0

    cdef float lightness = 255/24
    cdef int j
    cdef int i
    cdef float d1x
    cdef float d1y
    cdef float d2xx
    cdef float d2yy
    cdef float d2xy
    cdef float theta
    cdef float val
    cdef unsigned short* Mj
    cdef unsigned short* Mjp1
    cdef unsigned short* Mjm1
    cdef unsigned short* d2j
    cdef unsigned short* d2jp1
    cdef unsigned short* d2jm1
    cdef unsigned short* d1j
    cdef unsigned short* d1jp1
    cdef unsigned short* d1jm1

    cdef int points_for_dist[4]

    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] Mj_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] Mjp1_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] Mjm1_buff

    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d1j_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d1jp1_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d1jm1_buff

    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d2j_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d2jp1_buff
    #cdef np.ndarray[np.uint32_t, ndim=1, mode = 'c'] d2jm1_buff

    #Mj_buff = np.ascontiguousarray(rawDepth[:,j], dtype = np.uint32)
    #Mj = Mj_buff.data
    #Mjp1_buff = np.ascontiguousarray(rawDepth[:,j+1], dtype = np.uint32)
    #Mjp1 =  Mjp1_buff.data
    #Mjm1_buff = np.ascontiguousarray(rawDepth[:,j-1], dtype = np.uint32)
    #Mjm1 = Mjm1_buff.data
    #d1j_buff = np.ascontiguousarray(d1[:,j], dtype = np.uint32)
    #d1j = <unsigned int*> d1j_buff.data
    #d1jp1_buff = np.ascontiguousarray(d1[:,j+1], dtype = np.uint32)
    #d1jp1 = <unsigned int*> d1jp1_buff.data
    #d1jm1_buff = np.ascontiguousarray(d1[:,j-1], dtype = np.uint32)
    #d1jm1 = <unsigned int*> d1jm1_buff.data

    #d2j_buff = np.ascontiguousarray(d2[:,j], dtype = np.uint32)
    #d2j = <unsigned int*> d2j_buff.data
    #d2jp1_buff = np.ascontiguousarray(d2[:,j+1], dtype = np.uint32)
    #d2jp1 = <unsigned int*> d2jp1_buff.data
    #d2jm1_buff = np.ascontiguousarray(d2[:,j-1], dtype = np.uint32)
    #d2jm1 = <unsigned int*> d2jm1_buff.data
    #startcy = time.clock()
    for j in range (1,width - 1):
        Mj = &rawDepth[j, 0]
        Mjp1 = &rawDepth[j+1, 0]
        Mjm1 = &rawDepth[j-1, 0]
        for i in range(1,height - 1):
            d1x = 0.5 * (Mj[i+1] - Mj[i-1])
            d1y = 0.5 * (Mjp1[i] - Mjm1[i])
            d1[i,j] = <unsigned short>sqrt((d1x*d1x) + (d1y*d1y))

            d2xx = Mj[i+1] - 2 * Mj[i] + Mj[i-1]
            d2yy = Mjp1[i] - 2 * Mj[i] + Mjm1[i]
            d2xy = Mjp1[i+1] + Mj[i] - Mjp1[i] - Mj[i+1]
            if d2xx == d2yy:
                theta = -1
            else:
                theta = 0.5 * atan((2 * d2xy) / (d2xx - d2yy))
            #print("==================")
            #print(theta)
            if theta == -1:
                d2[i,j] = 0
            else:
                val = (d2xx * (cos(theta) * cos(theta))) + (2 * d2xy * cos(theta) * sin(theta)) + (d2yy * (sin(theta) * sin(theta)))
                if (val > 0):
                    d2[i,j] = <unsigned short>val
                else:
                    d2[i,j] = 0
            d1[i,j] = <unsigned short>sqrt((d1x*d1x) + (d1y*d1y))
    #print(time.clock() - startcy)
    #startcy = time.clock()
    for j in range (1,width - 1):
        Mj = &rawDepth[j, 0]
        Mjp1 = &rawDepth[j+1, 0]
        Mjm1 = &rawDepth[j-1, 0]

        d2j = &d2[j, 0]
        d2jp1 = &d2[j+1, 0]
        d2jm1 = &d2[j-1, 0]

        d1j = &d1[j, 0]
        d1jp1 = &d1[j+1, 0]
        d1jm1 = &d1[j-1, 0]

        for i in range(1,height - 1):
            points = 0
            if(Mj[i] - Mjp1[i-1] >3 and Mj[i] - Mjp1[i+1] >3):
                points = points + 1
            if(Mj[i] - Mjm1[i-1] >3 and Mj[i] - Mjp1[i-1] >3):
                points = points + 1
            if(Mj[i] - Mjm1[i-1] >3 and Mj[i] - Mjm1[i+1] >3):
                points = points + 1
            if(Mj[i] - Mjm1[i+1] >3 and Mj[i] - Mjp1[i+1] >3):
                points = points + 1
            if(Mj[i] - Mj[i-1] >3 and Mj[i] - Mjp1[i] >3):
                points = points + 1
            if(Mj[i] - Mjm1[i] >3 and Mj[i] - Mj[i-1] >3):
                points = points + 1
            if(Mj[i] - Mjm1[i] >3 and Mj[i] - Mj[i+1] >3):
                points = points + 1
            if(Mj[i] - Mjp1[i] >3 and Mj[i] - Mj[i+1] >3):
                points = points + 1


            if(d1j[i] - d1jp1[i-1] >3 and d1j[i] - d1jp1[i+1] >3):
                points = points + 1
            if(d1j[i] - d1jm1[i-1] >3 and d1j[i] - d1jp1[i-1] >3):
                points = points + 1
            if(d1j[i] - d1jm1[i-1] >3 and d1j[i] - d1jm1[i+1] >3):
                points = points + 1
            if(d1j[i] - d1jm1[i+1] >3 and d1j[i] - d1jp1[i+1] >3):
                points = points + 1
            if(d1j[i] - d1j[i-1] >3 and d1j[i] - d1jp1[i] >3):
                points = points + 1
            if(d1j[i] - d1jm1[i] >3 and d1j[i] - d1j[i-1] >3):
                points = points + 1
            if(d1j[i] - d1jm1[i] >3 and d1j[i] - d1j[i+1] >3):
                points = points + 1
            if(d1j[i] - d1jp1[i] >3 and d1j[i] - d1j[i+1] >3):
                points = points + 1

            if(d2j[i] - d2jp1[i-1] >3 and d2j[i] - d2jp1[i+1] >3):
                points = points + 1
            if(d2j[i] - d2jm1[i-1] >3 and d2j[i] - d2jp1[i-1] >3):
                points = points + 1
            if(d2j[i] - d2jm1[i-1] >3 and d2j[i] - d2jm1[i+1] >3):
                points = points + 1
            if(d2j[i] - d2jm1[i+1] >3 and d2j[i] - d2jp1[i+1] >3):
                points = points + 1
            if(d2j[i] - d2j[i-1] >3 and d2j[i] - d2jp1[i] >3):
                points = points + 1
            if(d2j[i] - d2jm1[i] >3 and d2j[i] - d2j[i-1] >3):
                points = points + 1
            if(d2j[i] - d2jm1[i] >3 and d2j[i] - d2j[i+1] >3):
                points = points + 1
            if(d2j[i] - d2jp1[i] >3 and d2j[i] - d2j[i+1] >3):
                points = points + 1
            c2[i,j] = int(points * lightness)
            points_for_dist[:] = [headpoint[1], headpoint[0], j, i]
            c3[i,j] = GetDist(rawDepth, &points_for_dist[0] ) 
    #print(time.clock() - startcy)
    #image = image.astype('float64')
    #x = np.asarray(c2)
    mod = cv2.merge([np.asarray(rawDepth), np.asarray(c2) ,np.asarray(c3)])    
    return mod

@cython.boundscheck(False)
@cython.wraparound(False)
cdef GetDist(unsigned short [:,:] img, int* points):
    #cdef float x1 = 0
    cdef float y1 = 0
    cdef float z1 = 0
    #cdef float x2 = 0
    cdef float y2 = 0
    cdef float z2 = 0

    cdef int p1x = points[0]
    cdef int p1y = points[1]
    cdef int p2x = points[2]
    cdef int p2y = points[3]


    cdef unsigned short* Mi1 = &img[points[1],0]
    cdef unsigned short* Mi2 = &img[points[3],0]
    cdef float d1 = Mi1[points[0]]
    cdef float d2 = Mi2[points[2]]

    if (d1 > 800 and d1 < 10000):
        z1 = float(d1) / camera_factor
        #x1 = (p1[0] - camera_cx) * z1 / camera_fx
        y1 = (p1y - camera_cy) * z1 / camera_fy
    if (d2 > 800 and d2 < 10000):
        z2 = float(d2) / camera_factor
        #x2 = (p2[0] - camera_cx) * z2 / camera_fx
        y2 = (p2y - camera_cy) * z2 / camera_fy
    cdef float dist = 0
    #dist = np.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2) *(z1 - z2))
    dist = sqrt((y1 - y2)*(y1 - y2))# + (z1 - z2) *(z1 - z2))
    #dist = dist/255
    return dist