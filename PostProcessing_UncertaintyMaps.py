import cv2
import numpy as np
import matplotlib.pyplot as plt 

def map_whitout_edges(img:,
                    kernel_size : int = 5):
    '''
    Perform erosion operation to remove the edges on the uncertainty map
    '''
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)

    return img_erosion

def binary_map(img:,
            kernel_size : int = 5):

     '''
    Perform a tresholding using Otsus method following 
    by an erosion operation to create binary uncertainty map
    '''
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((kernel_size,kernel_size), np.uint8)

    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary = cv2.erode(th2, kernel, iterations=1)

    return img_binary