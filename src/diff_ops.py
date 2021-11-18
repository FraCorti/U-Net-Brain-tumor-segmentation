from scipy import ndimage
import numpy as np

def dx_backward(u):
    
    M, N = u.shape

    return np.concatenate((u[:, :-1], np.zeros([M, 1])), axis = 1) - \
           np.concatenate((np.zeros([M, 1]), u[:, :-1]), axis = 1)
 
                    
def dx_forward(u):
    
    M, N = u.shape
    
    return np.concatenate((u[:, 1:], u[:, -1, None]), axis = 1) - u


def dy_backward(u):
    
    M, N = u.shape

    return np.concatenate((u[:-1, :], np.zeros([1, N])), axis = 0) - \
           np.concatenate((np.zeros([1, N]), u[:-1, :]), axis = 0)  
 
           
def dy_forward(u):
    
    M, N = u.shape

    return np.concatenate((u[1:, :], u[None, -1, :]), axis = 0) - u