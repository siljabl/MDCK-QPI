import imageio
import numpy as np
import skimage.io as io

def convert_to_old_format(stack):
    if (stack[0].dtype == np.uint16):
        new_stack = []
        for f in range(len(stack)):
            new_stack.append(np.array(stack[f],dtype=np.float32)/10000)
        return new_stack
    return stack

def convert_to_new_format(stack):
    if (stack[0].dtype == np.float32):
        new_stack=[]
        for f in range(len(stack)):
            new_stack.append(np.array(stack[f]*10000,dtype = np.uint16))
        return new_stack
    return stack

def is_tile(filepath):
    ### A tile does not satisfy one of the two following conditions:
    # - does not contain 61 frames
    # - is not 829 width and heigh. Maybe this is sufficient?
    im = io.imread(filepath,key=0)[0]
    return (len(im)!=829  or len(im[0])!=829)
