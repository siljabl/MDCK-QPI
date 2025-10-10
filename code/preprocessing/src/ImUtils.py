import scipy.ndimage as nd
import numpy as np
from copy import deepcopy
from skimage.transform import resize
import skimage.io as io
import imageio
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform
from skimage.transform import warp
import os as os
from skimage.morphology import binary_dilation
from skimage.morphology import binary_closing

def laplace2D(stack):
    """
    Returns the 2D laplacian (not in z) stack from a stack.
    """
    res = []
    for i in stack:
        res.append(nd.laplace(np.array(i,dtype=np.float64)))
    res = np.array(res)
    return res

def laplace2D_norm(stack):
    """
    Returns the 2D laplacian (not in z) stack from a stack.
    """
    res = []
    for i in stack:
        res.append(nd.laplace(np.array(i,dtype=np.float64))/(np.mean(i,dtype=np.float64)**2))
    res = np.array(res)
    return res


def COV(st1,st2):
    """
    Returns the non-normalized covariance between two stacks.
    """
    return np.abs(np.mean((st1-np.mean(st1))*(st2-np.mean(st2))))


def computeAlignedStack(args, threshold=5, strategy="laplace", XY_alignment=False, **kwargs):

    """
    Given any number of stacks, returns the aligned sub-stack.
    Alignement is performed between the first (reference) and the second one provided.
    If there are more, they are all aligned like the second one..
    """

    N_stack = len(args)
    ref = deepcopy(args[0])
    fl = []
    # Copy (to avoid side effects)
    for i in range(1,N_stack):
        fl.append(deepcopy(args[i]))
        
    # Resizing 
    for i in range(len(fl)):
        fl[i] = resize(np.array(fl[i]),(len(fl[i]),len(ref[0]),len(ref[0][0])))

    correlations = []
    correlations_IDX = []

    if (strategy=="laplace"):
        ref_lap = laplace2D(ref)
        fl_lap = laplace2D(fl[0])
        for s in fl_lap:
            tamp_corr = []
            for h in ref_lap:
                tamp_corr.append(COV(h,s))
            correlations.append(np.amax(tamp_corr))
            correlations_IDX.append(np.argmax(tamp_corr))
        
        print(" * Maximum correlation:", np.amax(correlations))
        ret = []
        for i in range(N_stack):
            ret.append([])
        for c in range(len(correlations)): ## len of fl[0]
            #print(correlations, np.amax(correlations))
            if (correlations[c]>np.amax(correlations)/threshold):
                ret[0].append(ref[correlations_IDX[c]])
                for i in range(len(fl)):
                    ret[i+1].append(fl[i][c])
    if (XY_alignment):
        ret = computeXYAlignedStack(ret, **kwargs)
    return ret, [correlations, correlations_IDX]



def computeXYAlignedStack(args,individual_align = False,idx = -1, **kwargs):
    """
    Given any number of stack, returns a stack of the same size, when images are xy-aligned with respect to the
    first stack.
    If individual_align = False (default), everything is aligned w/r to the first channel using the correlation with the
    second channel. If true, everything is aligned w/r to the first channel using its own correlation.
    If idx is given, the alignment is combined only on this img of the stack, and applied to all. Incompatible with
    individual_align.
    """
    
    N_stack = len(args)
    ref = deepcopy(args[0])
    fl = []
    for i in range(1,N_stack):
        fl.append(deepcopy(args[i]))
    
    ret = []
    for i in range(N_stack):
        ret.append([])

    if (idx==-1):  
        for i in range(len(ref)):
            ret[0].append(ref[i])
            if not (individual_align):
                shift, error, phasediff = phase_cross_correlation(ref[i],fl[0][i],upsample_factor=100)
                transform = EuclideanTransform(translation=np.flip(shift)).inverse
                for f in range(1,N_stack):
                    ret[f].append(warp(fl[f-1][i],transform))
            else:
                for f in range(1,N_stack):
                    shift, error, phasediff = phase_cross_correlation(ref[i],fl[f-1][i],upsample_factor=100)
                    transform = EuclideanTransform(translation=np.flip(shift)).inverse
                    ret[f].append(warp(fl[f-1][i],transform))
    else:
        shift, error, phasediff = phase_cross_correlation(ref[idx],fl[0][idx],upsample_factor=100)
        transform = EuclideanTransform(translation=np.flip(shift)).inverse
        for i in range(len(ref)):
            ret[0].append(ref[i])
            for f in range(1,N_stack):
                ret[f].append(warp(fl[f-1][i],transform))
        
        
    return ret


def getXYAlignment(args, **kwargs):
    """
    A similar to the previous one, in order to see how the alignment evolves in the stacks
    """
    N_stack = len(args)
    ref = deepcopy(args[0])
    fl = []
    for i in range(1,N_stack):
        fl.append(deepcopy(args[i]))
 
    shifts = []
    errors= []
    for i in range(len(ref)):
        for f in range(1,N_stack):
            shift, error, phasediff = phase_cross_correlation(ref[i],fl[f-1][i],upsample_factor=100)
            shifts.append(shift)
            errors.append(error)
    return shifts,errors

    

def commonStackReader(filename):
    ## imageio.v3
    img = imageio.v3.imread(filename)
    if (img.ndim == 3):
        return img
    ## skimage
    del img
    img = io.imread(filename,plugin="pil")    
    if (img.ndim == 3):
        return img

    print("/!\ No valid reader found!!!")



def commonMultiChannelStackReader(filename):
    ## imageio.v3
    img = imageio.v3.imread(filename)
    if (img.ndim == 4):
        return img
    ## skimage
    del img
    img = io.imread(filename,plugin="pil")    
    if (img.ndim == 4):
        return img

    print("/!\ No valid reader found!!!")
    


def binary_FEF(img, shape = (1,1,1)):
    """
    Computes the binary fourrier ellipsoid filtered of img.
    img has to be a binary, 3D stack
    """
    fft = np.fft.fftn(img)
    fft = nd.fourier_ellipsoid(fft, size=shape)
    ret = np.fft.ifftn(fft).real
    return ret > 0.5


def XY_surface(img):
    """
    Returns a stack of same shape with only the XY boundaries set as True
    """
    imgc = np.array(img,dtype = int)
    footprint = np.zeros((3,3,3))
    footprint[1] = [[0,1,0],[1,1,1],[0,1,0]]
    return np.array(binary_dilation(imgc, footprint=footprint) - imgc,dtype=bool)

def Z_surface(img):
    """
    Returns a stack of same shape with only the Z boundaries set as True
    """
    imgc = np.array(img,dtype = int)
    footprint = np.zeros((3,3,3))
    footprint[:,1,1] = [1,1,1]
    return np.array(binary_dilation(imgc, footprint=footprint) - imgc,dtype=bool)



def getElementSpacing(_shape):
    """ 
    Returns the spacings of the stack, in micron, given its shape. The dimension order is preserved.
    Call with img.shape.   
    """
    xy_shape = np.flip(_shape)[:2]
    
    ls = os.listdir("mhds")
    for i in ls:
        file = open("mhds"+os.sep+i,"r")
        lines = file.readlines()
        dims = []
        # voxel size appears to be determined by dims of x and y only
        for j in range(2):
            dims.append(int(lines[10].split(" ")[2+j]))
        dims = np.array(dims)
        dims2 = dims+1
        if (dims == xy_shape).all() or (dims2 == xy_shape).all():
            spacing = []
            for j in range(3):
                spacing.append(float(lines[9].split(" ")[2+j]))
            return np.flip(spacing)
        
    if 'spacing' not in locals():
        print(f"\nVoxel size for xy-dimension {xy_shape[0]}x{xy_shape[1]} is unknown.\nPlease add to /mhds\n")
        return np.array([-1, -1, -1])


