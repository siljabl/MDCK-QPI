import numpy as np

from tqdm  import tqdm
from numba import njit, prange
import warnings

def scalar_temporal_correlation(var1, var2, Nframes, t_max=None):

    '''
    Computes temporal correlation of two scalars
    '''
    if t_max==None:
        t_max = Nframes

    delta_f = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)  
    C_norm  = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)
    N       = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)

    for i in tqdm(range(Nframes)):
    
        if  np.any(var1[i,:].mask==False):
            Noki = len(var1[i,:].compressed())
            
            if Noki>0:
                bool_oki = ~var1.mask[i,:]
            
                for j in range(i, Nframes):

                    #If X not masked for this frame do
                    if  np.any(var1[j,:].mask==False):
                        Nokj = len(var1[j,:].compressed())
                        
                        #If at least one acceptable PIV vector
                        if Nokj>0:        
                            bool_okj = ~var1.mask[j,:]   
                            bool_ok  = bool_oki * bool_okj
                            
                            var1_i = var1[i,bool_ok].compressed()
                            var1_j = var1[j,bool_ok].compressed()
                            
                            var2_i = var2[i,bool_ok].compressed()
                            var2_j = var2[j,bool_ok].compressed()
                                                            
                            # Normalize by the rms
                            rms = np.ma.sqrt(abs( np.ma.mean(var1_i * var2_i) * np.ma.mean(var1_j * var2_j) ))
                            C_norm[i,j-i] = np.ma.mean(var1_i * var2_j) / rms
                            N[i,j] = np.min([Noki, Nokj])
                        
                            delta_f[i,j] = j-i

    return C_norm[:,:t_max], N[:,:t_max], delta_f[:,:t_max]


def scalar_vector_temporal_correlation(var1, vec2, Nframes, t_max=None):
    '''
    Computes temporal correlation of two scalars
    '''
    if t_max==None:
        t_max = Nframes

    var2x, var2y = vec2

    delta_f = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)  
    C_norm  = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)
    N       = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)

    for i in tqdm(range(Nframes)):
    
        if  np.any(var2x[i,:].mask==False):
            Noki = len(var2x[i,:].compressed())
            
            if Noki>0:
                bool_oki = ~var2x.mask[i,:] * ~var2y.mask[i,:]
            
                for j in range(i, Nframes):

                    #If X not masked for this frame do
                    if  np.any(var2x[j,:].mask==False):
                        Nokj = len(var2x[j,:].compressed())
                        
                        #If at least one acceptable PIV vector
                        if Nokj>0:        
                            bool_okj = ~var2x.mask[j,:] * ~var2y.mask[j,:]
                            bool_ok  = bool_oki * bool_okj
                            
                            var1_i = var1[i,bool_ok].compressed()
                            var1_j = var1[j,bool_ok].compressed()
                            
                            var2x_i = var2x[i,bool_ok].compressed()
                            var2x_j = var2x[j,bool_ok].compressed()

                            var2y_i = var2y[i,bool_ok].compressed()
                            var2y_j = var2y[j,bool_ok].compressed()
                                                            
                            #Normalize by the rms
                            C_norm[i,j-i] = np.ma.mean(var1_i * (var2x_j + var2y_j)) /  np.ma.sqrt(abs( np.ma.mean(var1_i * (var2x_i + var2y_i)) * np.ma.mean(var1_j * (var2x_j + var2y_j)) ))
                            
                            N[i,j] = np.min([Noki, Nokj])
                        
                            delta_f[i,j] = j-i

    return C_norm[:,:t_max], N[:,:t_max], delta_f[:,:t_max]
    
    

def vector_temporal_correlation(vec1, vec2, Nframes, t_max=None):
    '''
    Computes temporal correlation of two scalars
    '''
    if t_max==None:
        t_max = Nframes
        
    var1x, var1y = vec1
    var2x, var2y = vec2

    delta_f = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)  
    C_norm  = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)
    N       = np.ma.masked_array(np.zeros(shape=(Nframes,Nframes)), True)

    for i in tqdm(range(Nframes)):
    
        if  np.any(var1x[i,:].mask==False):
            Noki = len(var1x[i,:].compressed())
            
            if Noki>0:
                bool_oki = ~var1x.mask[i,:] * ~var1y.mask[i,:]
            
                for j in range(i, Nframes):

                    #If X not masked for this frame do
                    if  np.any(var1x[j,:].mask==False):
                        Nokj     = len(var1x[j,:].compressed())
                        
                        #If at least one acceptable PIV vector
                        if Nokj>0:  
                            bool_okj = ~var1x.mask[j,:] * ~var1y.mask[j,:]   
                            bool_ok  = bool_oki * bool_okj
                            
                            var1x_i = var1x[i,bool_ok].compressed()
                            var1x_j = var1x[j,bool_ok].compressed()

                            var1y_i = var1y[i,bool_ok].compressed()
                            var1y_j = var1y[j,bool_ok].compressed()

                            var2x_i = var2x[i,bool_ok].compressed()
                            var2x_j = var2x[j,bool_ok].compressed()

                            var2y_i = var2y[i,bool_ok].compressed()
                            var2y_j = var2y[j,bool_ok].compressed()
                                                            
                            #Normalize by the rms
                            C_norm[i,j-i] = np.ma.mean(var1x_i * var2x_j + var1y_i * var2y_j) / np.ma.sqrt(abs( np.ma.mean(var1x_i*var2x_i + var1y_i*var2y_i ) * np.ma.mean(var1x_j*var2x_j + var1y_j*var2y_j ) ))
                            N[i,j] = np.min([Noki, Nokj])                       
                            delta_f[i,j] = j-i

    return C_norm[:,:t_max], N[:,:t_max], delta_f[:,:t_max]



def general_temporal_correlation(var1, var2=None, t_max=None, t_avrg=False):

    if np.any(var2==None):
        var2 = var1

    dim_var1 = np.shape(var1)
    dim_var2 = np.shape(var2)

    # len=2: variable is scalar, len=3: variable is vector
    assert len(dim_var1) in [2,3] and len(dim_var2) in [2,3]

    if len(dim_var1) == 2:
        Nframes = dim_var1[0]

        if len(dim_var2) == 2:
            C_norm, N, delta_f = scalar_temporal_correlation(var1, var2, Nframes, t_max)

        else:

            var2x = var2[0]
            var2y = var2[1]
            C_norm, N, delta_f = scalar_vector_temporal_correlation(var1, [var2x, var2y], Nframes, t_max)


    if len(dim_var1) == 3:
        Nframes = dim_var1[1]
        var1x = var1[0]
        var1y = var1[1]

        if len(dim_var2) == 2:

            C_norm, N, delta_f = scalar_vector_temporal_correlation(var2, [var1x, var1y], Nframes, t_max)

        else:

            var2x = var2[0]
            var2y = var2[1]
            C_norm, N, delta_f = vector_temporal_correlation([var1x, var1y], [var2x, var2y], Nframes, t_max)

    if t_avrg:
        C_norm = np.mean(C_norm, axis=0)                  

    COR = {'delta_f':  delta_f,
             'C_norm': C_norm,
             'N':      N}
            
    return COR


########################
# Spatial correlations #
########################

@njit(parallel=True)
def scalar_spatial_correlation_loopv2(xf, yf, var1f, var2f, r_bin_edges): 
    
    Nbins = len(r_bin_edges)-1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask = np.ones(Nbins, dtype=np.bool_)
    
    
    Cf_values = np.zeros(Nbins)
    Cf_mask = np.ones(Nbins, dtype=np.bool_)
    varf_rms = np.sqrt(abs(np.mean(var1f*var2f)))
    
    for i in prange(Nbins):
        
        for j in range(len(var1f)):
            
            #Get the coordinates of PIV vector i location
            xf_i = xf[j]
            yf_i = yf[j]
            
            #Shift to coordinate system with vector i at the center, and only consider the other vectors than vector i
            xf_rel = xf - xf_i
            yf_rel = yf - yf_i

            #Radial distances to the vectors
            rf_rel = np.sqrt( xf_rel**2 + yf_rel**2)
    
            if i==0:
                bool_in_bin = rf_rel==0
            else:
                bool_in_bin = (rf_rel> r_bin_edges[i] ) * (rf_rel<= r_bin_edges[i+1] )
    
    
            if np.any(bool_in_bin):
                Cf_mask[i] = False
                
                Nf_in_rbin_mask[i] = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var2f_in_bin = var2f[bool_in_bin]

                var1var2 = var1f[j] * var2f_in_bin
                
                # + keeps the mask intact when none in bin
                Cf_values[i] += np.sum( var1var2 / (varf_rms ** 2) )
        
    return Nf_in_rbin_values, Nf_in_rbin_mask, Cf_values, Cf_mask



def scalar_spatial_correlation(x, y, var1, var2, dr, r_max, Nmax=5000, every_n_frames = 1):
    
    Nframes = x.shape[0]
    
    #Add a point at zero
    r_bin_edges   = np.concatenate( ( [0],np.arange(0, r_max, dr) ) )
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1])/2

    Nbins     = len(r_bin_centers)
    C_norm    = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    
    frame_axis = np.arange(0, Nframes, every_n_frames, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)
    
    for f in tqdm(frame_axis):
        
        if  np.any(var1[f,:].mask==False):
        
            Nok = len(var1[f,:].compressed())
            
            #If at least one acceptable PIV vector
            if Nok>0:
                
                xf, yf, var1f, var2f  = x[f,:].compressed(),  y[f,:].compressed(),  var1[f,:].compressed(),  var2[f,:].compressed()
                
                if Nok>=Nmax:
                    
                    ind_selected = np.random.choice(np.arange(len(xf)), replace=False, size = Nmax)
                    xf = xf[ind_selected]  
                    yf = yf[ind_selected]  
                    var1f = var1f[ind_selected]  
                    var2f = var2f[ind_selected]  
                        

                frame_axis_masked.mask[f] = False   
                            
                Nf_in_rbin_values, Nf_in_rbin_mask, Cvf_norm_values, Cvf_mask = scalar_spatial_correlation_loopv2(xf, yf, var1f, var2f, r_bin_edges)
                
                C_norm[f,:]         = Cvf_norm_values
                C_norm.mask[f,:]    = Cvf_mask
                N_in_rbin[f,:]      = Nf_in_rbin_values
                N_in_rbin.mask[f,:] = Nf_in_rbin_mask


    C_norm = C_norm/N_in_rbin
              
    return C_norm, N_in_rbin, r_bin_centers, frame_axis_masked



@njit(parallel=True)
def scalar_vector_spatial_correlation_loopv2(xf, yf, var1f, vec2f, r_bin_edges): 
    
    var2xf, var2yf = vec2f

    Nbins = len(r_bin_edges)-1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask = np.ones(Nbins, dtype=np.bool_)
    
    
    Cf_values = np.zeros(Nbins)
    Cf_mask  = np.ones(Nbins, dtype=np.bool_)
    varf_rms = np.sqrt(np.mean(var1f*var2xf + var1f*var2yf))
    
    for i in prange(Nbins):
        
        for j in range(len(var2xf)):
            
            #Get the coordinates of PIV vector i location
            xf_i = xf[j]
            yf_i = yf[j]
            
            #Shift to coordinate system with vector i at the center, and only consider the other vectors than vector i
            xf_rel = xf - xf_i
            yf_rel = yf - yf_i

            #Radial distances to the vectors
            rf_rel = np.sqrt( xf_rel**2 + yf_rel**2)
    
            if i==0:
                bool_in_bin = rf_rel==0
            else:
                bool_in_bin = (rf_rel> r_bin_edges[i] ) * (rf_rel<= r_bin_edges[i+1] )
    
    
            if np.any(bool_in_bin):
                Cf_mask[i] = False
                
                Nf_in_rbin_mask[i] = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var1f_in_bin = var1f[bool_in_bin]

                var1var2 = var1f_in_bin * var2xf[j] + var1f_in_bin * var2yf[j]
                
                # + keeps the mask intact when none in bin
                Cf_values[i] += np.sum( var1var2 / (varf_rms ** 2) )
        
    return Nf_in_rbin_values, Nf_in_rbin_mask, Cf_values, Cf_mask



def scalar_vector_spatial_correlation(x, y, var1, vec2, dr, r_max, Nmax=5000, every_n_frames = 1):
    
    var2x, var2y = vec2

    Nframes = x.shape[0]
    
    #Add a point at zero
    r_bin_edges   = np.concatenate( ( [0],np.arange(0, r_max, dr) ) )
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1])/2

    Nbins     = len(r_bin_centers)
    C_norm    = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    
    frame_axis = np.arange(0, Nframes, every_n_frames, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)
    
    for f in tqdm(frame_axis):
        
        if  np.any(var2x[f,:].mask==False):
        
            Nok = len(var2x[f,:].compressed())
            
            #If at least one acceptable PIV vector
            if Nok>0:
                
                xf, yf =     x[f,:].compressed(),      y[f,:].compressed()
                var1f  =  var1[f,:].compressed()
                vec2f  = var2x[f,:].compressed(),  var2y[f,:].compressed()

                if Nok>=Nmax:
                    
                    ind_selected = np.random.choice(np.arange(len(xf)), replace=False, size = Nmax)
                    xf = xf[ind_selected]  
                    yf = yf[ind_selected]  
                    var1f = var1f[ind_selected]  
                    var2f = var2f[ind_selected]  
                        

                frame_axis_masked.mask[f] = False   
                            
                Nf_in_rbin_values, Nf_in_rbin_mask, Cvf_norm_values, Cvf_mask = scalar_vector_spatial_correlation_loopv2(xf, yf, var1f, vec2f, r_bin_edges)
                
                C_norm[f,:]         = Cvf_norm_values
                C_norm.mask[f,:]    = Cvf_mask
                N_in_rbin[f,:]      = Nf_in_rbin_values
                N_in_rbin.mask[f,:] = Nf_in_rbin_mask


    C_norm = C_norm/N_in_rbin
              
    return  C_norm, N_in_rbin, r_bin_centers, frame_axis_masked



@njit(parallel=True)
def vector_spatial_correlation_loopv2(xf, yf, vec1f, vec2f, r_bin_edges):

    var1xf, var1yf = vec1f
    var2xf, var2yf = vec2f
    
    Nbins = len(r_bin_edges)-1
    Nf_in_rbin_values = np.zeros(Nbins)
    Nf_in_rbin_mask   = np.ones(Nbins, dtype=np.bool_)
    
    Cvf_norm_values = np.zeros(Nbins)
    Cvf_mask = np.ones(Nbins, dtype=np.bool_)
    vf_rms   = np.sqrt(np.mean(var1xf*var2xf + var1yf*var2yf))
    
    
    for i in prange(Nbins):
    
        for j in range(len(var1xf)):

            #Get the coordinates of PIV vector i location
            xf_i = xf[j]
            yf_i = yf[j]
            
            #Shift to coordinate system with vector i at the center, and only consider the other vectors than vector i
            xf_rel = xf - xf_i
            yf_rel = yf - yf_i
        
            #Radial distances to the vectors
            rf_rel = np.sqrt( xf_rel**2 + yf_rel**2)
    
            if i==0:
                bool_in_bin = rf_rel==0
            else:
                bool_in_bin = (rf_rel> r_bin_edges[i] ) * (rf_rel<= r_bin_edges[i+1] )
    
            if np.any(bool_in_bin):
                Cvf_mask[i] = False
                
                Nf_in_rbin_mask[i] = False
                Nf_in_rbin_values[i] += np.sum(bool_in_bin)

                var1xf_in_bin = var1xf[bool_in_bin]
                var1yf_in_bin = var1yf[bool_in_bin]
            
                vxvx_vyvy = var1xf_in_bin * var2xf[j] + var1yf_in_bin * var2yf[j]
                
                # + keeps the mask intact when none in bin
                Cvf_norm_values[i]+= np.sum( vxvx_vyvy/(vf_rms * vf_rms) )
    

    return Nf_in_rbin_values, Nf_in_rbin_mask, Cvf_norm_values, Cvf_mask



def vector_spatial_correlation(x, y, vec1, vec2, dr, r_max):

    var1x, var1y = vec1
    var2x, var2y = vec2

    #Add a point at zero
    r_bin_edges   = np.concatenate( ( [0],np.arange(0, r_max, dr) ) )
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1])/2
    
    Nframes = x.shape[0]
    Nbins   = len(r_bin_centers)

    C_norm    = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    N_in_rbin = np.ma.masked_array(np.zeros(shape = (Nframes, Nbins)), True)
    
    frame_axis        = np.arange(0, Nframes, 1, dtype=int)
    frame_axis_masked = np.ma.masked_array(np.arange(0, Nframes, dtype=int), True)
    

    for f in tqdm(frame_axis):
        
        if  np.any(var1x[f,:].mask==False):
            
            Nok = len(var1x[f,:].compressed())
            
            #If at least one acceptable PIV vector
            if Nok>0:
                
                xf, yf =     x[f,:].compressed(),      y[f,:].compressed()
                vec1f  = var1x[f,:].compressed(),  var1y[f,:].compressed()
                vec2f  = var2x[f,:].compressed(),  var2y[f,:].compressed()
                        
                frame_axis_masked.mask[f] = False   
                            
                Nf_in_rbin_values, Nf_in_rbin_mask, Cvf_norm_values, Cvf_mask   =   vector_spatial_correlation_loopv2(xf, yf, vec1f, vec2f, r_bin_edges)
                
                C_norm[f,:]      = Cvf_norm_values
                C_norm.mask[f,:] = Cvf_mask

                N_in_rbin[f,:]      = Nf_in_rbin_values
                N_in_rbin.mask[f,:] = Nf_in_rbin_mask


    C_norm = C_norm/N_in_rbin
    
    return C_norm, N_in_rbin, r_bin_centers, frame_axis_masked




def general_spatial_correlation(x, y, var1, var2=None, dr=40, r_max=500, t_avrg=False):

    if np.any(var2==None):
        var2 = var1

    dim_var1 = np.shape(var1)
    dim_var2 = np.shape(var2)

    # len=2: variable is scalar, len=3: variable is vector
    assert len(dim_var1) in [2,3] and len(dim_var2) in [2,3]

    if len(dim_var1) == 2:

        if len(dim_var2) == 2:
            # print("scalar spatial correlation")

            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = scalar_spatial_correlation(x, y, var1, var2, dr, r_max)

        else:
            # print("scalar-vector spatial correlation")

            var2x = var2[0]
            var2y = var2[1]

            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = scalar_vector_spatial_correlation(x, y, var1, [var2x, var2y], dr, r_max)


    if len(dim_var1) == 3:

        var1x = var1[0]
        var1y = var1[1]

        if len(dim_var2) == 2:
            # print("scalar-vector spatial correlation")

            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = scalar_vector_spatial_correlation(x, y, var2, [var1x, var1y], dr, r_max)

        else:
            # print("vector spatial correlation")

            var2x = var2[0]
            var2y = var2[1]

            C_norm, N_in_rbin, r_bin_centers, frame_axis_masked = vector_spatial_correlation(x, y, [var1x, var1y], [var2x, var2y], dr, r_max)

    if t_avrg:
        C_norm = np.mean(C_norm, axis=0)
        r_bin_centers = np.ma.array(r_bin_centers, mask=C_norm.mask)

    
    COR = {'C_norm':          C_norm,
           'N_pairs_in_rbin': N_in_rbin,
           'r_bin_centers':   r_bin_centers,
           'frame_axis':      frame_axis_masked}
            
    return COR

