import numpy as np
import os
import shutil # copying files
from hdf5storage import loadmat#load matrices
import keras.backend as K
import scipy


def create_dir(name):
    """
    Creates the folder to save the model and makes a copy of the python
    training script (this is useful after training 100+ models :)   ) 
    """

    if not os.path.exists('savedModels/%s' % name):
        os.mkdir('savedModels/%s' % name)
        print("Creating " , name ,  " directory ")
    else:    
        print("Directory: " , name ,  "Folder already exists!")
    
    shutil.copy2('train.py', ( 'savedModels/%s' % name )) #created a copy of training file
    
    
def load_data( sets, path, split=False , input_size = np.nan, 
               overlap = np.nan ):
    
    """
    Loads the data (for train or test)
    
    The data was saved in matlab matrix format because of ... <- insert good reason here
    
    sets : tuple of integers indicating the domain number: 1 is a finneypack, etc
    path: location of the training data
    split: bool indicating wheter to split the domains in subdomains
    input_size: subdomain size
    overlap: indicates if these subdomains should be sample with overlap
    """
    
    if np.isnan(overlap) == True:
        overlap = input_size/2 - 1
            
    for i in range( 0 , np.size( sets ) ) :
        load_set = sets[i]
        print('Loading set no. %d' % load_set)
        
        solid           = loadmat('%s/solid_full_%d'        % (path , load_set))
        euclidean       = loadmat('%s/euclidean_pore_%d'    % (path , load_set))
        euclideanZ      = loadmat('%s/euclidean_poreZ_%d'   % (path , load_set))
        euclidean_total = loadmat('%s/euclidean_total_%d'   % (path , load_set))
        velocity_z      = loadmat('%s/velocity_z_full_%d'   % (path , load_set))
        time_flight_L   = loadmat('%s/ToF_l_%d'             % (path , load_set))
        time_flight_R   = loadmat('%s/ToF_r_%d'             % (path , load_set))
        mis_full        = loadmat('%s/MIS_full_%d'          % (path , load_set))
        mis_z           = loadmat('%s/MIS_z_inlet_%d'       % (path , load_set))
        
        p_solid             = solid['domain'] ; p_solid.astype('float32')
        p_euclidean_pore    = euclidean['e_domain'].astype('float32')
        p_euclidean_poreZ   = euclideanZ['e_z'].astype('float32')
        p_euclidean_total   = euclidean_total['e_full'].astype('float32')
        p_velocity_z        = velocity_z['vz'].astype('float32')
        p_tof_L             = time_flight_L['tOf_L'].astype('float32')
        p_tof_R             = time_flight_R['tOf_R'].astype('float32')
        p_mis_f             = mis_full['MIS_3D'].astype('float32')
        p_mis_z             = mis_z['MIS_3D'].astype('float32')
        
        phi = np.sum(p_solid<1)/p_solid.size
        print(f'The porosity of this domain is {phi}')
        p_solid = calculate_weighted_mask(p_solid)
        
        if split==True:

            vz_tmp      = split_matrix( p_velocity_z     , input_size , overlap)
            binary_tmp  = split_matrix( p_solid          , input_size , overlap) 
            e_pore_tmp  = split_matrix( p_euclidean_pore , input_size , overlap)
            e_poreZ_tmp = split_matrix( p_euclidean_poreZ, input_size , overlap)
            e_total_tmp = split_matrix( p_euclidean_total, input_size , overlap)
            tof_L_tmp   = split_matrix( p_tof_L          , input_size , overlap)
            tof_R_tmp   = split_matrix( p_tof_R          , input_size , overlap)
            MIS_f_tmp   = split_matrix( p_mis_f          , input_size , overlap)
            MIS_z_tmp   = split_matrix( p_mis_z          , input_size , overlap)
            
            if i == 0:
                
                vz          = vz_tmp
                binary      = binary_tmp
                e_pore      = e_pore_tmp
                e_poreZ     = e_poreZ_tmp
                e_total     = e_total_tmp
                tof_L       = tof_L_tmp
                tof_R       = tof_R_tmp
                MIS_f       = MIS_f_tmp
                MIS_z       = MIS_z_tmp
                
            else:
                
                vz        = np.concatenate( ( vz      , vz_tmp      ),  axis=0) 
                binary    = np.concatenate( ( binary  , binary_tmp  ),  axis=0)
                e_pore    = np.concatenate( ( e_pore  , e_pore_tmp  ),  axis=0) 
                e_poreZ   = np.concatenate( ( e_poreZ , e_poreZ_tmp ),  axis=0)
                e_total   = np.concatenate( ( e_total , e_total_tmp ),  axis=0) 
                tof_L     = np.concatenate( ( tof_L   , tof_L_tmp   ),  axis=0)
                tof_R     = np.concatenate( ( tof_R   , tof_R_tmp   ),  axis=0)
                MIS_f     = np.concatenate( ( MIS_f   , MIS_f_tmp   ),  axis=0)
                MIS_z     = np.concatenate( ( MIS_z   , MIS_z_tmp   ),  axis=0)
        
            t_set = { 'vz' : vz, 
                     'e_pore' : e_pore, 'e_total':e_total, 'e_poreZ': e_poreZ,
                     'tof_L':tof_L,'tof_R':tof_R,
                     'mis_f': MIS_f, 'mis_z': MIS_z, 'binary':binary}
                
        else: #if no splitting is requested
            t_set = {'vz':p_velocity_z,
                     'binary':p_solid,'e_pore':p_euclidean_pore, 
                     'e_poreZ':p_euclidean_poreZ, 
                     'e_total':p_euclidean_total,'tof_L': p_tof_L,'tof_R': p_tof_R,
                     'mis_f': p_mis_f, 'mis_z': p_mis_z}
                          
    return t_set

def calculate_weighted_mask(solid_mask):
    
    """
    Calculates the porosity weighted mask
    """
    for i in range( 0,solid_mask.shape[2] ):
        porosity = 1 - np.sum(solid_mask[:,:,i])/np.size(solid_mask[:,:,i])
        solid_mask[:,:,i][ solid_mask[:,:,i] == 0 ] = 1/porosity
        solid_mask[:,:,i] = solid_mask[:,:,i]/np.sum(solid_mask[:,:,i])*np.size(solid_mask[:,:,i])
    return solid_mask


def split_matrix(m, w_size, w_stride=0, erase_bcs=True):
    """
    Splits the 3D domain into smaller subdomains
    
    m: 3D domain
    w_size: size of the subsamples
    w_stride: stride lenght
    erase_bcs: bool. if true erases the boundary layers (to avoid noise)
    """
    
    w_stride=int(w_stride)
    
    if erase_bcs==True:
        m=np.delete(m,-1,0) #get rid of the boundaries
        m=np.delete(m,0 ,0)
        
        m=np.delete(m,-1,1)
        m=np.delete(m,0 ,1)
        
        m=np.delete(m,-1,2)
        m=np.delete(m,0 ,2)
        
    sample_start=np.arange(0,m.shape[0],w_size)
    sample_start=sample_start[sample_start<(m.shape[0]-(w_size+1))]
    sub_sample_start=sample_start+w_stride
    sub_sample_start=sub_sample_start[sub_sample_start<(m.shape[0]-(w_size+1))]
    
    if w_stride == 0: #if no overlap is requested
        mt=np.zeros((sample_start.size**3,w_size,w_size,w_size))
    else: #subsamples + overlap
        mt=np.zeros((sample_start.size**3+sub_sample_start.size**3,
                     w_size,w_size,w_size))
    
    ii=0
    
    for j in range(sample_start.size):
        for k in range(sample_start.size):
            for i in range(sample_start.size):
                
                mt[ii,:,:,:]=np.expand_dims(m[sample_start[k]:sample_start[k]+w_size, \
                                 sample_start[j]:sample_start[j]+w_size, \
                                 sample_start[i]:sample_start[i]+w_size],axis=0)
        
                ii=ii+1        
                 
    if w_stride!=0:
        for i in range(sub_sample_start.size):
            for j in range(sub_sample_start.size):
                for k in range(sub_sample_start.size):
    
                    mt[ii,:,:,:]=np.expand_dims(m[sub_sample_start[k]:sub_sample_start[k]+w_size, \
                                                     sub_sample_start[j]:sub_sample_start[j]+w_size, \
                                                     sub_sample_start[i]:sub_sample_start[i]+w_size],axis=0)
                    
                            
                    ii=ii+1

    return mt


def transform( x, tName, modelName, fileName='tmp', isTraining=True):
    """
    Performs the desired data transform
    x: array w/data
    tName: name of desired transformation
    modelName: name of the model (to save the summary stats)
    isTraining: bool. If true overwirtes existing file w/sum stats
    
    """
    
    
    if isTraining == True:
        x_stats = calculate_stats( x, modelName, fileName )
        x_mean    = x_stats['mean']
        x_min     = x_stats['min']
        x_max     = x_stats['max']
        x_maxAbs  = x_stats['maxAbs']
        x_minAbs  = x_stats['minAbs']
        x_std     = x_stats['std']
        x_range   = x_stats['range']
        x_p95     = x_stats['p95']
        x_new_min = x_stats['x_new_min']
        
    else:
        x_stats = np.loadtxt( 'savedModels/%s/%s.txt' % (modelName, fileName) , 
                             delimiter = ',' )
        
        x_mean    = x_stats[0]
        x_min     = x_stats[1] 
        x_range   = x_stats[2]
        x_std     = x_stats[3]
        x_max     = x_stats[4]
        x_maxAbs  = x_stats[5]
        x_minAbs  = x_stats[6]
        x_p95     = x_stats[7]
        x_new_min = x_stats[8]
    
    
    if tName == 'Constant':
        print( 'Dividing by 6e-6 Transform' )
        xt =  x/6e-6
    
    if tName == 'minMax_abs':
        print('minMax_abs')
        xt = (x-x_minAbs)/(x_maxAbs-x_minAbs)
    
    if tName == 'minMax_eps_2':
        print( 'minMax EPS 2 Transform' )
        xt = (   ( (x     - x_min)*(x_max - x_new_min)/x_range )/
                 ( (x_max - x_min)*(x_max - x_new_min)/x_range )   )*2-1    
    
    if tName == 'minMax':
        print( 'minMax Transform' )
        xt = ( x - x_min ) / x_range
        
    if tName == 'minMax_2':
        print( 'minMax 2 Transform' )
        xt = ( x - x_min )*2 / x_range - 1
        
    if tName == 'minMax_8':
        print( 'minMax 8 Transform' )
        xt = ( x - x_min )*8 / x_range - 2
        
    if tName == 'mMP95_2':
        print( 'mMP95 2 Transform' )
        xt = ( x )*2 / x_p95 - 1
        
    if tName == 'minMax_4':
        print( 'minMax 4 Transform' )
        xt = ( x - x_min )*4 / x_range - 2
        
    if tName == 'minMax_noZ':
        print( 'minMax Transform' )
        xt = ( x - x_min ) / x_range
        xt[ x==0 ] = 0
        
    if tName == 'normal':
        print('normal Transform')
        xt = ( x - x_mean ) / x_std
        #xt[ x==0 ] = 0
        
    if tName == 'normal_range2':
        print('normal Transform')
        xt = ( x - x_mean ) / x_std
        max_x = np.max( np.abs(xt) )
        print(max_x)
        xt = xt/max_x
        
    if tName == 'range':
        print('range Transform')
        xt = x / x_range
        
    if tName == 'max':
        print('max Transform')
        xt = x / x_maxAbs
        
    if tName == 'logCNN': 
        print('logCNN Transform')
        tmp  = np.abs(x) #absolute value
        tmp  = tmp / 3e-18 #divides by the min value
        tmp  = np.log10( tmp + 1 ) #plus one to eliminate the zeros
        xt   = tmp # this dist goes from 0 to ~13
        xt[ x<0 ] = ( -1 )*xt[ x<0 ] #adds the negative sign back
        xt = xt/13
        
    if tName == 'log_tmp': 
        print('logTMP Transform')
        x[ x==0 ]  = 1 #absolute value
        xt  = x/np.abs(x)*np.log10( np.abs(x) ) #plus one to eliminate the zeros

    if tName == 'log_tmp2': 
        print('logTMP2 Transform')
        x[ x==0 ]  = 1
        xt  = -x/np.abs(x)*np.log10( np.abs(x) ) - x/np.abs(x)*4.5
        xt[ np.abs(xt) == 4.5 ] = 0
        #xt = xt/6

    if tName == 'logCNN_test': 
        print('logCNN_test Transform')
        tmp  = np.abs(x) #absolute value
        tmp  = tmp / x_minAbs #divides by the min value
        tmp  = np.log10( tmp + 1 ) #plus one to eliminate the zeros
        xt   = tmp # this dist goes from 0 to ~13
        xt[ x<0 ] = ( -1 )*xt[ x<0 ] #adds the negative sign back
        xt = xt/13
                        
    if tName=='none':
        print('no Transform')
        xt = x
        
    summary_stats = { 'mean':x_mean, 'min':x_min, 'range':x_range, 'std':x_std, 
                'max':x_max, 'maxAbs':x_maxAbs, 'minAbs':x_minAbs }
    
    return xt , summary_stats

def calculate_stats( x, modelName, fileName ):
    
    x_mean  = x.mean()
    x_min   = x.min()
    x_max   = x.max()
    x_std   = x.std()
    x_range = x_max - x_min
    
    eps         = np.finfo(np.float32).eps
    x_new_min   = eps/(1/x_max) 
    
    x_maxAbs = np.max( np.abs(x) )
    x_minAbs = np.min( np.abs( x[ x>0 ] ) )
    
    x_p95 = np.percentile( x[x!=0], 95 )
    
    x_stats = { 'mean':x_mean, 'min':x_min, 'range':x_range, 'std':x_std, 
                'max':x_max, 'maxAbs':x_maxAbs, 'minAbs':x_minAbs, 'p95': x_p95,
                'x_new_min': x_new_min}
    
    np.savetxt( ('savedModels/%s/%s.txt' % (modelName, fileName) ),
               (x_mean, x_min, x_range, x_std, x_max, x_maxAbs, x_minAbs,
                x_p95, x_new_min), 
               delimiter=",", header="mean, min, range, std, max, maxAbs, minAbs, \
               P95, new_min")
    
    return x_stats

def custom_loss(y_true_weights, y_pred):
    
    weights = y_true_weights[:,:,:,:,0]
    y_true1 = y_true_weights[:,:,:,:,1]
    
    y_pred = K.squeeze(y_pred, axis = 4)
    
    y_true1 = y_true1*weights
    y_pred1 = y_pred*weights
    
    print(K.int_shape(y_pred1))
    print(K.int_shape(y_true1))
    
    return K.mean(K.square(y_pred1 - y_true1), axis=-1)

def mean_absolute_percentage_error_custom(y_true_weights, y_pred):
    
    y_true1 = y_true_weights[:,:,:,:,1]
    y_pred1 = K.squeeze(y_pred, axis = 4)
    diff    = K.abs( (y_true1 - y_pred1) / K.clip(K.abs(y_true1),
                                           K.epsilon(),
                                           None))
    return 100. * K.mean(diff, axis=-1)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_loc, list_IDs, branches, 
                 batch_size=5, dim=(80,80,80), 
                 n_channels_in=6,n_channels_out=1,shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.branches = branches
        
        self.file_loc = file_loc
        
       
        
        self.list_IDs = list_IDs
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        
        if self.branches == 0:
            pass
        if self.branches > 0:
            Xc = X
            x0 = Xc[:,:,:,:,0]; x0 = x0[:,:,:,:,np.newaxis]
            X = [x0]
        if self.branches > 1:
            x1 = Xc[:,:,:,:,1]; x1 = x1[:,:,:,:,np.newaxis]
            X.append(x1)
        if self.branches > 2:
            x2 = Xc[:,:,:,:,2]; x2 = x2[:,:,:,:,np.newaxis]
            X.append(x2)
        if self.branches > 3:
            x3 = Xc[:,:,:,:,3]; x3 = x3[:,:,:,:,np.newaxis]
            X.append(x3)
        if self.branches > 4:
            x4 = Xc[:,:,:,:,4]; x4 = x4[:,:,:,:,np.newaxis]
            X.append(x4)
        if self.branches > 5:
            x5 = Xc[:,:,:,:,5]; x5 = x5[:,:,:,:,np.newaxis]
            X.append(x5)
                
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_in ))
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = X_full[ID,:,:,:,:]
         
            X[i,] = np.load((self.file_loc + '/X/' + str(ID)+'.npy'))
            
            y[i,] = np.load((self.file_loc + '/y/' + str(ID)+'.npy'))

        return X, y

def write_data_chunks(X,y,name='tmp'):
    #folder name should be samples(21_26)_transform_features
    #sample name from 0 to len(X)
    dir_2write = "D:/SPLBM_output/chunks/"
    #dir_2write = ('../chunks/') #Darwin
    
    dir_name = dir_2write + name + '/'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir(dir_name + 'X')
        os.mkdir(dir_name + 'y')
    
    for i in range(0,X.shape[0]):
        np.save( (dir_name+'X/'+str(i)), X[i,:,:,:,:] )
        np.save( (dir_name+'y/'+str(i)), y[i,:,:,:,:] )

def calculate_DarcyPerm(v_avg,d_size=500):
    mu = 1/3
    dp = 0.0000001*(d_size/500)
    dpdx = dp/d_size
    k = v_avg*mu/dpdx
    return k   

def crop_sample(y, crop_size):
    
    m = np.copy(y)
    
    for i in range(0,crop_size):
        m=np.delete(m,-1,0) #get rid of the boundaries
        m=np.delete(m,0 ,0)
        
        m=np.delete(m,-1,1)
        m=np.delete(m,0 ,1)
        
        m=np.delete(m,-1,2)
        m=np.delete(m,0 ,2)
    
    return m

def remove_solid(y1, solid_val=0):
    y = np.copy(y1)
    tmp = y.flatten()
    tmp = tmp[1:1000]
    solid_value = scipy.stats.mode(tmp) #find the mode
    solid_value = solid_value[0] #value
    y[ y==solid_value ] = solid_val
    return y