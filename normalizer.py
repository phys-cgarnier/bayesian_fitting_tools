import numpy as np
import torch



class Transformer_2D:
    def __init__(self, x, transform_type = 'normalize'):
        '''
        Transformer class that allows normalization and standardization of parameters.
        - Use forward method to normalize input vector
        - Use backward method to unnormalize input vector
        Does not support backpropagation!
        
        Arguments
        ---------
        x : ndarray, shape (N x M), optional, default None
             Input data to determine normalization parameters where N is the number of points and M is the dimensionality
        
        bounds : ndarray, shape (M x 2), optional, default None
             Alternate specification of normalization bounds instead of data, bounds[M][0] is the M'th lower bound,
                                                                              bounds[M][1] is the M'th upper bound
        
        transform_type : ['unitary', 'normalize', standardize']
            Transformation method.
                - 'unitary' : No modification of input data
                - 'normalize' : Scales and shifts data s.t. data is between 0 and 1
                - 'standardize' : Scales and shifts data s.t. data has a mean of 0.0 and a rms size of 1.0
        
        
        '''
        
        possible_transformations = ['normalize','standardize']
        assert transform_type in possible_transformations
        
        self.ttype = transform_type
       
        assert len(x.shape) == 2
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            
        self.x = x

        self._get_stats()
        
    def _get_stats(self):
        if self.ttype == 'normalize':
            self.mins = np.min(self.x, axis = 0)
            self.maxs = np.max(self.x, axis = 0) 

        elif self.ttype == 'standardize':
            self.means = np.mean(self.x, axis = 0)
            self.stds = np.std(self.x, axis = 0)

    def recalculate(self, x):
        #change transformer data and recalculate stats
        self.x = x
        self._get_stats()
    
    def forward(self, x_old):
        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True
        
        x = x_old.copy()
        assert len(x.shape) == 2

        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                if self.maxs[i] - self.mins[i] == 0.0:
                    x[:,i] = x[:,i] - self.mins[i]
                else:
                    x[:,i] = (x[:,i] - self.mins[i]) /(self.maxs[i] - self.mins[i])
                    
        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                if self.stds[i] == 0:
                    x[:,i] = x[:,i] - self.means[i]
                else:
                    x[:,i] = (x[:,i] - self.means[i]) / self.stds[i]

        #if torch_input:
            #x = torch.from_numpy(x)
        
        return x
                
    def backward(self, x_old):
        
        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True

        x = x_old.copy()
        assert len(x.shape) == 2
        
        if self.ttype == 'normalize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * (self.maxs[i] - self.mins[i]) + self.mins[i]

        elif self.ttype == 'standardize':
            for i in range(x.shape[1]):
                x[:,i] = x[:,i] * self.stds[i] + self.means[i]
    
       # if torch_input:
           # x = torch.from_numpy(x)
            
        return x




class Transformer_1D:
    def __init__(self, x, transform_type = 'normalize'):

    
        possible_transformations = ['normalize','standardize']
        assert transform_type in possible_transformations
        
        self.ttype = transform_type
       
        assert len(x.shape) == 1
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            
        self.x = x

        self._get_stats()
        
    def _get_stats(self):
        if self.ttype == 'normalize':
            self.max = np.max(self.x)

        elif self.ttype == 'standardize':
            self.mean = np.mean(self.x)
            self.std = np.std(self.x)


    def recalculate(self, x):
        #change transformer data and recalculate stats
        self.x = x
        self._get_stats()

    def forward(self, x_old):

        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True
        
        x = x_old.copy()
        assert len(x.shape) == 1
        
        if self.ttype == 'normalize':
        #min minmax scaler
        #from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # scaler.fit_transform(x)

            x = x/self.max
            
        elif self.ttype == 'standardize':
            #from sklearn.prepocessing import StandardScaler
            #scaler = StandardScaler()
            #scaler.fit_tranform(x)
            # x_scaled =  scaler.fit_transform(x)
            # col = x_scaled[:,0]
            # np.var(col)

            x = (x - self.mean)/self.std
        
        return x

    
    def backward(self, x_old):
        
        #if x_old is a torch tensor get numpy array from it
        if isinstance(x_old, torch.Tensor):
            x_old = x_old.detach().numpy()
            torch_input = True

        x = x_old.copy()
        assert len(x.shape) == 1
        
        if self.ttype == 'normalize':
            x = x*self.max

        elif self.ttype == 'standardize':
            x = x*self.std + self.mean
        
        return x
    



            
        
        
                
if __name__ == '__main__':
    #testing suite
    x = np.random.uniform(size = (10,3)) * 10.0
    #x = [[1,2,3],[4,5,6],[7,8,9]] 
    #x = np.array(x)   
    print(x)
    print('\n')
    t = Transformer_1D(x, 'normalize')
    x_new = t.forward(x)
    print(x_new)
    
    #print(t.backward(t.forward(x)))
