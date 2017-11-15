import numpy as np
import metric_learn
import traceback
from scipy.linalg import expm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist

class MetricTransform():
    def __init__(self, **param):
        
        self.model_name = param['model']
        self.mode = param['mode']
        
        self.param = param
        
        if self.mode == 'dist_exp':
            self.alpha = param['alpha']
        
        if self.model_name == 'LMNN':
            '''
            Initialize the LMNN object
            k=3, min_iter=50, max_iter=1000, learn_rate=1e-07, regularization=0.5, 
            convergence_tol=0.001, verbose=False
            k: number of neighbors to consider. (does not include self-edges)
            learn_rate: 1e-07
            regularization: weighting of pull and push terms
            '''
            self.k = param['k']
            self.min_iter = param['min_iter']
            self.max_iter = param['max_iter']
            self.learn_rate = param['learn_rate']
            self.regularization = param['regularization']
            self.conv_tol = param['conv_tol']
            
            self.model = getattr(metric_learn, self.model_name)(param['k'], param['min_iter'],
                                                              param['max_iter'], param['learn_rate'],
                                                              param['regularization'],
                                                              param['conv_tol'])
            
        if self.model_name == 'SDML':
            '''
            balance_param: float, optional
            trade off between sparsity and M0 prior
            
            sparsity_param: float, optional
            trade off between optimizer and sparseness (see graph_lasso)
            
            use_cov: bool, optional
            controls prior matrix, will use the identity if use_cov=False

            '''
            self.balance = param['balance']
            self.sparsity = param['sparsity']
            self.use_cov = param['use_cov']
            
            self.model = getattr(metric_learn, self.model_name)(param['balance'], param['sparsity'],
                                                              param['use_cov'])
            
        
        if self.model_name == 'LSML':
            '''
            tol=0.001, max_iter=1000
            '''
            self.num_constr = param['num_constr']
            
            self.model = getattr(metric_learn, self.model_name)(param['tol'],
                                                              param['max_iter'])
            self.tol = param['tol']
            self.max_iter = param['max_iter']
            
        if self.model_name == 'NCA':
            '''
            max_iter=100, learning_rate=0.01
            '''

            self.model = getattr(metric_learn, self.model_name)(param['max_iter'], 
                                                               param['learn_rate'])
            
        if self.model_name == 'LFDA':
            '''
            dim : dimensionality of reduced space (defaults to dimension of X)
            k : nearest neighbor used in local scaling method (default: 7)
            metric : type of metric in the embedding space (default: 'weighted')
            'weighted'        - weighted eigenvectors
            'orthonormalized' - orthonormalized
            'plain'           - raw eigenvectors
            '''
            self.model = getattr(metric_learn, self.model_name)(param['dim'], param['k'],
                                                              param['metric'])
            
            
        if self.model_name == 'RCA':
            '''
            dim : int, optional
            embedding dimension (default: original dimension of data)
            '''
            
            self.model = getattr(metric_learn, self.model_name)(param['dim'])
            #self.num_chunks = param['chunks']
            #self.chunk_size = param['chunk_size']
        
    
    def fit(self, X, y, grid = False):
        self.X = X
        
        if np.unique(y).shape[0]>2:
                print('Not binary task')
        else:
            self.labels = np.where(y > 0, 1, 0).reshape(-1)
            
        if self.model_name not in {'LMNN', 'NCA', 'LFDA'}:
            if self.model_name == 'SDML':
                new_y = np.where(self.labels > 0, 1, -1).reshape(-1,1)
                self.constraints = new_y.dot(new_y.T)
            
            if self.model_name == 'LSML':
                c = metric_learn.Constraints(self.labels) 
                self.constraints = c.positive_negative_pairs(self.num_constr)
                
            if self.model_name == 'RCA':
                
                self.constraints = self.labels
        else:
            self.constraints = self.labels
        try:
            self.model.fit(self.X, self.constraints)
            self.M = self.model.metric()
            if 'nan' in str(self.M[0][0]):
                print('M - not converge, consist of nan. Try to rescale data.')
                self.M = np.zeros((self.X.shape[1], self.X.shape[1]))
        except Exception as e:
            with open('Erros_msg.txt', 'a') as f:
                f.write(str(e) + ' \n' + traceback.format_exc())
                f.write('Cant fit metric, some reasons:')
                f.write('- too ill conditions \n- does not converge (big num) \n- read file Erros_msg.txt\n' )
            self.M = np.zeros((self.X.shape[1], self.X.shape[1]))
            
        
        
        
        
    def dist(self, X_test, mode = None):
        X = np.concatenate((self.X, X_test))
        m = X.shape[0]
        idx_train = range(0, self.X.shape[0])
        idx_test = range(self.X.shape[0], m)
        
#         dist_bt = np.array([X[i] - X[j] for i in range(m) 
#                             for j in range(i+1, m)])
#         dist_bt_M = dist_bt.dot(self.M)
#         dist_bt_ = [ dist_bt_M[i].dot(dist_bt[i].T) for i in range(dist_bt.shape[0])]
        
       
           
#         idx_r = np.triu_indices(m, k = 1)
        
#         new = np.zeros((m,m))
#         new[idx_r] = dist_bt_
#         new = new.T + new
        row, column = np.triu_indices(m, k = 1)
        mat = np.zeros((m,m))
 
        mat[row, column] = pdist(X, metric='mahalanobis', VI=self.M)**2
        mat = mat.T + mat
        #print('dist', np.allclose(mat, new))
        if mode =='exp':
            return mat, idx_test, idx_train
        else:
            return mat[np.ix_(idx_test, idx_train)]
    def transform(self, X = []):
        '''
        mode: what kind of metric return
                'inner_product' = X * M * X.T
                'dist' = (x_i - x_j) * M * (x_i - x_j).T
                'dist_exp' = exp(-alpha * dist(x_i, x_j))
        return: pairwise inner product
        '''
        if len(X) == 0:
            X = self.X
        if self.mode == 'inner_product':
            return X.dot(self.M).dot(self.X.T)/X.dot(self.X.T)
        if self.mode == 'dist':
            return self.dist(X)
        if self.mode == 'dist_exp':
            if self.alpha == None:
                print('error: No parametr for kernel')
            if self.alpha < 0:
                print('error: Not correct parametr, alpha should be positive')
            else:
                distance, idx_test, idx_train = self.dist(X, 'exp')
                if not np.all(np.linalg.eigvals(distance)>0):
                    min_all = abs(min(np.linalg.eigvals(distance)))+1e-10
                    distance += min_all * np.eye(distance.shape[0], distance.shape[1])
                if True in np.isnan(distance):
                    distance = np.zeros((distance.shape))
                if np.inf in distance:
                    distance = np.zeros((distance.shape))
                d = expm(-self.alpha*distance)[np.ix_(idx_test,idx_train)]
                #d = np.exp(-self.alpha*distance)[np.ix_(idx_test,idx_train)]
                if True in np.isnan(d):
                    d = np.zeros((d.shape))
                if np.inf in d:
                    d = np.zeros((d.shape))
                
                return d
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        X_new = self.transform()
        return X_new
    
    def get_params(self, deep = True ):
        return self.param
        
    def set_params(self, **param):
        for one in param:
            if one == 'model':
                setattr(self, one, getattr(metric_learn, param[one])())
            else:
                setattr(self,one, param[one])
            #print('self.', one, ' = ', param[one])