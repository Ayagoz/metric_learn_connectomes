def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from norms import OrigN, OrigS, SpectralNorm, WbysqDist, BinarNorm

import os
import pandas as pd
import numpy as np

from MetricTransform import MetricTransform

from reskit.core import DataTransformer, MatrixTransformer, Pipeliner

from sklearn.feature_selection import VarianceThreshold

from reskit.features import degrees, bag_of_edges, closeness_centrality, betweenness_centrality


from convert import convert
from load_data import load_ucla

from norms import OrigN, OrigS, SpectralNorm, WbysqDist, BinarNorm

path_ucla = '../../../Connectomics/Autism/Data/'

matrix_ucla, target_ucla, xyz = load_ucla(path_ucla)
X = {}
X['data'] = matrix_ucla
X['dist'] = xyz
y = target_ucla

param_SDML = {'model': 'SDML',
              'mode': 'dist_exp',
              'alpha': 1e-5,
             'use_cov': False,
             'sparsity': 0.01,
             'balance': 0.5 
             }

normalizers1 = [('wbysqdist', WbysqDist())]
normalizers2 = [('spectral', SpectralNorm())]

featurizers = [('degrees', MatrixTransformer(degrees))]

selection = [
            ('origS', OrigS())]

pipe_sdml = Pipeline(steps=[('metric', MetricTransform(**param_SDML)), ('clf', SVC(kernel='precomputed'))])

metric_trans_learn = [('SDML', pipe_sdml)]

steps = [('normalization1', normalizers1),
         ('normalization2', normalizers2),
         ('features', featurizers),
         ('selection', selection),
         ('machine_learning', metric_trans_learn)]

param_grid = {'SDML': {'metric__use_cov': [False],
                   'metric__sparsity': [0.01, 0.15, 0.25,0.5],
                   'metric__balance': [0.1,  0.3, 0.6, 0.8],
                    'metric__alpha': [0.1,  1e-4, 1e-7, 1,10, 100],
                   'clf__C':[1e-7, 1e-3,  0.1, 1, 10, 100]}
    }
    


grid_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
eval_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

pipeliner = Pipeliner(steps=steps, grid_cv=grid_cv, 
                      eval_cv=eval_cv, param_grid=param_grid)



table = pipeliner.get_results(X, y, scoring=['roc_auc'], caching_steps = ['normalization1','normalization2','features','selection'], logs_file = 'log/ucla_results_SDML_exp_comp.log')
print('finished')

table.to_csv('ucla_results_SDML_exp_comp', sep = '\t')