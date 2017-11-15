def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

from norms import OrigN, OrigS, SpectralNorm, WbysqDist, BinarNorm
from sklearn.feature_selection import VarianceThreshold

import os
import pandas as pd
import numpy as np

from MetricTransform import MetricTransform

from reskit.core import DataTransformer, MatrixTransformer, Pipeliner

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




param_LMNN = {'model': 'LMNN',
              'mode': 'inner_product',
                'max_iter': 100,
              'min_iter':10,
              'learn_rate':  1e-7,
              'regularization': 0.5,
              'conv_tol': 1e-5,
              'k': 5,
             }
normalizers1 = [('wbysqdist', WbysqDist())]
normalizers2 = [('spectral', SpectralNorm())]

featurizers = [('degrees', MatrixTransformer(degrees))]

selection = [('vartreshold', VarianceThreshold()),
            ('origS', OrigS())]

featurizers = [('degrees', MatrixTransformer(degrees)),]


pipe_lmnn = Pipeline(steps=[('metric', MetricTransform(**param_LMNN)), ('clf', SVC(kernel='precomputed'))])

metric_trans_learn = [('LMNN', pipe_lmnn)]


steps = [('normalization1', normalizers1),
         ('normalization2', normalizers2),
         ('features', featurizers),
         ('selection', selection),
         ('machine_learning', metric_trans_learn)]

param_grid = {'LMNN': {'metric__max_iter':[100, 500],
                       'metric__min_iter':[10],
                  'metric__learn_rate': np.linspace(1e-10, 1e-2, 3),
                  'metric__regularization': [0.1, 0.25, 0.65, 0.9],
                  'metric__conv_tol': [0.001, 1e-5],
                  'metric__k': [3, 15, 30, 55],
                  'clf__C':[1e-7, 1e-2, 0.1, 1, 10 ]},
    }
    


grid_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
eval_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

pipeliner = Pipeliner(steps=steps, grid_cv=grid_cv, 
                      eval_cv=eval_cv, param_grid=param_grid)


table = pipeliner.get_results(X, y, scoring=['roc_auc'], caching_steps = ['normalization1','normalization2','features','selection'], logs_file = 'log/ucla_results_LMNN_inn_spectral.log')

table.to_csv('ucla_results_LMNN_inn_spectral', sep = '\t')