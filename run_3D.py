import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import scipy.linalg
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
import SR_DGN
os.environ['R_HOME'] = '/home/zw/software/miniconda3/envs/stpython/lib/R'
adata1 = sc.read('Data/Ms_Brain_Aging/MsBrainAgingSpatialDonor_3_0.h5ad')
adata1.var_names_make_unique()
sc.pp.scale(adata1, zero_center=False, max_value=10)
sc.pl.embedding(adata1, alpha=1, color="tissue", legend_fontsize=18, basis="spatial",save="3_1")
adata2 = sc.read('Data/Ms_Brain_Aging/MsBrainAgingSpatialDonor_3_1.h5ad')
adata2.var_names_make_unique()
sc.pp.scale(adata2, zero_center=False, max_value=10)
sc.pl.embedding(adata2, alpha=1, color="tissue", legend_fontsize=18, basis="spatial",save="3_2")
SR_DGN.Cal_Spatial_Net(adata1, rad_cutoff=60)
SR_DGN.Cal_Spatial_Net(adata2, rad_cutoff=60)
adata = ad.concat([adata1,adata2])
adata.uns['Spatial_Net'] = pd.concat([adata1.uns['Spatial_Net'], adata2.uns['Spatial_Net']])
adata = SR_DGN.train(adata)
adata = SR_DGN.mclust_R(adata, used_obsm='SR-DGN', num_cluster=8)


