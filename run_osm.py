import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
import SR_DGN
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stpython/lib/R'

adata = sc.read('Data/OSM/osmFISH_codeluppi2018spatial_cortex_data.h5ad')
adata.var_names_make_unique()
adata.obsm['spatial'][:, 1] *= -1
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.scale(adata, zero_center=False, max_value=10)


SR_DGN.Cal_Spatial_Net(adata, k_cutoff=6,model='KNN')
SR_DGN.Stats_Spatial_Net(adata)
adata = SR_DGN.train(adata)

sc.pp.neighbors(adata, use_rep='SR_DGN')
sc.tl.umap(adata)
adata = SR_DGN.mclust_R(adata, used_obsm='SR_DGN', num_cluster=11)

obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df["Region"])
print('Adjusted rand index = %.5f' % ARI)



