import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
import SR_DGN
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stagate/lib/R'

# adata = sc.read('Data/Mouse_hippocampus_Slide-seqV2/filtered_feature_bc_matrix_200115_08.h5ad')
# adata.var_names_make_unique()
#
# used_barcode = pd.read_csv('Data/Mouse_hippocampus_Slide-seqV2/used_barcodes.csv', sep=',', header=0, index_col=0)
# used_barcode = used_barcode["barcodes"]
# adata = adata[used_barcode,]
# adata.write('Data/Mouse_hippocampus_Slide-seqV2/qiu.h5ad')


adata = sc.read('Data/Mouse_hippocampus_Slide-seqV2/qiu.h5ad')
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.scale(adata, zero_center=False, max_value=10)




SR_DGN.Cal_Spatial_Net(adata, rad_cutoff=40)
SR_DGN.Stats_Spatial_Net(adata)
adata = SR_DGN.train(adata,device="cpu")


sc.pp.neighbors(adata, use_rep='SR-DGN')
sc.tl.umap(adata)
adata = SR_DGN.mclust_R(adata, used_obsm='SR-DGN', num_cluster=10)
adata.obs.to_csv("slide.csv")


sc.pl.umap(adata, color="mclust", title='SR_DGN')
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="mclust", show=True, title='SR_DGN',save = "slide")

sc.tl.paga(adata, groups='mclust')
plt.rcParams["figure.figsize"] = (4,3)
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,
                   title='SR_DGN', legend_fontoutline=2, show=True,save = "slide")