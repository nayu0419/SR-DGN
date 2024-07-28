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
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stagate/lib/R'
# 七层 小鼠嗅球 2
# input_dir = 'Data/Slide-seqV2_MoB'
# counts_file = os.path.join(input_dir, 'Puck_200127_15.digital_expression.txt')
# coor_file = os.path.join(input_dir, 'Puck_200127_15_bead_locations.csv')
#
# counts = pd.read_csv(counts_file, sep='\t', index_col=0)
# coor_df = pd.read_csv(coor_file, index_col=3)
# print(counts.shape, coor_df.shape)
#
# adata = sc.AnnData(counts.T)
# adata.var_names_make_unique()
# coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
# adata.obsm["spatial"] = coor_df.to_numpy()
#
# used_barcode = pd.read_csv('Data/Slide-seqV2_MoB/used_barcodes.txt', sep='\t', header=None)
# used_barcode = used_barcode[0]
# adata = adata[used_barcode,]
#
# adata.write('Data/Slide-seqV2_MoB/mob2.h5ad')

adata = sc.read('Data/Slide-seqV2_MoB/mob2.h5ad')
adata.var_names_make_unique()
adata.obsm['spatial'][:, 1] *= -1
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.scale(adata, zero_center=False, max_value=10)

SR_DGN.Cal_Spatial_Net(adata, rad_cutoff=70)
SR_DGN.Stats_Spatial_Net(adata)
adata = SR_DGN.train(adata)

sc.pp.neighbors(adata, use_rep='SR-DGN')
sc.tl.umap(adata)
adata = SR_DGN.mclust_R(adata, used_obsm='SR-DGN', num_cluster=10)
adata.obs.to_csv("mob2.csv")
sc.pl.umap(adata, color="mclust", title='SR-DGN')
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.embedding(adata, basis="spatial", color="mclust", show=True, title='SR_DGN',save = "mob2")

sc.tl.paga(adata, groups='mclust')
plt.rcParams["figure.figsize"] = (4,3)
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,
                   title='SR_DGN', legend_fontoutline=2, show=True,save = "mob2")