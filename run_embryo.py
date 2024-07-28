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
from SR_DGN.utils import refine_label
os.environ['R_HOME'] = '/home/dell/anaconda3/envs/stagate/lib/R'

slide_id =9.5
slide_id =10.5
slide_id =11.5
slide_id =12.5

adata = sc.read('Data/Mouse_embryo/E%.1f_E1S1.MOSTA.h5ad'%slide_id)
adata.var_names_make_unique()
adata.obsm['spatial'][:, 1] *= -1

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.scale(adata, zero_center=False, max_value=10)
sc.pp.log1p(adata)

SR_DGN.Cal_Spatial_Net(adata, rad_cutoff=1.4)
SR_DGN.Stats_Spatial_Net(adata)
adata = SR_DGN.train(adata,n_epochs=100,device="cpu")
k = len(adata.obs["annotation"].unique())

sc.pp.neighbors(adata, use_rep='SR-DGN')
sc.tl.umap(adata)
adata = SR_DGN.mclust_R(adata, used_obsm='SR-DGN', num_cluster=k)
new_type = refine_label(adata, radius=20, key='mclust')
adata.obs['refine'] = new_type

adata.obs.to_csv("embryo125.csv")
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df["annotation"])
print('Adjusted rand index = %.5f' % ARI)


ARI = adjusted_rand_score(obs_df['refine'], obs_df["annotation"])
print('refine Adjusted rand index = %.5f' % ARI)


sc.pl.umap(adata, color="mclust", title='SR_DGN')

title = 'SR_DGN(ARI=%.2f)'%ARI
ax = sc.pl.embedding(adata, alpha=1, color="mclust", legend_fontsize=18, show=False,basis="spatial",
                   size=100000 / adata.shape[0])

ax.set_title(title, fontsize=23)
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])
ax.axes.invert_yaxis()
plt.savefig("figures/embryo_E%.1f.pdf"%slide_id)
plt.close()
