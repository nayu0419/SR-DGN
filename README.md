# Deciphering spatial domains from spatially resolved tran-scriptomics through spatially regularized deep graph networks

![](./Fig_1.png)

## Overview
SR-DGN utilizes graph attention networks (GATs) to adaptively aggregate gene expression in-formation from neighboring spots or cells. We introduced spatial regularization constraints in SR-DGN to ensure the consistency of neighborhood relationships between the original space and embedding space. Additionally, SR-DGN employs cross-entropy (CE) loss to learn gene expression states, mitigating the impact of noisy gene dropouts. We evaluated the performance of SR-DGN across multiple datasets. The results demonstrate that SR-DGN can accurately identify tissue structures in SRT data of varying resolutions. Overall, SR-DGN is a powerful tool for precise spatial domain identification.

## Requirements
You'll need to install the following packages in order to run the codes.
* pandas~=1.3.5
* numpy~=1.21.5
* scanpy~=1.9.3
* matplotlib~=3.5.2
* scikit-learn~=1.0.2
* scipy~=1.7.3
* seaborn~=0.12.0
* torch~=1.11.0
* tqdm~=4.64.1
* R==4.2.1

## Tutorial
 Reproduce the result of article.

- olfactory bulb(Stereo-seq):
  ```python  
  python  run_mob.py
  ```

- hippocampus:

  ```python  
  python  run_slide.py
  ```

- somatosensory cortex embryos:

  ```python  
  python embryo.py
  ```

- olfactory bulb(Slide-seqV2):

  ```python  
  python run_mob2.py
  ```


## Contact
Feel free to submit an issue or contact us at gaorui@sdu.edu.cn for problems about the packages.
