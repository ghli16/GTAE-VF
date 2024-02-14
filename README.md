Guanghui Li, Peihao Bai, Jiao Chen, Cheng Liang,Identifying virulence factors using graph transformer autoencoder with ESMFold-predicted structures,
Computers in Biology and Medicine, 2024, 170: 108062.

## Code
### Environment Requirement
The code has been tested running under Python 3.8.16. The required packages are as follows:
- numpy == 1.23.5
- numpy-base == 1.23.5
- openfold == 1.0.0
- networkx == 3.1
- scipy == 1.10.1
- pytorch == 1.13.1
- pytorch-lightning == 1.5.10
- pytorch-cuda ==11.6

Files

1.dataset: dataset.fasta store the protein sequence information of the training set, validation set, and test set
2. src:        a.Model.py：the GTAE-VF framework； 
               b.graphT.py: the graph transformer framework;
               c.main.py: training model saves the optimal parameters of the model.



 

