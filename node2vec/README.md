# S3GC


to run Node2Vec on different dataset, you need to run node2vec.py by adding the name of dataset at the end of command.

all datasets are: {'Cora', 'Citeseer', 'Pubmed','ogbn-arxiv' , 'reddit','ogbn-products'}

So for different datasets please run this commands:

cora          : ./python3 node2vec.py cora

citeseer      : ./python3 node2vec.py citeseer

pubmed        : ./python3 node2vec.py pubmed

ogbn-arxiv    : ./python3 node2vec.py ogbn-arxiv

reddit        : ./python3 node2vec.py reddit

ogbn-products : ./python3 node2vec.py ogbn-products

DataSet:
you can find all necessary datasets in this drive:
https://drive.google.com/drive/folders/17uN67e6uZiOlzFeloU7t_DoOZezBt0Yp

These datasets must be deposited on datasets folder one level back to S3GC folder. (../datasets/)

Don't worry, if you don't donwload or the code doesnt find the datasets, it will just donwload them.



