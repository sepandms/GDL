# Graph Deep Learning @ Università della Svizzera Italiana
## S3GC Project SP 2023
Navdeep Singh Bedi, Sepehr Beheshti, Cristiano Colangelo

### S3GC code
to run S3GC on different dataset, you need to run S3GC.py by adding the name of dataset at the end of command.

all datasets are: {'Cora', 'Citeseer', 'Pubmed','ogbn-arxiv' , 'reddit','ogbn-products'}

So for different datasets please run this commands:


cora          : `./python3 S3GC.py cora`

citeseer      : `./python3 S3GC.py citeseer`

pubmed        : `./python3 S3GC.py pubmed`

ogbn-arxiv    : `./python3 S3GC.py ogbn-arxiv`

reddit        : `./python3 S3GC.py reddit`

ogbn-products : `./python3 S3GC.py ogbn-products`

DataSet:
you can find all necessary datasets in this drive:
https://drive.google.com/drive/folders/17uN67e6uZiOlzFeloU7t_DoOZezBt0Yp

These datasets must be deposited on datasets folder one level back to S3GC folder. (../datasets/)

Don't worry, if you don't donwload or the code doesnt find the datasets, it will just donwload them.

### DGI Code

cora          : `./python3 dgi_cora.py`

citeseer      : `./python3 dgi_citseer.py`

pubmed        : `./python3 dgi_pubmed.py pubmed`

ogbn-arxiv    : `./python3 dgi_arxiv.py ogbn-arxiv`

reddit        : `./python3 S3GC.py reddit`

ogbn-products : `./python3 S3GC.py ogbn-products`

### Node2Vec code