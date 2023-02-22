# Data Preprocess

## Preprocessed files


## Building MSA features for TAPE
file organization 
```
|-----------------------------
|- datapreprocess
|---- create_msa_query.py
|---- create_msa.py
|---- create_msa_dataset.py
|- databases
|---- pfamA_32.0
|---- UniRef30_2020_06_hhsuite
|- tape
|---- data
|--------secondary_structure
|--------proteinnet
|--------remote_homology
|--------fluorescence
|--------stability
|---- query
|---- msa
|-----------------------------
```

1. setup tool hhblits 
```
pip install -r data-preprocess/requirements.txt

git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"

# install via conda
# conda install -c conda-forge -c bioconda hhsuite (这个命令暂时失效了)
```
2. obtain databases
```
mkdir databases
cd databases
pfam(15G) : 
    wget http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/pfamA_32.0.tar.gz 
    tar -zxvf pfamA_32.0.tar.gz 
uniclust30(50G) : 
    wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz 
    tar -zxvf UniRef30_2020_06_hhsuite.tar.gz
```

3. download tape data
```
mkdir tape
cd tape
mkdir data
cd data

#Download Data Files
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/proteinnet.tar.gz
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz
wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz

tar -xzf secondary_structure.tar.gz 
tar -xzf proteinnet.tar.gz 
tar -xzf remote_homology.tar.gz 
tar -xzf fluorescence.tar.gz 
tar -xzf stability.tar.gz 

rm secondary_structure.tar.gz
rm proteinnet.tar.gz
rm remote_homology.tar.gz
rm fluorescence.tar.gz
rm stability.tar.gz
```

4. generate tape queries for msa generation
```
python data-preprocess/create_msa_query.py --path tape
```

5. generate tape msa files with hhblits
```
dataset=proteinnet # remote_homology stability secondary_structure fluorescence 
split=train # test valid 
# note that the test split for secondary_structure is named as casp12

# python data-preprocess/create_msa.py --database uniclust --dataset ${dataset} --type ${split} --cpu_num 64 --path tape --iterations 1 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset ${dataset} --type ${split} --cpu_num 64 --path tape --iterations 3 --cutoff 1
```

6. generate training files with msa

```
python data-preprocess/create_msa_dataset.py --path 'tape'
```


## Building Dense Retrieval MSA

```
bash retrieval_benchmark/run_create_retrieval_msa.sh
```