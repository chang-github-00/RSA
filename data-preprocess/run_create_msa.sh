cpuNum=64
echo $cpuNum

dataset=human_ppi
split=test

python data-preprocess/create_msa.py --database pfam --dataset $dataset --type $split --cpu_num $cpuNum --path tape --iterations 3 --cutoff 1