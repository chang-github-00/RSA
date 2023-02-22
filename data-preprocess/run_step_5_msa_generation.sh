# proteinnet
python data-preprocess/create_msa.py --database pfam --dataset proteinnet --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset proteinnet --type test --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset proteinnet --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1

# secondary_structure
python data-preprocess/create_msa.py --database pfam --dataset secondary_structure --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset secondary_structure --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset secondary_structure --type casp12 --cpu_num 64 --path tape --iterations 3 --cutoff 1

# stability
python data-preprocess/create_msa.py --database pfam --dataset stability --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset stability --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset stability --type test --cpu_num 64 --path tape --iterations 3 --cutoff 1

#remote_homology
python data-preprocess/create_msa.py --database pfam --dataset remote_homology --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset remote_homology --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset remote_homology --type test_family_holdout --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset remote_homology --type test_fold_holdout --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset remote_homology --type test_superfamily_holdout --cpu_num 64 --path tape --iterations 3 --cutoff 1

#flourescence
python data-preprocess/create_msa.py --database pfam --dataset fluorescence --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset fluorescence --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset fluorescence --type test --cpu_num 64 --path tape --iterations 3 --cutoff 1


#subcellular_localization
python data-preprocess/create_msa.py --database pfam --dataset subcellular_localization --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset subcellular_localization --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset subcellular_localization --type test --cpu_num 64 --path tape --iterations 3 --cutoff 1

#human_ppi
python data-preprocess/create_msa.py --database pfam --dataset human_ppi --type train --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset human_ppi --type valid --cpu_num 64 --path tape --iterations 3 --cutoff 1
python data-preprocess/create_msa.py --database pfam --dataset human_ppi --type test --cpu_num 64 --path tape --iterations 3 --cutoff 1