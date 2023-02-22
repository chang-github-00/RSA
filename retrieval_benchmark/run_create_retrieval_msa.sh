# subcellular_localization
python create_retrieval_seqs.py --k 500 --dataset subcellular_localization --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset subcellular_localization --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset subcellular_localization --path ../tape --type test --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors


#proteinnet
python create_retrieval_seqs.py --k 500 --dataset proteinnet --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset proteinnet --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset proteinnet --path ../tape --type test --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors


#secondary_structure
python create_retrieval_seqs.py --k 500 --dataset secondary_structure --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset secondary_structure --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset secondary_structure --path ../tape --type casp12 --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors

#remote_homology
python create_retrieval_seqs.py --k 500 --dataset remote_homology --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset remote_homology --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset remote_homology --path ../tape --type test_fold_holdout --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors

# stability
python create_retrieval_seqs.py --k 500 --dataset stability --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset stability --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset stability --path ../tape --type test --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors

#human_ppi
python create_retrieval_seqs.py --k 500 --dataset human_ppi --path ../tape --type train --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset human_ppi --path ../tape --type valid --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
python create_retrieval_seqs.py --k 500 --dataset human_ppi --path ../tape --type test --faiss_index ../retrieval-db/pfam.index --dstore_seqs ../retrieval-db/pfam_seq.txt --dstore_labels ../retrieval-db/pfam_label.txt --dstore_fvecs ../retrieval-db/pfam_vectors
