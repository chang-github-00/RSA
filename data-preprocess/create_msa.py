import argparse
import os
import subprocess
from tqdm import tqdm
import random

def arg_to_query_dir(path, dataset, type):
    newpath = path + "/query/" +dataset+'/'+ dataset + "_" + type
    print(newpath)
    assert os.path.exists(newpath)
    
    return newpath

def arg_to_output_dir(path, dataset, type):
    newpath = path + "/msa/" + dataset+'/'+dataset + "_" + type
    if not os.path.exists(path + "/msa/"):
        os.mkdir(path + "/msa/")
    if not os.path.exists(path + "/msa/"+dataset):
        os.mkdir(path + "/msa/"+dataset)   
    return newpath

def arg_to_summary_dir(path, dataset, type):
    newpath = path + "/msa/summary/" + dataset+'/'+dataset + "_" + type
    if not os.path.exists(path + "/msa/summary/"):
        os.mkdir(path + "/msa/summary/")
    if not os.path.exists(path + "/msa/summary/"+dataset):
        os.mkdir(path + "/msa/summary/"+dataset)   
    return newpath

        
DATABASE_DIR ={'uniclust': "databases/UniRef30_2020_06/UniRef30_2020_06",
               'pfam': "databases/pfam/pfam"
               }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, help='uniclust or pfam')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--type", type=str, help='train, valid or test')
    parser.add_argument("--cpu_num", type=int)
    parser.add_argument("--path", type=str, default='./', help="working directory")
    parser.add_argument("--iterations", type=int, default=1, help="number of search iterations")
    parser.add_argument("--cutoff", type=float, default=0.001, help="e-value cut off")
    args = parser.parse_args()
    return args


def generate_msa(args, path, dataset, type, cpu_num=64):
    database_dir = DATABASE_DIR[args.database]
    query_dir = arg_to_query_dir(path, dataset, type)
    output_dir = arg_to_output_dir(path, dataset, type)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    summary_dir = arg_to_summary_dir(path, dataset, type)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    all_files = os.listdir(query_dir)
    files = [ fname for fname in all_files if fname.endswith('.seq')]
    files_list = list(range(len(files)))
    random.shuffle(files_list)
    
    for i in tqdm(files_list):
        query_file = files[i]
        file_name = query_file.split('.')[0]
        input_file  = query_dir + '/' + query_file
        output_file = output_dir + '/' + file_name + '.a3m'
        if not os.path.exists(output_file):
            cmd = ["hhblits","-cpu", str(cpu_num),"-i",input_file,"-d",database_dir, "-oa3m", output_file,"-n",str(args.iterations), "-e", str(args.cutoff) ]
            print(cmd)
            output = subprocess.run(["hhblits","-cpu", str(cpu_num),"-i",input_file,"-d",database_dir, "-oa3m", output_file,"-n",str(args.iterations), "-e", str(args.cutoff) ], capture_output =True)
            if output.returncode == 0: 
                summary_file = summary_dir + '/' + file_name + '.summary'
                with open(summary_file, 'w') as f:
                    f.write(output.stdout.decode("utf-8"))
            else:
                print(output.stderr.decode("utf-8"))
                print("Error in hhblits")
                exit(1)

if __name__ == "__main__":
    args = get_args()
    generate_msa(args, args.path, args.dataset, args.type, args.cpu_num)

    