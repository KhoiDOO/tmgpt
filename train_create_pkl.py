from tokenizer import tokenize
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_in", type=str, required=True)
parser.add_argument("--pkl_out", type=str, required=True)
parser.add_argument("--augmentation", action='store_true')
parser.add_argument("--forever", action='store_true')

args = parser.parse_args()

MESH_IN = args.mesh_in
PKL_OUT = args.pkl_out

QUANT_BIT = 7
N_TRIAL = 10
MAX_N_FACES = 5500
AUGMENTATION = args.augmentation

mesh_fns = os.listdir(MESH_IN)

if not os.path.exists(PKL_OUT):
    os.makedirs(PKL_OUT)

while True:
    for fn in mesh_fns:
        mesh_path = os.path.join(MESH_IN, fn)
        io_dict = tokenize(mesh_path, QUANT_BIT, N_TRIAL, MAX_N_FACES, AUGMENTATION)
        
        if io_dict is not None:
            out_path = os.path.join(PKL_OUT, fn[:-4] + ".pkl")
            with open(out_path, 'wb') as f:
                pickle.dump(io_dict, f)
    
    if not args.forever:
        break   