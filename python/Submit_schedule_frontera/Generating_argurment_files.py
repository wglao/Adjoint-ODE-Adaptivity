import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--files", type=int, help="NODEs")
parser.add_argument("--GPU_per_node", type=int, help="NODEs")
args = parser.parse_args()

number_of_node = args.files
number_of_GPU_per_node = args.GPU_per_node

# files = ['complex', 'complex_no_refine', 'complex_no_refine_shallow', 'detect_complex', 'backtrack_complex']
# files = ['complex_no_refine', 'complex_no_refine_shallow']
files = ['new_loss']
seeds = [5,6,7,8]

LIST = []
for f in files:
  for s in seeds:
    line = '_' + str(f) + '.py --seed ' + str(s)
    LIST.append(line)

LIST2 = []
for file_i in range(number_of_node):
  for gpu_ind in range(number_of_GPU_per_node):
    if file_i*number_of_GPU_per_node + gpu_ind < len(LIST):
      line = 'cd $SCRATCH/adjoint/python; python3 Main' + LIST[
          file_i*number_of_GPU_per_node +
          gpu_ind] + ' --node ' + str(file_i +
                                      1) + ' --GPU_index ' + str(gpu_ind)
      # line = 'python ../Main_diffusion_based.py --node ' + str(
      #     file_i + 1) + ' --GPU_index ' + str(gpu_ind) + LIST[
      #         file_i*number_of_GPU_per_node + gpu_ind]
    if file_i*number_of_GPU_per_node + gpu_ind >= len(LIST):
      line = ' '

    LIST2.append(line)

for file_i in range(number_of_node):
  names = LIST2[file_i*number_of_GPU_per_node:(file_i+1)*number_of_GPU_per_node]

  with open(r'arguement_files' + str(file_i + 1), 'w') as fp:
    for item in names:
      fp.write("%s\n" % item)
  fp.close()
