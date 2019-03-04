#%%
import random
import os
from shutil import copy2
#split_folders.ratio('custom_data/*', output="output", seed=1337, ratio=(.8, .2))

#%%
input_folder = './asm_data'
out_folder = list(0 for i in range(3))
out_folder[0] = './asm_data_alt'	# Train
out_folder[1] = './asm_data_val_alt'	# Validation
out_folder[2] = './asm_data_test_alt'	# Test
ratio = [1.0, 0.0, 0.0] # Train : Val : Test
assert sum(ratio) == 1.0, "Sum of ratio not 1"
#%%
def safe_make_dir(path):
	if not os.path.exists(out_dir[i]):
		os.makedirs(out_dir[i])

random.seed(1337) # For reproducability of results
for root, dirs, files in os.walk(input_folder):
	out_dir = list(0 for i in range(3))
	for f in files:
		file_name = os.path.join(root, f)
		#print(file_name)
		tmp = root.split(os.sep)
		for i in range(3):
			out_dir[i] = os.path.join(out_folder[i], tmp[-2] + '_' + tmp[-1])
			print(out_dir[i])
			safe_make_dir(out_dir[i])
		rand = random.uniform(0, 1)
		if rand >= ratio[1] + ratio[2]:
			copy2(file_name, out_dir[0])
		elif rand < ratio[1] + ratio[2] and rand > ratio[2]:
			copy2(file_name, out_dir[1])
		else:
			copy2(file_name, out_dir[2])

#%%
