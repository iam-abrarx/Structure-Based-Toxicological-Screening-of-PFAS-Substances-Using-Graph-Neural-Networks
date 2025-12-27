import os
import pandas as pd

print(f"CWD: {os.getcwd()}")
root = '.'
target = os.path.join(root, '../data/processed/master_chem_table.csv')
abs_target = os.path.abspath(target)

print(f"Target logic: {target}")
print(f"Absolute Target: {abs_target}")
print(f"Exists? {os.path.exists(abs_target)}")

# List parent dir
parent = os.path.abspath('..')
print(f"Parent dir: {parent}")
print(f"Parent contents: {os.listdir(parent)}")

# check data dir
data_dir = os.path.join(parent, 'data')
if os.path.exists(data_dir):
    print(f"Data dir contents: {os.listdir(data_dir)}")
    proc_dir = os.path.join(data_dir, 'processed')
    if os.path.exists(proc_dir):
        print(f"Processed dir contents: {os.listdir(proc_dir)}")
