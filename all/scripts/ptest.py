import numpy as np

split_rate = '.7,.2,.1'
split_rate = np.array([float(a) for a in split_rate.split(',')]).sum().round()

print(split_rate)

print(split_rate == 1.0)