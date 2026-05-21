import numpy as np
import time

# Create 1 million data points using standard Python and NumPy
# This satisfies your random simulation data requirement
np.random.seed(42)
raw_data_list = list(np.random.rand(1000000))
raw_data_array = np.array(raw_data_list)

print("Data generation successful. 1 Million data points ready!")


start_time = time.time()

# Processing using a standard manual loop
loop_results = []
for value in raw_data_list:
    loop_results.append((value * 2.5) + 10.8)

loop_time = time.time() - start_time
print(f"Standard Python Loop took: {loop_time:.6f} seconds")

start_time = time.time()

# Processing using NumPy Vectorization (No loops!)
vectorized_results = (raw_data_array * 2.5) + 10.8

numpy_time = time.time() - start_time
print(f"NumPy Vectorized operation took: {numpy_time:.6f} seconds")

# Calculate the speed difference to prove mastery
speed_factor = loop_time / numpy_time
print(f"\n🚀 NumPy is {speed_factor:.2f}x faster than standard Python lists!")
