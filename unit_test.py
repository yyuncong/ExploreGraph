import numpy as np
import random

prediction = np.arange(20)


keep_indices = random.sample(range(20),15)

filter_index = random.sample(range(15),10)

print(prediction)

print(prediction[keep_indices][filter_index])

print(prediction[[keep_indices[idx] for idx in filter_index]])