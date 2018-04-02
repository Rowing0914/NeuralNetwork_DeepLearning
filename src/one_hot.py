import numpy as np
import random

n_labels = 5
_list = list(range(5))
random.shuffle(_list)
print(np.eye(n_labels)[_list])

import pandas as pd
df = pd.DataFrame(["paris", "paris", "tokyo", "amsterdam"], columns=['city'])
print(pd.get_dummies(df, columns=['city']))