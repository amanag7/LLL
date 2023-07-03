import numpy as np
import pandas as pd

df = pd.read_json('../../data/wikisql_to_squad-train-v2.0.json')

print(df['data'][0])