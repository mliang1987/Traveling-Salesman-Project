import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def blaaah():
    script_dir = os.path.dirname(__file__)
    rel_path = '../output/LS1-SA'
    path = os.path.join(script_dir, rel_path)
    instances = ['Atlanta','Berlin','Boston','Champaign','Cincinnati','Denver','NYC','Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState','UMissouri']
    for instance in instances:
        for filename in glob.glob(os.path.join(path, '{}*.trace'.format(instance))):
            df = pd.read_csv(filename, header = None, names = ["Time", "Fit"])
            last_time, last_val = df.iloc[-1]
            print(instance, last_time, last_val)


if __name__ == "__main__":
    blaaah()