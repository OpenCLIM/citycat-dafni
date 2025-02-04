import pandas as pd
import os


class Flow:
    """Flow time series

    Args:
        data: Pandas DataFrame containing flow values

    """
    def __init__(self, data: pd.Series):
        assert type(data) == pd.Series
        assert len(data) > 0, 'Flow Series is empty'
        self.data = data

    def write(self, path):
        with open(os.path.join(path, 'Flow_BC.flw'), 'w') as f:
            f.write('* * *\n')
            f.write('* * * discharge per unit width (m3/s/m)* * *\n')
            f.write('* * *\n')
            f.write('{}\n'.format(len(self.data)))
            f.write('* * *\n')
            self.data.to_csv(f, sep=' ', header=False, lineterminator='\n')
            #self.data.to_csv(f, sep=' ', header=False)
