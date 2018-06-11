from gentable import load_rows
import os
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import numpy as np


# This code is awful.
def main():
    pd.set_option('display.max_colwidth', 0)
    names = ['pcat.csv', 'acat.csv', 'ecat.csv', 'papers.csv']

    for name in names:
        df = load_rows(name)

        tex = df.to_latex(index=False, column_format='X'*len(df.columns))
        tex = tex.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
        tex = tex.replace("\\end{tabular}", "\\end{tabularx}")
        filename = "{}.tex".format(name)
        path = os.path.join("output_figures", filename)
        print(tex)
        with open(path, 'w') as f:
            f.write(tex)

if __name__ == '__main__':
    main()