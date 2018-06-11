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
        if name == 'papers.csv':
            # Move the year column right after article
            cols = df.columns.tolist()
            cols.remove('Year')
            idx = cols.index('Article')
            cols.insert(idx+1, 'Year')
            df = df[cols]

        coltitles_to_wrap = ['Article', 'Problem Category', 'Error Category', 'Assumption Category']
        wrap_these = lambda x: x.lower() in [y.lower() for y in coltitles_to_wrap]
        latex_col_formats = ['X' if wrap_these(col) else 'c' for col in df.columns]

        tex = df.to_latex(index=False, column_format=''.join(latex_col_formats))
        tex = tex.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
        tex = tex.replace("\\end{tabular}", "\\end{tabularx}")
        filename = "{}.tex".format(name)
        path = os.path.join("output_figures", filename)
        print(tex)
        with open(path, 'w') as f:
            f.write(tex)

if __name__ == '__main__':
    main()