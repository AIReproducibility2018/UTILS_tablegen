from gentable import load_rows
import os
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import numpy as np

def df_to_tex(df, coltitles_to_wrap, savepath, escape=True):
    wrap_these = lambda x: x.lower() in [y.lower() for y in coltitles_to_wrap]
    latex_col_formats = ['X' if wrap_these(col) else 'l' for col in df.columns]

    tex = df.to_latex(index=False, column_format=''.join(latex_col_formats), escape=escape)
    tex = tex.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
    tex = tex.replace("\\end{tabular}", "\\end{tabularx}")

    #print(tex)
    with open(savepath, 'w') as f:
        print("Writing to {}".format(savepath))
        f.write(tex)


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

        filename = "{}.tex".format(name)
        savepath = os.path.join("output_figures", filename)
        df_to_tex(df, coltitles_to_wrap, savepath)


    all_papers_dir = 'all_papers_used'
    for csv in os.listdir(all_papers_dir):
        if csv[-4:] != '.csv':
            continue

        inpath = os.path.join(all_papers_dir, csv)
        df = load_rows(inpath)
        df['Citation'] = df['Citation'].map(lambda x: "{%s}" % x)

        coltitles_to_wrap = ['Title']
        new_filename = "{}.tex".format(csv)
        savepath = os.path.join("all_papers_used", new_filename)
        df_to_tex(df, coltitles_to_wrap, savepath, escape=False)


if __name__ == '__main__':
    main()