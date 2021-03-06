import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import math
import re
import os

def load_rows(path_to_csv):
    df = pd.read_csv(path_to_csv,
                     sep=',',
                     quotechar='"',
                     doublequote=True,
                     header=0,
                     encoding='utf-8')
    # start counting at 1
    df.index += 1
    return df

def split_list(category_papers_field):
    """Given a text-field of paper indices (e.g. "1, 5, 7"),
       split out the paper indices as a list of integers.
       Perform a minimal amount of error correction like accounting for
       multiple or no spaces between numbers.
    """
    if type(category_papers_field) == float and math.isnan(category_papers_field):
        return []
    s = category_papers_field.replace(",", " ").replace(";", " ")
    s = re.sub("\s+", " ", s).strip()
    ints = [int(x) for x in s.split()]
    return ints

def create_heatmap(papers, categories):
    # rows of paper IDs, class, and titles -- the first three columns
    paper_rows = papers.iloc[:,0:4] #[papers.keys()[0:3]]
    # rows of category IDs (e.g P10)
    category_identifiers = categories.iloc[:,0]

    # for each X category out of the n, parse out the papers in the string
    # (e.g "1, 5, 6, ..."), and set the heatmap field for that (paper, Xn) to 1
    category_rows = [[0 for i in category_identifiers] for j in range(paper_rows.shape[0])]
    for IDX,(cat_ID,desc,indice_string) in categories.iterrows():
        paper_indices = split_list(indice_string)
        for paper_index in paper_indices:
            category_rows[paper_index-1][IDX-1] = 1

    category_map = pd.DataFrame.from_records(category_rows, columns=category_identifiers)
    category_map.index += 1

    # concatenate the rows of paper ids+titles with the rows of the heatmap
    # ID paper R1/R2 XC1 XC2 XC3 ...
    # 1   ABC   R1    1   0   1  ...
    # ...
    full_map = pd.concat([paper_rows, category_map], axis=1)
    return full_map

def load(path_to_papers_csv, path_to_category_csv):
    """Return the heatmap for the papers in a category (e.g problems)"""
    papers = load_rows(path_to_papers_csv)
    categories = load_rows(path_to_category_csv)
    heatmap = create_heatmap(papers, categories)
    return heatmap

def split_into_R1_R2(heatmap):
    R1 = heatmap.loc[heatmap["R1/R2"] == "R1"]
    R2 = heatmap.loc[heatmap["R1/R2"] == "R2"]
    return R1,R2

def split_on_outcome(heatmap):
    success = heatmap.loc[heatmap["Overall outcome"] == "Success"]
    partial = heatmap.loc[heatmap["Overall outcome"] == "Partial success"]
    failure = heatmap.loc[heatmap["Overall outcome"] == "Failure"]
    no_result = heatmap.loc[heatmap["Overall outcome"] == "No Result"]
    return success, partial, failure, no_result

def savefig(heatmap, path, **kwargs):
    fig = sns.heatmap(heatmap, **kwargs)
    fig.get_figure().savefig(path)
    # clear figure
    plt.clf()
    return fig

def save_heatmap_vertical(heatmap, path, ylabel, **kwargs):
    fig = sns.heatmap(heatmap, cmap=sns.cm.rocket_r, **kwargs)
    fig.xaxis.tick_top()
    fig.set_ylabel(ylabel)
    fig.get_figure().savefig(path)
    # clear figure
    plt.clf()
    return fig

def add_bar_labels(ax, vertical=True, bar_label_fontsize=8):
    """Adds a value at the top of the bar showing the size of it"""
    handles, labels = ax.get_legend_handles_labels()

    # In order to support stacked bar charts by only add a value for the
    # highest point (as stacked rectangles are counted individually),
    # store each bars identifier along with the size of the bar, and
    # keep the max size.
    max_values = {}
    neginf = -float('inf') # Placeholder for the first occurence of each key
                           # that is always replaced
    for i in ax.patches:
        if vertical:
            x = i.xy[0]
            y = i.xy[1] + i.get_height()
            max_values[x] = max(max_values.get(x, neginf), y)
        else: # Do the same except for horizontal bar charts.
            x = i.xy[0] + i.get_width()
            y = i.xy[1]
            max_values[y] = max(max_values.get(y, neginf), x)

    # Perform a second pass, this time fetching the sizes
    for i in ax.patches:
        # Only add a label once per set of (stacked) bars
        used = set()
        if vertical:
            x = i.xy[0]
            if x in used:
                continue
            used.add(x)
            y = max_values[x]
            # Avoid floating point checks on equality (e.g as keys), even if
            # it might be fine. Hence, do the offset to center the label here
            # after it has been used as a key.
            x += i.get_width()*0.5
            ha,va = 'center','bottom'
            size=y
        else: # Do the same except for horizontal bar charts.
            y = i.xy[1]
            if y in used:
                continue
            used.add(y)
            x = max_values[y]
            y += i.get_height()*0.5
            ha,va = 'left','center'
            size=x

        ax.text(x=x,
                y=y,
                s="{}".format(int(size)),
                fontsize=bar_label_fontsize,
                ha=ha,
                va=va)

def save_stacked_bar_chart(frame, path, xlabel, ylabel, **kwargs):
    frame = frame.T
    fig = frame.plot.bar(stacked=True, **kwargs)
    fig.set_xlabel(xlabel, labelpad=4)
    fig.set_ylabel(ylabel, labelpad=5)
    add_bar_labels(ax=fig, vertical=True)

    fig.get_figure().savefig(path)
    plt.clf()
    return fig

def save_horizontal_bar_chart(frame, path, xlabel, ylabel, **kwargs):
    fig = frame.plot.barh(**kwargs)
    fig.invert_yaxis()
    fig.xaxis.tick_top()
    fig.set_xlabel(xlabel)
    fig.set_ylabel(ylabel)
    fig.xaxis.set_label_position('top')
    add_bar_labels(ax=fig, vertical=False)

    fig.get_figure().savefig(path)
    plt.clf()
    return fig

def main(output_directory):
    plt.style.use('seaborn-white')
    custom_cmap = matplotlib.colors.ListedColormap(["blue", "green", "red"], name='from_list', N=None)
    #{ ass:[full_map, R1_map, R2_map],
    #  prob:[...],
    #  ...}
    maps = {"problem" : [load("papers.csv", "pcat.csv")],
            "assumption" : [load("papers.csv", "acat.csv")],
            "error" : [load("papers.csv", "ecat.csv")]
            }

    # create separate maps based on outcome
    outcome_maps = {"problem": [],
                    "assumption": [],
                    "error": []}
    for key in maps:
        heatmap = maps[key][0]
        outcomes = list(split_on_outcome(heatmap))
        outcome_maps[key].append(pd.concat(outcomes))
        outcome_maps[key].extend(split_on_outcome(heatmap))
        del heatmap["Overall outcome"]

    # add the maps where R1 and R2 are separate
    for key in maps:
        heatmap = maps[key][0]
        maps[key].extend(split_into_R1_R2(heatmap))
    # [assumption, problem, error]
    for overall_category in maps:
        # 1D version, i.e only one row where each column is the number of
        # occurences for that X category across the papers
        # [both, R1, R2]
        for division,divtag in zip(maps[overall_category], ["both", "R1", "R2"]):
            # skip the first three columns with paper title etc
            heatmap = pd.DataFrame(division.iloc[:,3:].sum(axis=0)).T
            # all rows are collapsed into one, so give it an appropriate label
            heatmap.index=["Collated count"]
            filename = "{}_{}_{}".format(
                "1D",
                overall_category,
                divtag)
            path = os.path.join(output_directory, filename)
            savefig(heatmap , path, square=True, annot=True)

        # generate bar charts
        r1_map = pd.DataFrame(maps[overall_category][1].iloc[:,3:].sum(axis=0)).T
        r1_map.index = ["R1"]
        r2_map = pd.DataFrame(maps[overall_category][2].iloc[:,3:].sum(axis=0)).T
        r2_map.index = ["R2"]
        combined_map = pd.concat([r1_map, r2_map])
        for division, divtag in zip([combined_map, r1_map, r2_map], ["both", "R1", "R2"]):
            filename = "{}_{}_bar".format(
                overall_category,
                divtag)
            path = os.path.join(output_directory, filename)
            save_stacked_bar_chart(division, path, overall_category.capitalize() + " Category", "Count")

        # 1D version, R1 and R2 normalized to number of papers
        # [R1, R2]
        R1 = pd.DataFrame(maps[overall_category][1].iloc[:,3:])
        R2 = pd.DataFrame(maps[overall_category][2].iloc[:,3:])
        weighted_R1 = pd.DataFrame(R1.sum(axis=0)/R1.shape[0]).T
        weighted_R2 = pd.DataFrame(R2.sum(axis=0)/R2.shape[0]).T

        # set the center value of the heatmap based on the category with the
        # highest count
        max1,max2 = weighted_R1.values.max(), weighted_R2.values.max()
        center = max(max1, max2)*0.5

    # plot problems, assumptions, and errors per paper
    #   bar plot
    category_maps = []
    for overall_category in maps:
        frame = pd.DataFrame(maps[overall_category][0].iloc[:,3:].sum(axis=1))
        # rename column
        frame.columns = [overall_category.capitalize()]
        category_maps.append(frame)
    frames = pd.concat(category_maps, axis=1)
    filename = "papers_bar.png"
    output_path = os.path.join(output_directory, filename)
    save_horizontal_bar_chart(frames, output_path, "Count", "Article", width=0.5, figsize=(6,12))

    #   plot heatmap
    filename = "papers_heatmap.png"
    output_path = os.path.join(output_directory, filename)
    for column in frames:
        max_value = frames[column].max()
        frames[column] = frames[column].divide(max_value)
    frames = frames.round(2)
    save_heatmap_vertical(frames, output_path, "Article", annot=False)

    # plot problems, assumptions, and errors per paper, grouped by type
    #   bar plot
    frame = pd.DataFrame(maps["problem"][0])
    sorted = frame.sort_values(['R1/R2', 'ID'])
    ids = sorted['ID']
    category_maps = [ids]
    for overall_category in maps:
        frame = pd.DataFrame(maps[overall_category][0])
        sorted = frame.sort_values(['R1/R2', 'ID'])
        frame = pd.DataFrame(sorted.iloc[:, 3:].sum(axis=1))
        # rename column
        frame.columns = [overall_category.capitalize()]
        category_maps.append(frame)
    frames = pd.concat(category_maps, axis=1)
    del frames['ID']
    filename = "papers_type_bar.png"
    output_path = os.path.join(output_directory, filename)
    save_horizontal_bar_chart(frames, output_path, "Count", "Article", width=0.5, figsize=(6, 12))


    # generate box plot
    frames = []
    for outcome_category in outcome_maps:
        frame = pd.DataFrame(outcome_maps[outcome_category][0])
        outcome_column = frame["Overall outcome"].apply(lambda row: row + " - " + outcome_category.capitalize())
        frame = pd.DataFrame(frame.iloc[:, 3:].sum(axis=1))
        frame = pd.concat([outcome_column, frame], axis=1)
        frames.append(frame)
    frames = pd.concat(frames)
    frames = frames.rename(index=str, columns={0:"Count"})
    frames['Overall outcome'] = pd.Categorical(frames['Overall outcome'], ["Success - Problem", "Success - Assumption", "Success - Error",
                                                                         "Partial success - Problem", "Partial success - Assumption", "Partial success - Error",
                                                                         "Failure - Problem", "Failure - Assumption", "Failure - Error",
                                                                         "No Result - Problem", "No Result - Assumption", "No Result - Error"])
    sorted = frames.sort_values(['Overall outcome'])
    ax = sns.boxplot(y="Overall outcome", x="Count", data=sorted)
    output_path = os.path.join(output_directory, "boxplot.png")
    ax.get_figure().savefig(output_path, bbox_inches='tight')
    # clear figure
    plt.clf()


    # plot problems, assumptions, and errors per paper, grouped by outcome
    #   bar plot
    outcomes = ["Success", "Partial success", "Failure", "No Result"]
    frame = pd.DataFrame(outcome_maps["problem"][0])
    nr_rows = frame.shape[0]
    # insert dummy rows for additional spacing between categories
    for outcome in outcomes:
        row = [frame.shape[0] + 1, "", "", outcome] + [0]*(frame.shape[1] - 4)
        frame.loc[-1] = row
        frame.index += 1
    # sort on outcome
    frame['Overall outcome'] = pd.Categorical(frame['Overall outcome'], outcomes)
    sorted = frame.sort_values(['Overall outcome', 'ID'])
    ids = sorted['ID']
    category_maps = [ids]
    for overall_category in outcome_maps:
        frame = pd.DataFrame(outcome_maps[overall_category][0])
        for outcome in outcomes:
            row = [frame.shape[0] + 1, "", "", outcome] + [0] * (frame.shape[1] - 4)
            frame.loc[-1] = row
            frame.index += 1
        frame['Overall outcome'] = pd.Categorical(frame['Overall outcome'], ["Success", "Partial success", "Failure", "No Result"])
        sorted = frame.sort_values(['Overall outcome', 'ID'])
        frame = pd.DataFrame(sorted.iloc[:, 3:].sum(axis=1))
        # rename column
        frame.columns = [overall_category.capitalize()]
        category_maps.append(frame)
    frames = pd.concat(category_maps, axis=1)
    data = []
    data.insert(0, {'ID': 0, 'Problem': 0, 'Assumption': 0, 'Error': 0})
    frames = pd.concat([pd.DataFrame(data), frames])
    frames.set_index('ID', inplace=True)
    columns = ['Problem','Assumption','Error']
    frames = frames[columns]

    # generate correct tick labels
    yticks = []
    indices = frames.index.tolist()
    for i in range(frames.shape[0]):
        if indices[i] == 0 or indices[i] > nr_rows:
            yticks.append("")
        else:
            yticks.append(indices[i])
    filename = "papers_outcome_bar.png"
    output_path = os.path.join(output_directory, filename)
    fig = frames.plot.barh(width=0.6, figsize=(6, 10))
    fig.invert_yaxis()
    fig.set_yticklabels(yticks)
    fig.xaxis.tick_top()
    fig.set_xlabel("Count")
    # hardcoded label to fit figure size
    fig.set_ylabel("              No Result                                          Failure                                                Partial Success                      Success")
    fig.xaxis.set_label_position('top')
    fig.get_figure().savefig(output_path)
    plt.clf()

if __name__ == '__main__':
    main(output_directory="output_figures")
