# # Install rpy2
# !pip install rpy2

# # Import rpy2's interface to R
# import rpy2.robjects.packages as rpackages
# from rpy2.robjects.vectors import StrVector

# # Define the package names
# packnames = ('Ecume',)

# # Install the packages
# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages(StrVector(packnames))

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as rpyn

from rpy2.robjects.packages import importr

r_Ecume = importr('Ecume')

def weighted_ks_2samp(x1, x2, w1=None, w2=None):
    """
    Compute weighted Kolmogorov-Smirnov two-sample test using R library (Ecume)
    :return statistic
    :return pvalue
    """
    robjects.r.assign('x1', rpyn.numpy2rpy(x1))
    robjects.r.assign('x2', rpyn.numpy2rpy(x2))

    robjects.r("""w1 <- rep(1, length(x1))"""
               ) if w1 is None else robjects.r.assign('w1', rpyn.numpy2rpy(w1))
    robjects.r("""w2 <- rep(1, length(x2))"""
               ) if w2 is None else robjects.r.assign('w2', rpyn.numpy2rpy(w2))
    

    robjects.r("""ks_result <- ks_test(x=as.matrix(x1), y=as.matrix(x2),
           w_x=as.matrix(w1), w_y=as.matrix(w2), thresh = 0)""")
    ks_result = robjects.globalenv['ks_result']
    statistic = rpyn.rpy2py_floatvector(ks_result[0])[0]
    pvalue = rpyn.rpy2py_floatvector(ks_result[1])[0]
    return statistic, pvalue


import numpy as np
import pandas as pd


def create_df_for_cdf(List_data,List_labels,Weights):
    
    list_df = []
    for i in range(len(List_labels)):
        df_i = pd.DataFrame({
            'Label': [List_labels[i]] * List_data[i].shape[0],
            'Data': List_data[i],
            'Weight': Weights[i],
        })
        list_df.append(df_i)
    df_data = pd.concat(list_df)
    df_data.reset_index(inplace=True,drop=True)

    return df_data

def create_df_for_lineplot(List_data,List_labels):
    
    list_df = []
    for i in range(len(List_labels)):
        t = np.arange(0, -0.05*List_data[i].shape[0], -0.05)
        t = np.flip(t)
        df_i = pd.DataFrame({
            'Label': [List_labels[i]] * List_data[i].shape[0],
            't': t,
            'y': np.flip(List_data[i]),
        })
        list_df.append(df_i)
    df_data = pd.concat(list_df)
    df_data.reset_index(inplace=True,drop=True)

    return df_data


import seaborn as sns
import matplotlib.pyplot as plt

def sns_comparison(data, #dataframe
                   x, # key 
                   hue,
                   type='hist',
                   ax=None,
                   y=None,
                   bins=None,
                   weights=None,
                   element='poly',
                   loc='best',
                   loc_outside=False,
                   palette='Set1',
                   common_bins=True,
                   xlim=None,
                   ylim=None,
                   labels=None,
                   xlabel=None,
                   ylabel=None,
                   legend_off=False,
                   xlabel_off=False,
                   ylabel_off=False,
                   xaxis_off=False,
                   yaxis_off=False,
                   xticks_off=False,
                   yticks_off=False,
                   xticklabels_off=False,
                   yticklabels_off=False,
                   xlabelfontsize=18,
                   ylabelfontsize=18,
                   ticksfontsize=18,
                   legendfontsize=16,
                   figsize=None,
                   fig_name=None):
    if type == 'ecdf':
        h = sns.ecdfplot(data=data,
                         x=x,
                         hue=hue,
                         weights=weights,
                         palette=palette,
                         linewidth=1.5)

    elif type == 'hist':
        if bins:
            h = sns.histplot(ax=ax,
                             data=data,
                             x=x,
                             y=y,
                             weights=weights,
                             stat='percent',
                             hue=hue,
                             bins=bins,
                             element=element,
                             common_bins=common_bins,
                             common_norm=False,
                             palette=palette,
                             linewidth=1.5)
        else:
            if weights:
                print('Input bins is missing when using weights.')
                exit()
            h = sns.histplot(ax=ax,
                             data=data,
                             x=x,
                             y=y,
                             stat='percent',
                             hue=hue,
                             element=element,
                             common_bins=common_bins,
                             common_norm=False,
                             palette=palette,
                             linewidth=1.5)
    else:
        print('This function only supports ecdf and hist plots.')

    if hue is not None:
        if legend_off:
            h.axes.get_legend().remove()
        else:
            h.axes.get_legend().set_title(None)
            h.axes.get_legend().get_title().set_fontsize(legendfontsize)
            if loc_outside:
                sns.move_legend(h,
                                loc,
                                bbox_to_anchor=(1, 1),
                                frameon=False,
                                fontsize=legendfontsize)
            else:
                sns.move_legend(h, loc, frameon=False, fontsize=legendfontsize)
            if labels:
                # replace labels
                for t in h.axes.get_legend().texts:
                    t.set_text(labels[t.get_text()])
    if not ax:
        ax = h.axes
    ax.tick_params(labelsize=ticksfontsize)
    if xlabel_off:
        ax.set_xlabel(None)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabelfontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabelfontsize)
    if xlabel_off:
        ax.set_xlabel(None)
    if ylabel_off:
        ax.set_ylabel(None)
    if xticklabels_off:
        h.axes.get_xaxis().set_ticklabels([])
    if yticklabels_off:
        h.axes.get_yaxis().set_ticklabels([])
    if xticks_off:
        h.axes.get_xaxis().set_ticks([])
    if yticks_off:
        h.axes.get_yaxis().set_ticks([])
    if xaxis_off:
        h.axes.spines['top'].set_visible(False)
    if yaxis_off:
        h.axes.spines['right'].set_visible(False)
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=300)


from matplotlib.lines import Line2D
def sns_ks(data, #dataframe
           x = 't', # time 
           y = 'y', # value
           hue = 'Label',
           ax=None,
           loc='best',
           loc_outside=False,
           palette='Set1',
           xlim=None,
           ylim=None,
           labels=None,
           xlabel=None,
           ylabel=None,
           legend_off=False,
           xlabel_off=False,
           ylabel_off=False,
           xaxis_off=False,
           yaxis_off=False,
           xticks_off=False,
           yticks_off=False,
           xticklabels_off=False,
           yticklabels_off=False,
           xlabelfontsize=18,
           ylabelfontsize=18,
           ticksfontsize=18,
           legendfontsize=16,
           figsize=None,
           fig_name=None):

    h = sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        linewidth=1.5
    )
    plt.hlines(
        y = [0.05],
        xmin = -5, xmax = 0,
        linestyle='--', linewidth=1.5,
       #  label = "threshold"
        )
    custom_lines = [Line2D([0], [0],linestyle='--', linewidth=1.5)]
    h.axes.get_xaxis().set_ticks([-5,-4,-3,-2,-1,0])
    h.axes.get_yaxis().set_ticks([0,0.2,0.4,0.6,0.8,1])
    
    if hue is not None:
        if legend_off:
            h.axes.get_legend().remove()
        else:
            ## add label "Threshold"
            h.legend(handles=h.legend_.legendHandles + custom_lines, labels=[t.get_text() for t in h.legend_.get_texts()] + ['threshold'])
            
            h.axes.get_legend().set_title(None)
            h.axes.get_legend().get_title().set_fontsize(legendfontsize)
            if loc_outside:
                sns.move_legend(h,
                                loc,
                                bbox_to_anchor=(1, 1),
                                frameon=False,
                                fontsize=legendfontsize)
            else:
                sns.move_legend(h, loc, frameon=False, fontsize=legendfontsize)
            if labels:
                # replace labels
                for t in h.axes.get_legend().texts:
                    t.set_text(labels[t.get_text()])
    if not ax:
        ax = h.axes
    ax.tick_params(labelsize=ticksfontsize)
    if xlabel_off:
        ax.set_xlabel(None)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabelfontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabelfontsize)
    if xlabel_off:
        ax.set_xlabel(None)
    if ylabel_off:
        ax.set_ylabel(None)
    if xticklabels_off:
        h.axes.get_xaxis().set_ticklabels([])
    if yticklabels_off:
        h.axes.get_yaxis().set_ticklabels([])
    if xticks_off:
        h.axes.get_xaxis().set_ticks([])
    if yticks_off:
        h.axes.get_yaxis().set_ticks([])
    if xaxis_off:
        h.axes.spines['top'].set_visible(False)
    if yaxis_off:
        h.axes.spines['right'].set_visible(False)
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=300)
