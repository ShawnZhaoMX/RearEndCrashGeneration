import numpy as np
import matplotlib.pyplot as plt

def extract_data_from_dfgen(df_gen,IDs,para,t=None):

    ## set filter
    filter_gen_IDs = [df_gen["id"].isin(IDs),df_gen["t"]>-4.84]
    if t:
        filter_gen_IDs.append(df_gen["t"]==t)
    filter_gen_IDs = np.all(filter_gen_IDs,axis=0)

    ## extract data from df based on filter
    df_gen_IDs = df_gen[filter_gen_IDs].reset_index(drop=True)

    ## select specific parameter
    if para in ['v_f','v_l','d','weight']:
        para_gen_IDs = df_gen_IDs[para].to_numpy()
        if not t: ## read entire timeserise from begin to crash
            if para == "weight":
                para_gen_IDs = para_gen_IDs[::97]
            elif para in ['v_f','v_l','d']:
                para_gen_IDs = para_gen_IDs.reshape(-1,97)
    elif para in ['label','qualify']:
        try:
            para_gen_IDs = df_gen_IDs[para].to_numpy()
            if not t: ## read entire timeserise from begin to crash
                para_gen_IDs = para_gen_IDs[::97]
        except:
            raise Exception(f"para{para} does not in df_gen")
    else:
        raise Exception("wrong para. (para: 'v_f','v_l','d','weight','label','qualify')")
        
    return para_gen_IDs


def idxSample(labels,cls,weights,traj_number = None):
    '''
    Sample indexes of a given clster based on weights. 
    The number of indexes is less than a given traj_number.
    '''
    indexes_cls = np.where(labels == cls)[0]
    weights_cls = weights[labels==cls]
    if traj_number == None:
        return indexes_cls
    else:
        if traj_number < indexes_cls.shape[0]:
            if weights_cls.shape[0]!=0:
                indexes_sampled = np.random.choice(indexes_cls, size=traj_number, replace=False, p=weights_cls/np.sum(weights_cls))
                return indexes_sampled
            else:
                return indexes_cls
        else:
            return indexes_cls

def calculateTrajMeanMaxMin(XX,weights,length):
    if weights.shape[0]!=0:
        mean_xx = np.average(XX,axis = 0, weights=weights)
        max_xx1, min_xx1 = np.max(XX[:,:length]), np.min(XX[:,:length])
        max_xx2, min_xx2 = np.max(XX[:,length:length*2]), np.min(XX[:,length:length*2])
        max_xx3, min_xx3 = np.max(XX[:,length*2:length*3]), np.min(XX[:,length*2:length*3])
    else:
        mean_xx = np.zeros(length*3)*np.nan
        max_xx1, min_xx1 = np.NaN, np.NaN
        max_xx2, min_xx2 = np.NaN, np.NaN
        max_xx3, min_xx3 = np.NaN, np.NaN
    return mean_xx, (max_xx1,max_xx2,max_xx3), (min_xx1,min_xx2,min_xx3)

def clsInspection(
        X,
        labels,
        weights,
        dimen = 3,
        cluster_selection = None,
        traj_number = None,
        normalized_results = False,
        line_transparency = 1,
        line_width = 0.2,
        line_width_ratio = 20,
        fig_name=None):
    '''
    cluster_selection: a list that contains Cluster for inspection (e.g. [1,2,4,5])
    traj_number: a int value, represents the maximum trajectory number for plotting
    '''
    ## set timestamp for plotting
    l = int(X.shape[1]/dimen)
    t = np.arange(0, -0.05 * l, -0.05)
    t = np.flip(t)

    D_norma = 1 if normalized_results else 0
        
    if cluster_selection == None:
        cluster_selection = list(np.unique(labels))
    else: # filter no exist cluster
        cluster_selection = [item for item in cluster_selection if item in np.unique(labels)]
        if len(cluster_selection) == 0:
            cluster_selection = list(np.unique(labels))
    n_clusters = len(cluster_selection)
    plt.figure(figsize=(n_clusters*3, 6))

    for i, cls in enumerate(cluster_selection):

        idx = idxSample(labels,cls,weights,traj_number)
        mean_xx, max_xx, min_xx = calculateTrajMeanMaxMin(X[labels == cls,:,D_norma],weights[labels == cls],length=l)

        plt.subplot(3, n_clusters, i + 1)
        for xx in X[idx,:l,D_norma]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[:l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        plt.title(f"Cluster: {cls}")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[0]-.3,max_xx[0]+.3])
        if i == 0:
            plt.ylabel("Lead Vehicle Velocity (m/s)")
        if cls == cluster_selection[-1]:
            plt.legend()

        plt.subplot(3, n_clusters, n_clusters + i + 1)
        for xx in X[idx,l:2*l,D_norma]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[l:2*l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[1]-.3,max_xx[1]+.3])
        if i == 0:
            plt.ylabel("Following Vehicle Velocity (m/s)")
        # plt.legend()

        plt.subplot(3, n_clusters, 2*n_clusters + i + 1)
        for xx in X[idx,2*l:3*l,D_norma]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[2*l:3*l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        plt.xlabel("time before crash (s)")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[2]-.3,max_xx[2]+.3])
        if i == 0:
            plt.ylabel("Distance (m)")
        # plt.legend()

    
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        plt.tight_layout()
    ## Save fig
    if fig_name:
        plt.savefig(fig_name, dpi=300)



def trajInspection(
        X,
        X_gen,
        labels,
        labels_gen,
        weights,
        weights_gen,
        dimen = 3,
        cluster_selection = None,
        traj_number = None,
        line_transparency = 1,
        line_width = 0.2,
        line_width_ratio = 20,
        fig_name=None):
    '''
    cluster_selection: a list that contains Cluster for inspection (e.g. [1,2,4,5])
    traj_number: a int value, represents the maximum trajectory number for plotting
    '''
    ## set timestamp for plotting
    l = int(X.shape[1]/dimen)
    t = np.arange(0, -0.05 * l, -0.05)
    t = np.flip(t)

    if cluster_selection == None:
        cluster_selection = list(np.unique(labels))
    else: # filter no exist cluster
        cluster_selection = [item for item in cluster_selection if item in np.unique(labels)]
        if len(cluster_selection) == 0:
            cluster_selection = list(np.unique(labels))
    n_clusters = len(cluster_selection)
    plt.figure(figsize=(n_clusters*3, 6))

    for i, cls in enumerate(cluster_selection):

        idx = idxSample(labels,cls,weights,traj_number)
        idx_gen = idxSample(labels_gen,cls,weights_gen,traj_number)
        mean_xx, max_xx, min_xx = calculateTrajMeanMaxMin(X[labels == cls,:,0],weights[labels == cls],length=l)
        mean_xx_gen, max_xx_gen, min_xx_gen = calculateTrajMeanMaxMin(X_gen[labels_gen == cls,:,0],weights_gen[labels_gen == cls],length=l)
        max_xx = tuple(map(max, zip(max_xx, max_xx_gen))) ## ylim max
        min_xx = tuple(map(min, zip(min_xx, min_xx_gen))) ## ylim min

        plt.subplot(3, n_clusters, i + 1)
        for xx in X[idx,:l,0]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[:l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        if idx_gen.shape[0]>0:
            for xx_gen in X_gen[idx_gen,:l,0]:
                plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width)
            plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width,label="Synthetic")
            plt.plot(t, mean_xx_gen[:l], "r-", alpha=1, linewidth = line_width*line_width_ratio,label="Synthetic, mean")
        plt.title(f"Cluster: {cls}")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[0]-.3,max_xx[0]+.3])
        if i == 0:
            plt.ylabel("Lead Vehicle Velocity (m/s)")
        if cls == cluster_selection[-1]:
            plt.legend()

        plt.subplot(3, n_clusters, n_clusters + i + 1)
        for xx in X[idx,l:2*l,0]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[l:2*l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        if idx_gen.shape[0]>0:
            for xx_gen in X_gen[idx_gen,l:2*l,0]:
                plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width)
            plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width,label="Synthetic")
            plt.plot(t, mean_xx_gen[l:2*l], "r-", alpha=1, linewidth = line_width*line_width_ratio,label="Synthetic, mean")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[1]-.3,max_xx[1]+.3])
        if i == 0:
            plt.ylabel("Following Vehicle Velocity (m/s)")
        # plt.legend()

        plt.subplot(3, n_clusters, 2*n_clusters + i + 1)
        for xx in X[idx,2*l:3*l,0]:
            plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width)
        plt.plot(t,xx, "b-", alpha=line_transparency, linewidth = line_width,label="Raw")
        plt.plot(t, mean_xx[2*l:3*l], "b-", alpha=1, linewidth = line_width*line_width_ratio,label="Raw, mean")
        if idx_gen.shape[0]>0:
            for xx_gen in X_gen[idx_gen,2*l:3*l,0]:
                plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width)
            plt.plot(t,xx_gen, "r-", alpha=line_transparency, linewidth = line_width,label="Synthetic")
            plt.plot(t, mean_xx_gen[2*l:3*l], "r-", alpha=1, linewidth = line_width*line_width_ratio,label="Synthetic, mean")
        plt.xlabel("time before crash (s)")
        plt.xticks(np.arange(-5, 0.1, 1))
        plt.xlim([-5.05,0.15])
        plt.ylim([min_xx[2]-.3,max_xx[2]+.3])
        if i == 0:
            plt.ylabel("Distance (m)")
        # plt.legend()

    
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        plt.tight_layout()
    ## Save fig
    if fig_name:
        plt.savefig(fig_name, dpi=300)



from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
def tsne_anslysis(dataset1,
                  dataset2,
                  index_cutoff_gen,
                  fig_name = None
                 ):
    
    '''
    dataset1: raw data
    dataset2: gen data
    '''

    combined_data = np.concatenate((dataset2, dataset1))

    tsne = TSNE(n_components=2, init='pca',learning_rate='auto')
    # tsne_result = tsne.fit_transform(combined_data_standardized)
    result = tsne.fit_transform(combined_data)

    n = 3
    # HUE = ["Synthetic"]*dataset2.shape[0]+["Raw"]*dataset1.shape[0]
    HUE = np.array(["Synthetic (Remaining)"]*dataset2.shape[0]+["Raw"]*dataset1.shape[0])
    HUE_newgen = HUE[:dataset2.shape[0]]
    HUE_newgen[~index_cutoff_gen] = "Synthetic (Removed)"
    HUE[:dataset2.shape[0]] = HUE_newgen

    df = pd.DataFrame()
    df['y'] = HUE
    df['comp-1'] = result[:, 0]
    df['comp-2'] = result[:, 1]
    df['weights'] = 1
    plt.clf()
    h = sns.scatterplot(x='comp-1',
                        y='comp-2',
                        hue=df.y.tolist(),
                        palette=sns.color_palette('Set1', n)[::-1],
                        data=df)

    h.axes.axis('off')
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=300)
    plt.show()
    return df



# def utils_plotClsResults6(XX,weights,traj_number,length):
#     idx = np.arange(XX.shape[0])
#     if traj_number != None:
#         if traj_number < XX.shape[0]:
#             np.random.shuffle(idx)
#             idx = idx[:traj_number]
#         alpha1 = 0.3
#     else:
#         alpha1 = 0.2
#     ## mean traj
#     if weights.shape[0]>0:
#         mean_xx = np.average(XX,axis = 0, weights=weights)
#         max_xx1, min_xx1 = np.max(XX[:,:length]), np.min(XX[:,:length])
#         max_xx2, min_xx2 = np.max(XX[:,length:length*2]), np.min(XX[:,length:length*2])
#         max_xx3, min_xx3 = np.max(XX[:,length*2:]), np.min(XX[:,length*2:])
#     else:
#         mean_xx = np.empty((XX.shape[1]))
#         max_xx1, min_xx1 = 0, 0
#         max_xx2, min_xx2 = 0, 0
#         max_xx3, min_xx3 = 0, 0
#     return idx, alpha1, mean_xx, (max_xx1,max_xx2,max_xx3), (min_xx1,min_xx2,min_xx3)

# def plotClsResults6(X,labels,weights,n_clusters,traj_number = None):
#     l = int(X.shape[1]/3)
#     t = np.arange(0, -0.05 * l, -0.05)
#     t = np.flip(t)
    
#     # n_clusters = 3
    
#     plt.figure(figsize=(n_clusters*3, 7))
#     # for i,cls in enumerate(range(n_clusters+1,n_clusters+n_clusters+1)):
#     for i,cls in enumerate(range(1,n_clusters+1)):
        
#         XX = X[labels == cls,:,0]
#         idx, alpha1, mean_xx, max_xx, min_xx = utils_plotClsResults6(XX,weights[labels == cls],traj_number,length=l)
        
#         plt.subplot(3, n_clusters, i + 1)
#         if XX.shape[0]>0:
#             for xx in XX[idx,:l]:
#                 plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8)
#             plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8,label="Raw")
#             plt.plot(t, mean_xx[:l], "b-", alpha=1, linewidth = 3,label="Raw, mean")
#         plt.title(f"Cluster: {cls}")
#         plt.xticks(np.arange(-5, 0.1, 1))
#         plt.xlim([-5.05,0.15])
#         plt.ylim([min_xx[0]-.3,max_xx[0]+.3])
#         if i == 0:
#             plt.ylabel("Lead Vehicle Velocity (m/s)")
#         plt.legend()

#         plt.subplot(3, n_clusters, n_clusters + i + 1)
#         if XX.shape[0]>0:
#             for xx in XX[idx,l:2*l]:
#                 plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8)
#             plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8,label="Raw")
#             plt.plot(t, mean_xx[l:2*l], "b-", alpha=1, linewidth = 3,label="Raw, mean")
#         plt.xticks(np.arange(-5, 0.1, 1))
#         plt.xlim([-5.05,0.15])
#         plt.ylim([min_xx[1]-.3,max_xx[1]+.3])
#         if i == 0:
#             plt.ylabel("Following Vehicle Velocity (m/s)")
#         plt.legend()

#         plt.subplot(3, n_clusters, 2*n_clusters + i + 1)
#         if XX.shape[0]>0:
#             for xx in XX[idx,2*l:]:
#                 plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8)
#             plt.plot(t,xx, "b-", alpha=alpha1, linewidth = 0.8,label="Raw")
#             plt.plot(t, mean_xx[2*l:], "b-", alpha=1, linewidth = 3,label="Raw, mean")
#         plt.xlabel("time before crash (s)")
#         plt.xticks(np.arange(-5, 0.1, 1))
#         plt.xlim([-5.05,0.15])
#         plt.ylim([min_xx[2]-.3,max_xx[2]+.3])
#         if i == 0:
#             plt.ylabel("Distance (m)")
#         plt.legend()
        
#     plt.tight_layout()