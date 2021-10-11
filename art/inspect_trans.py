import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt


data0 = pd.read_csv('AI_Art3.csv')
data1 = pd.read_csv('AI_Art1.csv')
data = pd.concat([data0,data1],axis=0)
cols = data.columns
cm_data =data[cols[5:]].to_numpy()
sh_cm_data = np.shape(cm_data)
xy=int(sh_cm_data[1]**(.5))
cm_data_3d =np.zeros((sh_cm_data[0],xy,xy))
cm_acc =[]

for i in range(sh_cm_data[0]):
    temp = cm_data[i]
    temp = temp[~np.isnan(temp)]
    xy = int(len(temp) ** (.5))
    temp = temp.reshape(xy, xy)
    tp = np.nansum(np.diag(temp))
    fp = np.tril(temp,-1)
    fn = np.triu(temp,-1)
    fp_fn = np.nansum(fp)+np.nansum(fn)
    cm_data_3d[i,:xy,:xy] = temp
    cm_acc.append((tp/(tp+fp_fn)))


f_data = cm_data_3d.reshape(sh_cm_data)
data=data[cols[:5]]
for i in range(np.shape(f_data)[1]):
    data.insert(5+i,cols[5+i],f_data[:,i],True)
data.insert(5,'Acc',cm_acc,True)

colss = data.columns
mean = []
mea =[]
max = []
for i in pd.unique(data['Data Split']):
    g = data[data['Data Split']==i]
    for j in pd.unique(g['Model']):
        tg = g[g['Model'] == j]
        me = tg[colss[5:]].mean(numeric_only=True)
        mea.append([j,i])
        ma = tg[tg['Acc']==tg['Acc'].max()]
        mean.append(me)
        max.append(ma.iloc[0])
mean = pd.DataFrame(mean)
mea = pd.DataFrame(mea,columns=['Model','Data Split'])
max = pd.DataFrame(max)
mean = pd.concat([mea,mean],axis=1)


def plot_confusion_matrix(data, labels, output_filename,Titl,form):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(figsize=(28,16))

    plt.title("Confusion Matrix " + Titl,fontsize=20)

    seaborn.set(font_scale=1.2)
    ax = seaborn.heatmap(data, annot=True, fmt = form , cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")
    #plt.show()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()



colm = mean.columns
colM = max.columns
labs = [str(i) for i in range(20)]
for i in range(15):
    x = max.iloc[i]
    cm = x[colM[6:]].to_numpy(dtype=float).reshape((20,20))
    name = 'max'+x[colM[0]]+ x[colM[1]]
    titl = 'Max'+x[colM[0]]+ " " + x[colM[1]] +" Accuracy: "+ str(x[colM[5]])
    plot_confusion_matrix(data=cm, labels=labs, output_filename=name,Titl=titl,form= "g")
    x = mean.iloc[i]
    cm = x[colm[3:]].to_numpy(dtype=float).reshape((20, 20))
    name = 'mean'+x[colm[0]] + x[colm[1]]
    titl = 'Mean ' + x[colm[0]] + " " + x[colm[1]] + " Accuracy: " + str(x[colm[2]])
    plot_confusion_matrix(data=cm, labels=labs, output_filename=name,Titl = titl, form= ".2g")