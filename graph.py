import matplotlib.pyplot as plt
import pandas as pd

col_list= ['#FFC000','#87CDEE','#91A95B','#5986A7','#D9757D','#E08800','#D9757D','#AE967C','#9185CD','#3BBE3E','#1064AC','#38AFB5','#5B9BD5','#ED7D31','#A 5A5A5']


means = res.cv_results_['mean_test_score']
stds = res.cv_results_['std_test_score']
params = res.cv_results_['params']

mean_data = {'batch_size':batch_size}
for i in epochs:
    mean_data[str(i)]=[]
for i in epochs:
    for j in params:
        for k in batch_size:
            if j['batch_size']==k and j['epochs']==i:
                ind_ = params.index(j)
                mean_data[str(i)].append(means[ind_])
                
std_data = {'batch_size':batch_size}
for i in epochs:
    std_data[str(i)]=[]
for i in epochs:
    for j in params:
        for k in batch_size:
            if j['batch_size']==k and j['epochs']==i:
                ind_ = params.index(j)
                std_data[str(i)].append(stds[ind_])

plt.figure(figsize=(10,5))
pos = list(range(len(batch_size)))
width = 0.15

for i in range(0,len(epochs)):
    j = epochs[i]

    plt.bar([p + width*(i+2) for p in pos],
            mean_data[str(j)],
            width,
            yerr = std_data[str(j)],
            alpha=0.95,
            color=col_list[i])

plt.xlabel('batch_size',alpha=0.7)
plt.ylabel('accuracy',alpha=0.7)
plt.xticks([p + len(epochs) * width for p in pos],batch_size,alpha=0.7)
plt.ylim([0.5, 1])

plt.legend(epochs,title='epoch',loc='center left',bbox_to_anchor=(1, 0.88))
plt.show()

mean_data = {'dropout_rate':dropout_rate}
for i in weight_constraint:
    mean_data[str(i)]=[]
for i in weight_constraint:
    for j in params:
        for k in dropout_rate:
            if j['dropout_rate']==k and j['weight_constraint']==i:
                ind_ = params.index(j)
                mean_data[str(i)].append(means[ind_])

std_data = {'dropout_rate':dropout_rate}
for i in weight_constraint:
    stds_data[str(i)]=[]
for i in weight_constraint:
    for j in params:
        for k in dropout_rate:
            if j['dropout_rate']==k and j['weight_constraint']==i:
                ind_ = params.index(j)
                std_data[str(i)].append(std[ind_])
    
plt.figure(figsize=(10,5))
pos = list(range(len(dropout_rate)))
width = 0.15

for i in range(0,len(weight_constraint)):
    j = weight_constraint[i]

    plt.bar([p + width*(i+3.5) for p in pos],
            mean_data[str(j)],
            width,
            yerr = std_data[str(j)],
            alpha=0.9,
            color=col_list[i])

plt.xlabel('dropout_rate',alpha=0.7)    
plt.ylabel('accuracy',alpha=0.7)
plt.xticks([p + len(weight_constraint) * width for p in pos],dropout_rate,alpha=0.7)

plt.ylim([0.5, 1])

plt.legend(weight_constraint,title='weight_contraint',loc='center left',bbox_to_anchor=(1, 0.8))
plt.show()

# 1. Learning rate
means = res_learn_rate.cv_results_['mean_test_score']
stds = res_learn_rate.cv_results_['std_test_score']

num_bars = len(learn_rate)
positions = range(1,num_bars+1)

plt.figure(figsize=(10,5))
plt.bar(positions,means,yerr=stds,align='center',color='#FFC000',alpha=0.9)
plt.xticks(positions,learn_rate,alpha=0.7)
plt.xlabel('learn_rate',alpha=0.7)
plt.ylabel('accuracy',alpha=0.7)
plt.ylim([0.5,1])
plt.show()

# 3. Activation function
means = res_act_func.cv_results_['mean_test_score']
stds = res_act_func.cv_results_['std_test_score']

num_bars = len(activation)
positions = range(1,num_bars+1)

plt.figure(figsize=(10,5))
plt.bar(positions,means,yerr=stds,align='center',color='#FFC000',alpha=0.9)
plt.xticks(positions,activation,alpha=0.7)
plt.xlabel('activation',alpha=0.7)
plt.ylabel('accuracy',alpha=0.7)
plt.ylim([0.5,1])
plt.show()

# 4. dropout
means = res_dropout.cv_results_['mean_test_score']
stds = res_dropout.cv_results_['std_test_score']
params = res_dropout.cv_results_['params']

mean_data = {'dropout_rate':dropout_rate}
for i in weight_constraint:
    mean_data[str(i)]=[]
for i in weight_constraint:
    for j in params:
        for k in dropout_rate:
            if j['dropout_rate']==k and j['weight_constraint']==i:
                ind_ = params.index(j)
                mean_data[str(i)].append(means[ind_])

std_data = {'dropout_rate':dropout_rate}
for i in weight_constraint:
    std_data[str(i)]=[]
for i in weight_constraint:
    for j in params:
        for k in dropout_rate:
            if j['dropout_rate']==k and j['weight_constraint']==i:
                ind_ = params.index(j)
                std_data[str(i)].append(stds[ind_])
    
plt.figure(figsize=(10,5))
pos = list(range(len(dropout_rate)))
width = 0.15

for i in range(0,len(weight_constraint)):
    j = weight_constraint[i]

    plt.bar([p + width*(i+3.5) for p in pos],
            mean_data[str(j)],
            width,
            yerr = std_data[str(j)],
            alpha=0.9,
            color=col_list[i])

plt.xlabel('dropout_rate',alpha=0.7)    
plt.ylabel('accuracy',alpha=0.7)
plt.xticks([p + len(weight_constraint) * width for p in pos],dropout_rate,alpha=0.7)

plt.ylim([0.5, 1])

plt.legend(weight_constraint,title='weight_contraint',loc='center left',bbox_to_anchor=(1, 0.8))
plt.show()

# 5. network num
means = res_neurons.cv_results_['mean_test_score']
stds = res_neurons.cv_results_['std_test_score']

num_bars = len(neurons)
positions = range(1,num_bars+1)

plt.figure(figsize=(10,5))
plt.bar(positions,means,yerr=stds,align='center',color='#FFC000',alpha=0.9)
plt.xticks(positions,neurons,alpha=0.7)
plt.xlabel('neurons',alpha=0.7)
plt.ylabel('accuracy',alpha=0.7)
plt.ylim([0.5,1])
plt.show()
