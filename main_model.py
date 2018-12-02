from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
from keras import losses
import keras
import time
from time import strftime


##########################################################################################
#init_param
opt_param = dict(optimizer = 'RMSprop',
                 activation = 'relu',
                 dropout_rate = 0.25,
                 neurons = 512,
                 batch_size = 100,
                 epochs = 50,
                 learn_rate = 0.001,
                 weight_constraint = 1,
                 init_mode = 'normal')
# img_load
def load_img(categories):
    x = []
    y = []
    
    for idx, cat in enumerate(categories):
        dir = "c:/data/Dataset/trainset/"+cat
        files = glob.glob(dir+"/*.jpg")
        label = [0 for i in range(nb_class)]
        label[idx] = 1
        
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB") 
            img = img.resize((img_size, img_size))
            data = np.array(img)
            x.append(data)                                              
            y.append(label)
            
            for angle in range(-20,20,5):
                img2 = img.rotate(angle)
                data = np.array(img2)
                x.append(data)
                y.append(label)
                # img2.save('c:/data/Dataset/trainset/total/'+str(cn)+'.jpg')
    
                img3 = img2.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.array(img3)
                x.append(data)
                y.append(label)
                # img3.save('c:/data/Dataset/trainset/total/'+str(cn)+'.jpg')
    return np.array(x),np.array(y)

def build_model(optimizer = opt_param['optimizer'],
                activation = opt_param['activation'],
                dropout_rate = opt_param['dropout_rate'],
                neurons = opt_param['neurons'],
                weight_constraint = opt_param['weight_constraint'],
                learn_rate = opt_param['learn_rate'],
                init_mode = opt_param['init_mode']):
    model = Sequential()
    
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X.shape[1:]))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(Convolution2D(128, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten()) 
    model.add(Dense(neurons,
                    kernel_initializer = init_mode,
                    kernel_constraint = maxnorm(weight_constraint)))
    model.add(Activation(activation))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    
    # compile
    model.compile(loss = 'binary_crossentropy',
                  optimizer = RMSprop(lr=learn_rate),
                  metrics = ['accuracy'])
    return model

def gridsearch(x,y,model,parameter):
    grid = GridSearchCV(estimator = model, param_grid = parameter)
    grid_res = grid.fit(x,y)
    print(grid_res.best_score_,grid_res.best_params_)
    for i in parameter.keys():
        opt_param[i] = grid_res.best_params_[i]
    print(parameter.keys(),'---------OPTIMIZED!')
    return grid_res
##########################################################################################

start = strftime("%y%m%d-%H%M%S")
print(start)
f = open('start.txt', 'w')
print(start, file=f)
f.close()


##########################################################################################
##########################################################################################
# dataset
#img load
categories = ["갈비","등심","부채살","살치","안심","양지","채끝"]
nb_class = len(categories)
img_size = 128

X,Y = load_img(categories)
x_train, x_test, y_train, y_test = train_test_split(X, Y)

#scale
x_train = x_train/256
x_test = x_test/256

###################################################################
# 파라미터 최적화
###################################################################

model = build_model()

# Batch_size, epochs 선정
m = KerasClassifier(build_fn=build_model)
batch_size = [10,20,40,60,80,100]
epochs = [10,30,50,70,90,100]
param = dict(batch_size = batch_size,epochs = epochs)

res_batch_epoch = gridsearch(x_train,y_train,m,param)
print(res_batch_epoch.cv_results_)
 
f = open('res_batch_epoch.txt', 'w')
print(res_batch_epoch.cv_results_, file=f)
f.close()

# define test model
m = KerasClassifier(build_fn = build_model,
                    epochs = opt_param['epochs'],
                    batch_size = opt_param['batch_size'])

# 1. Learning rate
m = KerasClassifier(build_fn = build_model,
                    epochs = opt_param['epochs'],
                    batch_size = opt_param['batch_size'])

learn_rate = [0.00001, 0.00003, 0.00005, 0.00008, 0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3]
param = dict(learn_rate = learn_rate)

res_learn_rate = gridsearch(x_train,y_train,m,param)
print(res_learn_rate.cv_results_)

f = open('res_learn_rate.txt', 'w')
print(res_learn_rate.cv_results_, file=f)
f.close()

# 2. Neural net weight init
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param = dict(init_mode = init_mode)

res_initializer = gridsearch(x_train,y_train,m,param)
print(res_initializer.cv_results_)


f = open('init_mode.txt', 'w')
print(init_mode.cv_results_, file=f)
f.close()


# 3. Activation function
m = KerasClassifier(build_fn = build_model,
                    epochs = opt_param['epochs'],
                    batch_size = opt_param['batch_size'])

activation = ['softmax', 'softplus', 'softsign', 'relu', 'selu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param = dict(activation = activation)

res_act_func = gridsearch(x_train,y_train,m,param)
print(res_act_func.cv_results_)

f = open('res_act_func.txt', 'w')
print(res_act_func.cv_results_, file=f)
f.close()
####
# 4. dropout
m = KerasClassifier(build_fn = build_model,
                    epochs = opt_param['epochs'],
                    batch_size = opt_param['batch_size'])

weight_constraint = [0, 1, 2, 3, 4,5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param = dict(dropout_rate = dropout_rate, weight_constraint = weight_constraint)

res_dropout = gridsearch(x_train,y_train,m,param)
print(res_dropout.cv_results_)

f = open('res_dropout.txt', 'w')
print(res_dropout.cv_results_, file=f)
f.close()

# 5. network num 
m = KerasClassifier(build_fn = build_model,
                    epochs = opt_param['epochs'],
                    batch_size = opt_param['batch_size'])

neurons = [8,16,32,64,128,256,512,1024,2048]
param = dict(neurons = neurons)

res_neurons = gridsearch(x_train,y_train,m,param)
print(res_neurons.cv_results_)

f = open('res_neurons.txt', 'w')
print(res_neurons.cv_results_, file=f)
f.close()
    
################################################################## 
# 모델 예측
################################################################### 
    
opt_param

model = build_model()
model.fit(x_train, y_train,
          batch_size=opt_param['batch_size'], 
          nb_epoch=10)

print(model.summary())

# 정확도 측정
## trainset
eval_model = model.evaluate(x_train,y_train)
eval_model

## test set
score = model.evaluate(x_test, y_test)

print('loss=', score[0])
print('accuracy=', score[1])  


test = model.predict(x_test)

for i,v in enumerate(test):
    pre_ans = v.argmax()
    ans = y_test[i].argmax()
    #dat = x_test[i]
    if ans == pre_ans: 
        continue
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)


# classification
X_pred = []
files = glob.glob('c:/data/Dataset/eval/*.jpg')

for i,f in enumerate(files):
    img = Image.open(f)
    img = img.convert('RGB')
    img = img.resize((img_size,img_size))
    data = np.asarray(img)
    X_pred.append(data)  
    
X_pred = np.array(X_pred)
X_pred
pred = model.predict(X_pred)
print(pred)

for i in model.predict(X_pred):
    print(categories[np.argmax(i)])


######
end = strftime("%y%m%d-%H%M%S")
print(end)
f = open('end.txt', 'w')
print(end, file=f)
f.close()
