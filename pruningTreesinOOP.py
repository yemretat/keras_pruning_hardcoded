# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from tensorflow.keras.callbacks import ModelCheckpoint
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
class PruningBase(ABC):
    def save_model(self,model:Sequential): ##pathi pycharm localine ver
        model.save('modelx.hdf5')
    
    def load_model(self):
        reconstructed_model=load_model('modelx.hdf5')
        return reconstructed_model

    def zipped_able_model_size(self,model:Sequential):
        import tensorflow_model_optimization as tfmot
        import tempfile
        import tensorflow as tf
        import os 
        import zipfile
        model_for_export = tfmot.sparsity.keras.strip_pruning(model)
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        _, zipped_file = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(pruned_keras_file)
        return os.path.getsize(zipped_file)
    
    def graph_visualization(self,pruning_percentages,allScoresToDisplay:object):
        from matplotlib import pyplot as plt
        from IPython import get_ipython

        get_ipython().run_line_magic('matplotlib', 'inline')
        plt.style.use('seaborn')
        fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3)
        fig.set_figheight(15)
        fig.set_figwidth(23)
        
        ax1.plot(pruning_percentages,allScoresToDisplay["Accuracy"] ,color='r',label='Accuracy Score')
        ax1.set_title('Model Accuracy Scores', weight='bold', fontsize=16)
        ax1.set_ylabel('Acurracy', weight='bold', fontsize=14)
        ax1.set_xlabel('Total Pruning Percentage of Tree', weight='bold', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.legend()

        ax2.plot(pruning_percentages,allScoresToDisplay["Loss"] ,color='r', label='Loss Scores')
        ax2.set_title('Categorical Crossentropy Loss Scores',weight='bold', fontsize=16)
        ax2.set_ylabel('Loss Scores', weight='bold', fontsize=14)
        ax2.set_xlabel('Total Pruning Percentage of Tree', weight='bold', fontsize=14)
        ax2.legend()

        ax3.plot(pruning_percentages,allScoresToDisplay["Total_Bytes"] ,color='r',label='Total Bytes')
        ax3.set_title('Model Size in Bytes', weight='bold', fontsize=16)
        ax3.set_ylabel('Model Size', weight='bold', fontsize=14)
        ax3.set_xlabel('Total Pruning Percentage of Tree', weight='bold', fontsize=14)
        ax3.ticklabel_format(style='plain')
        ax3.legend()
        
        plt.tight_layout()

        plt.show()
        return plt
        
      #  fig.saveFig('fig1.png')

    @abstractmethod
    def pruning_scores(self,pruningArray):
        pass

        


# %%
import pandas as pd

class WeightPruning(PruningBase):
    
    def __init__(self,trained_model:Sequential):
        self.__trained_model=trained_model
        self.__all_weights_sorted=None
        self.__total_No_Weights=None
        self.__total_No_Layers=len(trained_model.layers)
        self.__pruning_percentages=None
        super().save_model(trained_model)
        self.__allScoresToDisplay=None
    def __allWeights_Sorted(self):
        all_Weights={}
        for layer_no in range(self.__total_No_Layers-1): 
            layer_weights = (pd.DataFrame(self.__trained_model.layers[layer_no].get_weights()[0]).stack()).to_dict() 
            items=layer_weights.items()
            layer_weights = { (layer_no, k[0], k[1]): v for k, v in layer_weights.items() }
            all_Weights.update(layer_weights)
        self.__all_weights_sorted = {k: v for k, v in sorted(all_Weights.items(), key=lambda item: abs(item[1]))}
        self.__total_No_Weights=len(self.__all_weights_sorted)
      
    def pruning_scores(self, pruningPercentageArray,trained_model:Sequential,X_test_norm,Y_test_onehot):
        weight_pruning_scores = []
        total_bytes=[]
        loss_pruning_scores=[]
        self.__allWeights_Sorted()
        self.__pruning_percentages=pruningPercentageArray
        super().save_model(trained_model)
        for pruning_percent in self.__pruning_percentages:
            new_model=super().load_model()
            new_weights = trained_model.get_weights().copy()
            prune_fraction = pruning_percent/100
            number_of_weights_to_be_pruned = int(prune_fraction*self.__total_No_Weights)
            weights_to_be_pruned = {k: self.__all_weights_sorted[k] for k in list(self.__all_weights_sorted)[ :  number_of_weights_to_be_pruned]}   
            for k, v in weights_to_be_pruned.items():
                new_weights[k[0]][k[1], k[2]] = 0  
            for layer_no in range(self.__total_No_Layers - 1) :
                new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0], new_weights[layer_no].shape[1])
                new_model.layers[layer_no].set_weights(new_layer_weights)
            new_score  = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            loss_pruning_scores.append(new_score[0]) #categorical_crossentropy
            weight_pruning_scores.append(new_score[1]) # new_score[0] eklenecek
            bytes_size=super().zipped_able_model_size(new_model)
            total_bytes.append(bytes_size)
        
        self.__allScoresToDisplay={"Loss":loss_pruning_scores,"Accuracy":weight_pruning_scores,"Total_Bytes":total_bytes}
        graph = super().graph_visualization(self.__pruning_percentages,self.__allScoresToDisplay) 
        return graph

            
        
#dataları nasıl alacağız evaluate için ? 
# frontend nasıl düzenlenecek
        
        
    


# %%
import numpy as np

class NeuronPruning(PruningBase):
    
    def __init__(self,trained_model:Sequential):
        self.__total_No_Layers=len(trained_model.layers)
        self.__trained_model=trained_model
        self.__all_neurons_sorted=None
        self.__total_No_Neurons=None
        self.__pruning_percentages=None
    def __allNeurons_Sorted(self):
        all_neurons={}
        for layer_no in range(self.__total_No_Layers-1):
            layer_neurons = {}
            layer_neurons_df = pd.DataFrame(self.__trained_model.layers[layer_no].get_weights()[0])
            for i in range(len(layer_neurons_df.columns)):
                layer_neurons.update({ i : np.array( layer_neurons_df.iloc[:,i] ) })    
            
            layer_neurons = { (layer_no, k): v for k, v in layer_neurons.items() }
            all_neurons.update(layer_neurons)
        self.__all_neurons_sorted={k: v for k, v in sorted(all_neurons.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
        self.__total_No_Neurons=len(self.__all_neurons_sorted)
    def pruning_scores(self, pruningArray,trained_model:Sequential,X_test_norm,Y_test_onehot):
        neuron_pruning_scores = []
        total_bytes_neuron=[]
        loss_pruning_scores_neuron=[]
        self.__allNeurons_Sorted()
        self.__pruning_percentages=pruningArray
        super().save_model(trained_model)
        for pruning_percent in self.__pruning_percentages:
            new_model=super().load_model()
            new_weights = trained_model.get_weights().copy()
            
            prune_fraction = pruning_percent/100
            number_of_neurons_to_be_pruned = int(prune_fraction*self.__total_No_Neurons)
            neurons_to_be_pruned = {k: self.__all_neurons_sorted[k] for k in list(self.__all_neurons_sorted)[ : number_of_neurons_to_be_pruned]}     

            for k, v in neurons_to_be_pruned.items():
                new_weights[k[0]][:, k[1]] = 0
            
            for layer_no in range(self.__total_No_Layers - 1) :
                new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0], new_weights[layer_no].shape[1])
                new_model.layers[layer_no].set_weights(new_layer_weights)
            
            new_score  = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            neuron_pruning_scores.append(new_score[1])
            loss_pruning_scores_neuron.append(new_score[0]) #categorical_crossentropy
            bytes_size=super().zipped_able_model_size(new_model)
            total_bytes_neuron.append(bytes_size) 
        self.__allScoresToDisplay={"Loss":loss_pruning_scores_neuron,"Accuracy":neuron_pruning_scores,"Total_Bytes":total_bytes_neuron}
        graph = super().graph_visualization(self.__pruning_percentages,self.__allScoresToDisplay) 
        return graph
# %%
from tensorflow.keras.datasets import fashion_mnist

((X_train, Y_train), (X_test, Y_test)) = fashion_mnist.load_data()

class_labels = pd.Series(['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Code', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'])
labels_dict = class_labels.to_dict()
# %%
X_train_reshaped = X_train.reshape(len(X_train), -1)   
X_test_reshaped = X_test.reshape(len(X_test), -1)

X_train_norm = X_train_reshaped/255            
X_test_norm = X_test_reshaped/255

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

n_classes =  len(class_labels)


Y_train_onehot = to_categorical(Y_train, num_classes = n_classes)
Y_test_onehot = to_categorical(Y_test, num_classes = n_classes)

X_train_final, X_valid, Y_train_final, Y_valid = train_test_split(X_train_norm, Y_train_onehot, 
                                                                  test_size=0.16666)

print('Shape of data used for training, and shape of training targets : \n ', X_train.shape, ',', Y_train.shape)
print('Shape of data used for validation, and shape of validation targets: \n ', X_valid.shape, ',', Y_valid.shape)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

n_features = X_train_norm.shape[1]

model = Sequential()
model.add(Dense(1000, input_dim = n_features, activation='relu', use_bias=False))
model.add(Dense(1000, activation='relu', use_bias=False))
model.add(Dense(500, activation='relu', use_bias=False))
model.add(Dense(200, activation='relu', use_bias=False))
model.add(Dense(n_classes, activation='softmax', use_bias=False))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

model.summary()
# %%
save_at = "/kaggle/working/model.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, 
                             save_best_only=True, save_weights_only=False, mode='max')
history = model.fit( X_train_final, Y_train_final, 
                    epochs = 1, batch_size = 20, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_valid, Y_valid) )

# %%
score = model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')

# %%
trained_model = load_model("/kaggle/working/model.hdf5")

trained_model.layers
total_no_layers = len(trained_model.layers)
print(total_no_layers)
K = [0, 13,25, 40, 60, 75,  92 ]  

# %%
from matplotlib import pyplot as plt

weight_pruning=WeightPruning(trained_model)

# %%
resp =weight_pruning.pruning_scores(K,trained_model,X_test_norm,Y_test_onehot)
resp.show()
# olay şu K ve datayı nasıl vereceğiz ?  4 dakika 45 saniye sürdü 10 tane percentage degeri ile

# %%
neuron_pruning=NeuronPruning(trained_model)

# %%
resp=neuron_pruning.pruning_scores(K,trained_model,X_test_norm,Y_test_onehot)
resp.show()