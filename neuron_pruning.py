import numpy as np
from mll.error_handling.mll_error import MLLError

from mll.lib.dl.model_pruning.base import PruningBase
from tensorflow.keras.models import Sequential
import pandas as pd


class NeuronPruning(PruningBase):

    def __init__(self, trained_model: Sequential,username):
        self.__total_no_layers = len(trained_model.layers)
        self.__trained_model = trained_model
        self.__username=username
        self.__all_neurons_sorted = None
        self.__total_no_neurons = None
        self.__pruning_percentages = None
        self.__bias_control_map = super()._bias_control_map(trained_model, self.__total_no_layers)
        self.__weight_index_map = super()._weight_index_map(self.__bias_control_map, self.__total_no_layers)
        super().save_model(self.__trained_model,user_name=username)
        self.__all_scores_to_display = None

    def __sort_all_neurons(self):  # sorted işine iyi bak neuronslarda!!
        all_neurons = {}
        for layer_no in range(self.__total_no_layers - 1):
            layer_neurons = {}
            layer_neurons_df = pd.DataFrame(self.__trained_model.layers[layer_no].get_weights()[0])
            for i in range(len(layer_neurons_df.columns)):
                layer_neurons.update({i: np.array(layer_neurons_df.iloc[:, i])})

            layer_neurons = {(self.__weight_index_map[layer_no], k): v for k, v in layer_neurons.items()}
            all_neurons.update(layer_neurons)
        self.__all_neurons_sorted = {k: v for k, v in sorted(all_neurons.items(),
                                                             key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
        self.__total_no_neurons = len(self.__all_neurons_sorted)

    def prune_scores(self, pruning_array, X_test_norm, Y_test_onehot):
        neuron_pruning_scores = []
        total_bytes_neuron = []
        loss_pruning_scores_neuron = []
        trained_model = self.__trained_model

        self.__sort_all_neurons()
        self.__pruning_percentages = pruning_array
        # super().save_model(trained_model)
        for pruning_percent in self.__pruning_percentages:

            new_model = trained_model
            new_weights = trained_model.get_weights().copy()

            prune_fraction = pruning_percent / 100
            number_of_neurons_to_be_pruned = int(prune_fraction * self.__total_no_neurons)
            neurons_to_be_pruned = {k: self.__all_neurons_sorted[k] for k in
                                    list(self.__all_neurons_sorted)[: number_of_neurons_to_be_pruned]}

            for k, v in neurons_to_be_pruned.items():
                new_weights[k[0]][:, k[1]] = 0

            for layer_no in range(self.__total_no_layers - 1):
                new_layer_weights = new_weights[self.__weight_index_map[layer_no]].reshape(1, new_weights[
                    self.__weight_index_map[layer_no]].shape[0], new_weights[self.__weight_index_map[layer_no]].shape[1])
                if self.__bias_control_map[layer_no] == 0:
                    new_model.layers[layer_no].set_weights(new_layer_weights)
                else:
                    new_model.layers[layer_no].set_weights(
                        [new_layer_weights[0], new_model.layers[layer_no].get_weights()[1]])

            new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            neuron_pruning_scores.append(new_score[1])
            loss_pruning_scores_neuron.append(new_score[0])  # categorical_crossentropy
            bytes_size = super().zipped_able_model_size(new_model)
            total_bytes_neuron.append(bytes_size)
        self.__all_scores_to_display = {"Loss": loss_pruning_scores_neuron, "Accuracy": neuron_pruning_scores,
                                     "Total_Bytes": total_bytes_neuron}
        graph = super().graph_visualization(self.__pruning_percentages, self.__all_scores_to_display)
        return graph

    def pruned_single_model(self, pruning_percent, X_test_norm, Y_test_onehot):
        import os
        if self.__all_neurons_sorted is None:  # direkt olarak pruned_single_modeli çağırmasını istemiyoruz önce fonksiyona bakacak
             self.__sort_all_neurons()
#            error_type = "You have to enter the pruning percentage array before trying to with a value"
#            error_message = f"Weight Pruning Error"
#            out_data_manager = MLLError(error_type=error_type, error_message=error_message)
#            raise out_data_manager

        new_model = super().load_model(self.__username)
        new_weights = new_model.get_weights().copy()
        prune_fraction = pruning_percent / 100
        number_of_neurons_to_be_pruned = int(prune_fraction * self.__total_no_neurons)
        neurons_to_be_pruned = {k: self.__all_neurons_sorted[k] for k in
                                list(self.__all_neurons_sorted)[: number_of_neurons_to_be_pruned]}

        for k, v in neurons_to_be_pruned.items():
            new_weights[k[0]][:, k[1]] = 0

        for layer_no in range(self.__total_no_layers - 1):
            new_layer_weights = new_weights[self.__weight_index_map[layer_no]].reshape(1, new_weights[
                self.__weight_index_map[layer_no]].shape[0], new_weights[self.__weight_index_map[layer_no]].shape[1])
            if self.__bias_control_map[layer_no] == 0:
                new_model.layers[layer_no].set_weights(new_layer_weights)
            else:
                new_model.layers[layer_no].set_weights(
                    [new_layer_weights[0], new_model.layers[layer_no].get_weights()[1]])

        new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
        neuron_pruning_score = new_score[1]
        loss_pruning_score = new_score[0]  # categorical_crossentropy
        bytes_size = super().zipped_able_model_size(new_model)
        total_bytes_neuron = bytes_size
        # son olarak datamanageri sil
        if os.path.exists('%s.hdf5'% self.__username):
            os.remove('%s.hdf5'% self.__username)
        scores = {"Loss": loss_pruning_score, "Accuracy": neuron_pruning_score,
                  "Total_Bytes": total_bytes_neuron}
        return {"pruned_single_model": new_model, "pruned_model_scores": scores}
