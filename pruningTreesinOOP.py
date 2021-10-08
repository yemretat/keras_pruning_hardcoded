import pandas as pd
from mll.lib.dl.model_pruning.base import PruningBase
from tensorflow.keras.models import Sequential
from mll.error_handling.mll_error import MLLError
from tensorflow.keras.models import clone_model


class WeightPruning(PruningBase):
    def __init__(self, trained_model: Sequential,user_name="desktop"):
        self.__trained_model =trained_model
        self.__all_weights_sorted = None
        self.__total_No_Weights = None
        self.__total_no_layers = len(trained_model.layers)
        self.__pruning_percentages = None
        self.__user_name=user_name
        self.__bias_control_map = super()._bias_control_map(trained_model, self.__total_no_layers)
        self.__weight_index_map = super()._weight_index_map(self.__bias_control_map, self.__total_no_layers)
        super().save_model(self.__trained_model, user_name=self.__user_name) #username eklendi çünkü eğer kodu sunucuya impelemente ettiğimizde
        #aynı codebasede dosya isimleri karışmasın diye.
        self.__all_scores_to_display = None

    def __sort_all_weights(self):
        all_Weights = {}
        for layer_no in range(
                self.__total_no_layers - 1):  # aşağıda weights index map fonksiyonunu kullanma nedeni get_weightsde bias değerlerini de dönüyor fonksiyon sadece synapse weighstlerini almamızı sağlıyor
            layer_weights = (pd.DataFrame(self.__trained_model.layers[layer_no].get_weights()[
                                              0]).stack()).to_dict()  # exp, (0,0):0.79 demek o layerdaki ve bir önündeki layerdaki ilk neuronlar arasındaki synapse ağrılığı 0.79
            layer_weights = {(self.__weight_index_map[layer_no], k[0], k[1]): v for k, v in
                             layer_weights.items()}  # 3lü dictionary oluşturduk ( layer no, layerdaki neuron yeri,karşısındaki layerdaki neuron yeri ) : synapse weighti
            all_Weights.update(
                layer_weights)  # tüm layerları gezerek weightleri 3 elemanı olan tupple şeklinde çıkardık
        self.__all_weights_sorted = {k: v for k, v in sorted(all_Weights.items(), key=lambda item: abs(
            item[1]))}  # weights değerlerine göre sıraladım
        self.__total_No_Weights = len(self.__all_weights_sorted)

    def prune_scores(self, pruning_percentage_array, X_test_norm, Y_test_onehot):
        weight_pruning_scores = []
        total_bytes = []
        loss_pruning_scores = []
        trained_model = self.__trained_model
        self.__sort_all_weights()
        self.__pruning_percentages = pruning_percentage_array  # arrayi alma

        for pruning_percent in self.__pruning_percentages:
            new_model = trained_model
            new_weights = new_model.get_weights().copy()  # tüm weightsleri aldık biasler de dahil

            prune_fraction = pruning_percent / 100
            number_of_weights_to_be_pruned = int(prune_fraction * self.__total_No_Weights)
            weights_to_be_pruned = {k: self.__all_weights_sorted[k] for k in
                                    list(self.__all_weights_sorted)[:  number_of_weights_to_be_pruned]}
            # weights_to_be_pruned'ın eğer bias varsa weighIndexMapdeki olaya dönmesi lazım.
            for k, v in weights_to_be_pruned.items():
                new_weights[k[0]][k[1], k[
                    2]] = 0  # burada weightsToBeprunedda biasleri olmayacağı için weightIndexMap sayesinde hata vermeyecek
            for layer_no in range(self.__total_no_layers - 1):
                # aşağıdaki new_layer_weights bize 3dimensionlı array döndürüyor böylece direkt olarak set_weightse koyabiliyoruz
                new_layer_weights = new_weights[self.__weight_index_map[layer_no]].reshape(1, new_weights[
                    self.__weight_index_map[layer_no]].shape[0], new_weights[self.__weight_index_map[layer_no]].shape[1])
                if self.__bias_control_map[
                    layer_no] == 0:  # bu controlün amacı önceden hazırladığımız bias map'de bakıp eğer layer biasli ise tekrardan biasini bozmadan set weights yapabilmek
                    new_model.layers[layer_no].set_weights(new_layer_weights)
                else:
                    new_model.layers[layer_no].set_weights(
                        [new_layer_weights[0], new_model.layers[layer_no].get_weights()[1]])
            new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            loss_pruning_scores.append(new_score[0])  # categorical_crossentropy
            weight_pruning_scores.append(new_score[1])  # accuracy değerini ekler
            bytes_size = super().zipped_able_model_size(new_model)
            total_bytes.append(bytes_size)

        self.__all_scores_to_display = {"Loss": loss_pruning_scores, "Accuracy": weight_pruning_scores,
                                     "Total_Bytes": total_bytes}
        graph = super().graph_visualization(self.__pruning_percentages, self.__all_scores_to_display)
        return graph

    def pruned_single_model(self,pruning_percent, X_test_norm, Y_test_onehot):
        import os
        new_model = super().load_model(self.__user_name)  # tek fark buradan yüklemesi

        if self.__all_weights_sorted is None:  # direkt olarak pruned_single_modeli çağırmasını istemiyoruz önce fonksiyona bakacak
            self.__sort_all_weights()

#            error_type = "You have to give the pruning percentage array before trying to with a value"
#            error_message = f"Weight Pruning Error"
#            out_data_manager = MLLError(error_type=error_type, error_message=error_message)
#            raise out_data_manager
        #new_model = super().load_model(self.__user_name) #tek fark buradan yüklemesi
        # kaydettiğimiz modeli yükle
        new_weights = new_model.get_weights().copy()  # tüm weightleri aldık ve yine sadece synapse weightleri update edeceğiz biasleri elleme
        prune_fraction = pruning_percent / 100
        number_of_weights_to_be_pruned = int(prune_fraction * self.__total_No_Weights)
        weights_to_be_pruned = {k: self.__all_weights_sorted[k] for k in
                                list(self.__all_weights_sorted)[:  number_of_weights_to_be_pruned]}
        for k, v in weights_to_be_pruned.items():
            new_weights[k[0]][k[1], k[2]] = 0
        for layer_no in range(self.__total_no_layers - 1):
            # set etmek için weightleri array formatına çevirdik eğer bias içermeyen bir layersa direkt set edeceğiz bias içeriyorusa biase dokumnadan set edeceğiz
            new_layer_weights = new_weights[self.__weight_index_map[layer_no]].reshape(1, new_weights[
                self.__weight_index_map[layer_no]].shape[0], new_weights[self.__weight_index_map[layer_no]].shape[1])
            if self.__bias_control_map[layer_no] == 0:
                new_model.layers[layer_no].set_weights(new_layer_weights)
            else:
                new_model.layers[layer_no].set_weights(
                    [new_layer_weights[0], new_model.layers[layer_no].get_weights()[1]])
        new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
        loss_pruning_score = new_score[0]  # categorical_crossentropy
        weight_pruning_score = new_score[1]  # new_score[0] eklenecek
        total_byte = super().zipped_able_model_size(new_model)
        if os.path.exists('%s.hdf5'% self.__user_name):
            os.remove('%s.hdf5' % self.__user_name)

        scores = {"Loss": loss_pruning_score, "Accuracy": weight_pruning_score,
                  "Total_Bytes": total_byte}
        return {"pruned_single_model": new_model, "pruned_model_scores": scores}  # alınca da hesapla karşılaştır.
