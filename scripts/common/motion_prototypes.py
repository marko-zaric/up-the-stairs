import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from common.rgng.rgng import RobustGrowingNeuralGas
from sklearn.preprocessing import StandardScaler
from copy import copy, deepcopy

class MotionPrototypes:
    def __init__(self, motion_samples: pd.DataFrame, motion_dimensions: list) -> None:
        self.motion_samples = copy(motion_samples)
        self.motion_dimensions = motion_dimensions
        self.motion_prototypes = None
        self.m_samples_labeled = None
        self.prototypes_per_label = None
        self.cluster_labels = None

    def generate(self, effect_dimensions, prototypes_per_cluster=1):
        # Assert effect_dimensions list
        if len(effect_dimensions) == 1:
            # histogram
            hist, bin_edges = np.histogram(self.motion_samples[effect_dimensions[0]])
            
            label = 0
            cluster_labels = {}
            for i, count in enumerate(hist):
                if count != 0:
                    cluster_labels[label] = (bin_edges[i], bin_edges[i + 1])
                    label += 1

            self.m_samples_labeled = copy(self.motion_samples)
            self.m_samples_labeled['cluster_label'] = self.m_samples_labeled[effect_dimensions[0]].apply(lambda x: self.find_position_hist(x, cluster_labels))
            

            if prototypes_per_cluster == 1:
                # Mean Action
                return self.single_prototype_per_class()
            else:
                return self.multi_prototypes(prototypes_per_cluster)


        elif len(effect_dimensions) > 1:
            # kmeans
            X = np.array(self.motion_samples[effect_dimensions])

            range_n_clusters = [3, 4, 5, 6]
            best_score = 0
            best_num_of_clusters = 0
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                silhouette_avg = silhouette_score(X, kmeans.labels_)
    
                if best_score < silhouette_avg:
                    best_score = silhouette_avg
                    best_num_of_clusters = n_clusters

            kmeans = KMeans(n_clusters=best_num_of_clusters, random_state=0).fit(X)

            self.m_samples_labeled = self.motion_samples
            self.m_samples_labeled.loc[:, ('cluster_label')] = kmeans.labels_
            self.cluster_labels = set(kmeans.labels_)

            self.generate_prototypes(effect_dimensions)
            
            # if prototypes_per_cluster == 1:
            #     return self.single_prototype_per_class()
            # else:
            #     return self.multi_prototypes(prototypes_per_cluster)
        
    def generate_prototypes(self, effect_dimensions):
        # Dynamic prototypes per cluster
        mean_stds = []
        for i in self.cluster_labels:
            cluster_samples = self.m_samples_labeled[self.m_samples_labeled['cluster_label'] == i]
            mean_stds.append(self.encode_mean_std([cluster_samples], effect_dimensions))
        
        mean_stds = np.stack(mean_stds)
        
        # mean_stds = mean_stds / np.max(mean_stds, axis=(0,1))
        # max_prototypes_per_cluster = np.ceil(10*np.add.reduce(np.multiply.reduce(mean_stds.T))).astype(int)
        
        mean_std_all_dims = np.add.reduce(mean_stds, axis=1)
        mean_std_all_dims = mean_std_all_dims / np.max(mean_std_all_dims, axis=(0,1))
        cv = mean_std_all_dims.T[1]/mean_std_all_dims.T[0]
        max_prototypes_per_cluster = (1-cv)*mean_std_all_dims.T[1]
        max_prototypes_per_cluster = max_prototypes_per_cluster / np.min(max_prototypes_per_cluster)
        max_prototypes_per_cluster = np.floor(max_prototypes_per_cluster)
        
        print("Max prototypes per cluster: ", max_prototypes_per_cluster)
        self.prototypes_per_label = {}
        self.pre_process = StandardScaler()
        scaled_m_samples = deepcopy(self.m_samples_labeled)
        scaled_m_samples[self.motion_dimensions] = self.pre_process.fit_transform(scaled_m_samples[self.motion_dimensions])

        for cluster_label, num_prototypes in zip(self.cluster_labels, max_prototypes_per_cluster):
            if num_prototypes == 1:
                self.single_prototype_per_class(cluster_label)
            else:
                self.multi_prototypes(num_prototypes, scaled_m_samples[scaled_m_samples['cluster_label'] == cluster_label], cluster_label)


    def single_prototype_per_class(self, cluster_label):
        single_cluster_df = self.m_samples_labeled[self.m_samples_labeled['cluster_label'] == cluster_label]
        cluster_action_means = single_cluster_df[self.motion_dimensions].mean().to_numpy()

        if self.motion_prototypes is None:
            self.motion_prototypes = cluster_action_means
        else:
            self.motion_prototypes = np.vstack((self.motion_prototypes, cluster_action_means))
        self.prototypes_per_label[cluster_label] = cluster_action_means
        #self.action_prototypes = self.m_samples_labeled.groupby(['cluster_label'])[self.motion_dimensions].mean().to_numpy()
        # return self.action_prototypes
    

    def multi_prototypes(self, num_prototypes, cluster_data, cluster_label):
        # RGNG
        data_np = cluster_data[self.motion_dimensions].to_numpy()
        rgng = RobustGrowingNeuralGas(input_data=data_np, max_number_of_nodes=num_prototypes, real_num_clusters=1)
        resulting_centers = rgng.fit_network(a_max=100, passes=20)
        # resulting_centers = rgng.fit_network(a_max=100, passes=25)
        local_prototype = self.pre_process.inverse_transform(resulting_centers)
        if self.motion_prototypes is None:
            self.motion_prototypes = local_prototype
            self.prototypes_per_label[cluster_label] = local_prototype
        else:
            self.motion_prototypes = np.vstack((self.motion_prototypes, local_prototype))
            self.prototypes_per_label[cluster_label] = local_prototype

    def encode_mean_std(self, dfs, effects):
        for df in dfs:
            df_effect = df[effects]
            effect_mean = df_effect.mean()
            effect_std = df_effect.std()
            effect_array = np.zeros((len(effects),2))
            i = 0
            for mean, std in zip(effect_mean, effect_std):
                effect_array[i] = np.array([mean,std])
                i += 1

            return effect_array 

    def find_position_hist(self, value, dict_labels):
        for key, border in dict_labels.items():
            if border[0] <= value and value <= border[1]:
                return key

      