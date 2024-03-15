import numpy as np
from scipy import spatial
import networkx as nx
from common.rgng.RGNG_network import RGNG_Graph
import matplotlib.pyplot as plt
from sklearn import decomposition
import statistics
import copy
import rospy

class RobustGrowingNeuralGas:

    def __init__(self, input_data, max_number_of_nodes, real_num_clusters, center=None):
        self.network = None
        self.optimal_network = None
        self.data = input_data
        self.units_created = 0
        self.beta_integral_mul = 2
        self.fsigma_weakening = 0.1
        self.e_bi = 0.1
        self.e_bf = 0.01
        self.e_ni = 0.005
        self.e_nf = 0.0005
        self.eta = 0.0001
        self.inputted_vectors = []
        self.outliers = []
        self.receptive_field = {}
        self.optimal_receptive_field = {}
        self.max_nodes = len(input_data)
        
        # Matlab 
        self.stopcriteria = 0.00001
        self.init_centers = center
        
        self.prenumnode = max_number_of_nodes
        self.num_features = len(input_data[0])
        self.num_classes = real_num_clusters
        self.optimal_center = []

        self.old_prototypes = []

        self.units_created = 0
        self.epsilon=10**(-5)

        # 0. start with two units a and b at random position w_a and w_b
        if self.init_centers is None:
            w_a = [np.random.uniform(np.min(np.min(self.data,0),0), np.max(np.max(self.data,1),0)) for _ in range(np.shape(self.data)[1])]
            w_b = [np.random.uniform(np.min(np.min(self.data,0),0), np.max(np.max(self.data,1),0)) for _ in range(np.shape(self.data)[1])]
        else:
            w_a = self.init_centers[0]
            w_b = self.init_centers[1]
        self.network = RGNG_Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0, e_b=0, e_n=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0, e_b=0, e_n=0)
        self.units_created += 1

        for node in self.network.nodes():
            self.network.nodes[node]["prenode_ranking"] = 1

        plt.style.use('ggplot')

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = (np.linalg.norm((observation - vector), ord=2) + self.epsilon)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)

    def fit_network(self, a_max, passes=10):
        #np.random.seed(1)
        self.epochspernode = passes

        nofirsttimeflag=0
        stopflag=1
        allmdlvalue=[]
        previousmdlvalue=999999999

        # 1. iterate through the data
        sequence = 0

        self.data_range = np.max(np.max(self.data, axis=0)) - np.min(np.min(self.data, axis=0))
        #np.random.shuffle(self.data)

        while len(self.network.nodes()) <= self.prenumnode and stopflag == 1:
            print("Training when the number of the nodes in RGNG is: " + str(len(self.network.nodes())))
            flag = 1
            self.old_prototypes = []
            harmdist=[]
            for i in self.network.nodes():
                self.old_prototypes.append([self.network.nodes[i]["vector"], i])
                temp = 0
                for x in self.data:
                    temp = temp + 1 / (np.linalg.norm(x - self.network.nodes[i]['vector'], 2) + self.epsilon)
                harmdist.append((temp / len(self.data)))
                self.network.nodes[i]['e_b'] = self.e_bi * pow((self.e_bf / self.e_bi ), (self.network.nodes[i]["prenode_ranking"]-1) / self.max_nodes)
                self.network.nodes[i]['e_n'] = self.e_ni * pow((self.e_nf / self.e_ni ), (self.network.nodes[i]["prenode_ranking"]-1) / self.max_nodes)

            rand_state_count = 0
            for iter2 in range(self.epochspernode):
                tempvalue = 1 / np.array(harmdist)

                # In case a node gets deleted the node index still stays the same for the nodes above this dict keeps track of that
                NODE_TO_VALUE_CORRESPONDANCE = {}
                for i, node in enumerate(self.network.nodes()):
                    NODE_TO_VALUE_CORRESPONDANCE[node] = i

                if flag == 1:
                    iter1 = 0
                    
                    np.random.shuffle(self.data)
                    workdata = list(copy.deepcopy(self.data))
                    for observation in workdata:
                        iter1 = iter1 + 1
                        t = iter1+iter2*len(self.data)
                        # self.d_restr = { n_[0]: (np.linalg.norm( (observation - n_[1]['vector'])**2, ord=2) + self.epsilon) for n_ in self.network.nodes.items() } 
                        self.d_restr = { n_: (np.linalg.norm( (observation - self.network.nodes[n_]['vector'])**2, ord=2) + self.epsilon) for n_ in self.network.nodes() } #.items
                       
                        
                        # 2: Determine the winner S1 and the second nearest node S2
                        nearest_units = self.find_nearest_units(observation)
                        s_1 = nearest_units[0]
                        s_2 = nearest_units[1]

                        # 3:Set up or refresh connection relationship between S1 and S2
                        self.network.add_edge(s_1, s_2, age=0)

                        # 4: Adapt the reference vectors of S1 and its direct topological neighbours
                        tempv = 0
                        if self.d_restr[s_1] > tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]:
                            tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]] = 2 / ((1/self.d_restr[s_1]) + (1/tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]))
                            tempv = tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]
                        else:
                            tempv = self.d_restr[s_1]
                            tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]] = (self.d_restr[s_1] + tempvalue[NODE_TO_VALUE_CORRESPONDANCE[s_1]]) / 2
                        

                        update_w_s_1 = self.network.nodes[s_1]['e_b'] * tempv*((observation - self.network.nodes[s_1]['vector']) / self.d_restr[s_1])
                        self.network.nodes[s_1]['vector'] = np.add(self.network.nodes[s_1]['vector'], update_w_s_1)

                        # find neighbours of the winning node S1 and update them
                        avg_neighbor_dist = 0
                        if len(list(self.network.neighbors(s_1))) > 0:
                            avg_neighbor_dist = sum([np.linalg.norm(self.network.nodes[nb]['vector'] - self.network.nodes[s_1]['vector'], ord=2) + self.epsilon for nb in self.network.neighbors(s_1)])/len(list(self.network.neighbors(s_1)))
                        
                        for neighbor in self.network.neighbors(s_1):
                            tempv = 0
                            if self.d_restr[neighbor] > tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]:
                                tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]] = 2 / ((1/self.d_restr[neighbor]) + (1/tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]))
                                tempv = tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]
                            else:
                                tempv = self.d_restr[neighbor]
                                tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]] = (self.d_restr[neighbor] + tempvalue[NODE_TO_VALUE_CORRESPONDANCE[neighbor]]) / 2

                            s1_to_neighbor = np.subtract(self.network.nodes[neighbor]['vector'], self.network.nodes[s_1]['vector'])

                            update_w_s_n = self.network.nodes[neighbor]['e_n'] * tempv*((observation - self.network.nodes[neighbor]['vector'])/self.d_restr[neighbor]) + np.exp(-(np.linalg.norm(self.network.nodes[neighbor]['vector'] - self.network.nodes[s_1]['vector'], ord=2) + self.epsilon)/self.fsigma_weakening)*self.beta_integral_mul*avg_neighbor_dist*(s1_to_neighbor/np.linalg.norm(s1_to_neighbor, ord=2) + self.epsilon)
                            self.network.nodes[neighbor]['vector'] = np.add(self.network.nodes[neighbor]['vector'], update_w_s_n)

                        # 5: Increase the age of all edges emanating from S1
                        for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                            self.network.add_edge(u, v, age=attributes['age']+1)

                        # 6: Removal of nodes
                        if nofirsttimeflag == 1:
                            self.prune_connections(a_max)
                    rand_state_count += 1
                    # Check if stopping criterion is meet 
                    crit = 0
                    for vector, i in self.old_prototypes:
                        try:
                            crit += np.linalg.norm(vector - self.network.nodes[i]['vector'], ord=2) + self.epsilon
                        except:
                            crit += np.linalg.norm(vector, ord=2) 
                    crit /= len(self.network.nodes())
                    if crit <= self.stopcriteria:
                        # print("stop")
                        flag = 0
                    else:
                        for i in self.network.nodes():
                            for x, [vector, j] in enumerate(self.old_prototypes):
                                if j == i:
                                    self.old_prototypes[x][0] = self.network.nodes[i]["vector"]


                    harmdist=[]
                    for i in self.network.nodes():
                        temp = 0
                        for x in self.data:
                            temp = temp + 1 / (np.linalg.norm(x - self.network.nodes[i]['vector'], 2) + self.epsilon)
                        harmdist.append((temp / len(self.data)))

                    a = 1
            # End Epoch Loop
            # print("Epoch loop ended!!!")
            
            # Rebuiding the topology relationship among all current nodes
            self.d = {} #np.zeros(len(self.network.nodes()))

            # Harmonic average distance for each node
            harmonic_average = {}
            self.receptive_field = {}
            for observation in self.data:
                self.d = { n_: (np.linalg.norm( (observation - self.network.nodes[n_]['vector']), ord=2)) for n_ in self.network.nodes() }
                # for i in range(len(self.d)):
                #     self.d[i] = np.linalg.norm( (observation - self.network.nodes[i]['vector']), ord=2) + self.epsilon
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                self.network.add_edge(s_1, s_2, age=0)
                # Update receptive field of winner (optimal cluster size)
                if s_1 not in self.receptive_field.keys():
                    self.receptive_field[s_1] = {'input': [observation]}
                else:
                    self.receptive_field[s_1]['input'].append(observation)
                if s_1 not in harmonic_average.keys():
                    harmonic_average[s_1] = 1 / self.d[s_1]
                else:
                    harmonic_average[s_1] += 1 / self.d[s_1]

            for s in self.network.nodes():
                if s in self.receptive_field.keys():
                    harmonic_average[s] = len(self.receptive_field[s]['input']) / harmonic_average[s]
            # print(harmonic_average)
            # Local Error in each node
            for u in self.network.nodes():
                self.network.nodes[u]['error'] = 0
            for obs in self.data:
                # d = { n_[0]: (np.linalg.norm( (obs - n_[1]['vector']), ord=2)) for n_ in self.network.nodes.items() }
                d = { n_: (np.linalg.norm( (obs - self.network.nodes[n_]['vector']), ord=2)) for n_ in self.network.nodes() }
                nearest_units = self.find_nearest_units(observation)
                s = nearest_units[0]
                self.network.nodes[u]["error"] = self.network.nodes[u]["error"] + np.exp(-(d[s]/harmonic_average[s]))*d[s]
            

            # MDL Calc
            prototypes = np.zeros((len(self.network.nodes), np.shape(self.data)[1])) #len(self.init_centers[0])
            for i, n in enumerate(self.network.nodes()):
                prototypes[i] = self.network.nodes[n]['vector']

            prototypes = np.array(prototypes)
            # print("Prototypes: ")
            # print(prototypes)


            mdlvalue = self.outliertest(prototypes)
            if mdlvalue < previousmdlvalue:
                previousmdlvalue = mdlvalue
                self.optimal_center = prototypes
                self.optimal_receptive_field = copy.deepcopy(self.receptive_field)
                self.optimal_network = copy.deepcopy(self.network)
            # print(prototypes.shape[0])
            if prototypes.shape[0] == self.num_classes:
                actcenter = prototypes
            allmdlvalue.append(mdlvalue)


            

            
            # new node calc 
            error_max = 0
            q = None
            for u in self.network.nodes():
                if self.network.nodes[u]['error'] > error_max:
                    error_max = self.network.nodes[u]['error']
                    q = u
            # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
            f = -1
            largest_error = -1
            for u in self.network.neighbors(q):
                if self.network.nodes[u]['error'] > largest_error:
                    largest_error = self.network.nodes[u]['error']
                    f = u
            w_r = 0.5 * (np.add(self.network.nodes[q]['vector'], self.network.nodes[f]['vector']))
            r = self.units_created
            self.units_created += 1
            # 10: Insert new node
            if len(self.network.nodes()) <  self.prenumnode:
                self.network.add_node(r, vector=w_r, error=0, e_b=0, e_n=0)
            else:
                stopflag=0

            nofirsttimeflag = 1
            self.network.add_edge(r, q, age=0)
            self.network.add_edge(r, f, age=0)
            self.network.remove_edge(q, f)
            
            # plototypes = np.zeros((len(self.network.nodes), len(self.init_centers[0])))
            # for n in self.network.nodes():
            #     print("Node ", n, ":")
            #     print(self.network.nodes[n]['vector'])
            #     plototypes[n] = self.network.nodes[n]['vector']

            # plototypes = np.array(plototypes)
            # colors = ["yellow", "green", "blue", "purple", "red"]

            # plt.scatter(INPUT.T[0], INPUT.T[1], c=INPUT.T[2], cmap=matplotlib.colors.ListedColormap(colors), s=10)
            # plt.scatter(plototypes.T[0], plototypes.T[1],c="black", marker="s")
            # plt.show()
                    
        return self.optimal_center#, actcenter, allmdlvalue


    def outliertest(self, center):
        yeta=0.000005 # data accuracy
        ki=1.2 # Error balance coefficient
        rangevalue = self.data_range
        harmdist = np.zeros(len(self.network.nodes()))
        counter = np.zeros(center.shape[0])
        inderrorvector = []
        totalerrorvector = []
       
        NODE_TO_VALUE_CORRESPONDANCE = {}
                    
        for i, node in enumerate(self.network.nodes()):
            inderrorvector.append(np.zeros_like(center[0]))
            NODE_TO_VALUE_CORRESPONDANCE[node] = i
        self.receptive_field = {}

        for observation in self.data:
            # rms error
            d= { n_[0]: (np.linalg.norm( (observation - n_[1]['vector']), ord=2)) for n_ in self.network.nodes.items() }
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0] 
            s_index_in_matrix = NODE_TO_VALUE_CORRESPONDANCE[s]
            harmdist[s_index_in_matrix] = harmdist[s_index_in_matrix] + 1 / d[s]
            counter[s_index_in_matrix] = counter[s_index_in_matrix] + 1
            # Update receptive field of winner (optimal cluster size)
            if s not in self.receptive_field.keys():
                self.receptive_field[s] = {'input': [observation]}
            else:
                self.receptive_field[s]['input'].append(observation)

            inderrorvector[s_index_in_matrix] = inderrorvector[s_index_in_matrix] + d[s]
            totalerrorvector.append(d[s])

        for n in self.network.nodes():
            if n not in self.receptive_field.keys():
                harmdist[NODE_TO_VALUE_CORRESPONDANCE[n]] = 99999999
            else:
                harmdist[NODE_TO_VALUE_CORRESPONDANCE[n]] = harmdist[NODE_TO_VALUE_CORRESPONDANCE[n]] / len(self.receptive_field[n]['input'])


        disvector = np.zeros(len(self.data))
        for i, obs in enumerate(self.data):
            d = np.zeros(len(self.network.nodes()))
            for n in self.network.nodes():
                d[NODE_TO_VALUE_CORRESPONDANCE[n]] = np.linalg.norm( (obs - self.network.nodes[n]['vector']), ord=2)
            disvector[i] = (sum(1/(d*harmdist)))

        outliercandidate = np.sort(disvector)
        outliercandidate_args = np.argsort(disvector)
        
        outdata=[]
        errorvalue=0
        protosize=center.shape[0]

        flagprototype=0

        for i in range(len(outliercandidate)):
            d = np.zeros(len(self.network.nodes()))
            for j in range(center.shape[0]):
                d[j] = np.linalg.norm(self.data[outliercandidate_args[i]] - center[j],2)
            minval = np.min(d)
            s = np.argmin(d)
            erroradd = 0
            for h in np.arange(0,(self.data.shape[1])).reshape(-1): #range(self.data.shape[1]): #
                if np.abs(self.data[outliercandidate_args[i],h] - center[s,h]) != 0:
                    erroradd = erroradd + np.max(np.log2(np.abs(self.data[outliercandidate_args[i],h] - center[s,h]) / yeta))
                else:
                    erroradd = erroradd + 1
            errorvalue = errorvalue + erroradd
            
            a = 1
            
            if counter[s] >= 2:
                flagprototype = 0
                indexchange = np.log2(protosize)
            else:
                indexchange = np.log2(protosize) + (len(self.data) - len(outdata) - 1) * (np.log2(protosize) - np.log2(protosize - 1))
                protosize = protosize - 1
                flagprototype = 1

            if (ki * erroradd + indexchange) + flagprototype * center.shape[1] * (np.ceil(np.log2(rangevalue / yeta)) + 1) > self.data.shape[1] * (np.ceil(np.log2(rangevalue / yeta)) + 1):
                outdata.append(self.data[outliercandidate_args[i],np.arange(0,self.data.shape[1])])
                counter[s] = counter[s] - 1
                errorvalue = errorvalue - erroradd
                
        
        indexvalue = (len(self.data) - len(outdata)) * np.log2(protosize + 1)
        mdlvalue = protosize * center.shape[1] * (np.ceil(np.log2(rangevalue / yeta)) + 1) + indexvalue + ki * errorvalue + len(outdata) * (self.data.shape[1]) * (np.ceil(np.log2(rangevalue / yeta)) + 1)
        # print("MDL Value: ", mdlvalue)

        return mdlvalue



    # ==================================
    # proto, obs, d(-1)
    #(self.network.nodes[s_1]['vector'], observation, self.d_restr_prev[s_1])
    def sigma_modulation(self, proto, observation, prev):
        # pseudo algo
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            return current_error
        else:
            return prev

    def update_restricting_dist(self, proto, observation, prev):
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            # arithmetic mean
            return 0.5 * (prev + current_error)
        else:
            # harmonic mean
            return statistics.harmonic_mean([prev, current_error])

    # array a
    def h_mean(self, a):
        return statistics.harmonic_mean(a)

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        for u in self.network.nodes():
            vector = self.network.nodes[u]['vector']
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)
    
    def cluster_data(self):
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        clustered_data = []
        for key in self.optimal_receptive_field.keys():
            vectors = np.array(self.optimal_receptive_field[key]['input'])
            plt.scatter(vectors.T[0], vectors.T[1], c=color[key], s=10)
            for obs in self.optimal_receptive_field[key]['input']:
                clustered_data.append((obs, key))
        plt.show()
            
        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def plot_clusters(self, clustered_data):
        number_of_clusters = len(self.optimal_network.nodes())
        # print("NUMBER OF CLUSTERS: ", number_of_clusters)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            # print(i)
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.show()
        #plt.savefig('visualization/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
        return global_error


