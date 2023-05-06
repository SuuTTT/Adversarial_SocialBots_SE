import gym
import numpy as np
from scipy import sparse
import os
import warnings
import math
import networkx as nx
import threading
from gym import error
from gym import spaces
from gym import utils
from gym.utils import seeding
from joblib import dump
from joblib import load
import torch
import time
import glob
from gym.spaces import Box, Discrete, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from keras.utils.np_utils import to_categorical   
from ge.gcn_test import *
from ge.graph_utils import *
import scipy.stats as ss
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Detector:
    def __init__(self, model_path):
        self.scaler, self.model = load(model_path) 

    def predict(self, action, follower=None, following=None):
        x = self.extract_features(action, follower, following)
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def extract_features(self, action, follower=None, following=None):
        num_tweets = action.count('T')
        num_replies = action.count('A')
        num_retweets = action.count('R')
        num_mentions = action.count('M')
    # 
        avg_mentions_per_tweet = num_mentions / max(1, num_tweets)
        retweet_ratio = num_retweets / max(1, num_tweets)
        reply_ratio = num_replies / max(1, num_tweets)
        retweet_reply_ratio = num_retweets / max(1, reply_ratio)
        num_interactions = num_retweets + num_replies + num_mentions
        avg_interaction_per_tweet = num_interactions / max(1, num_tweets)

        rt = [num_tweets, num_replies, num_retweets]
        rt += [retweet_ratio, reply_ratio, retweet_reply_ratio]
        rt += [num_mentions, avg_mentions_per_tweet]
        rt += [num_interactions, avg_interaction_per_tweet]

        rt = np.array(rt).reshape(1, -1)
        return rt
        

class AdvBotEnvSingleDetectLargeHiar(MultiAgentEnv): #advbot-v6
# class AdvBotEnvSingleDetectLargeHiar(gym.Env): #advbot-v6
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A", "M"]
    MAX_TIME_STEP = 60
    INTERVALS = [20, 40, 60, 80, 100, 120, 140, 160]
    INTERVAL = 20
    UPPER_TIME_LIMIT = 3000
    OUT_DEGREE_MIN = 0

    ORIGIN_GRAPH = "advbot_train"
    NUM_COMMUNITY = 100
    MODE = "out_degree"
    REWARD_SHAPING = None

    COMPRESS_FOLLOWSHIP = True
    VERBOSE = False

    def __init__(self, 
                num_bots=1, 
                discrete_history=False, 
                random_stimulation=True, 
                seed=77, 
                override={}, 
                validation=False,
                debug=False,
                graph_algorithm="node2vec",
                walk_p=1, 
                walk_q=1,
                flg_detection=True,
                model_type="FCN",
                node_embed_dim=2,
                probs=0.25,
                graph_feature="out_degree",
                custom_max_step=None,
                validation_graphs=[],
                reward_shaping=None,
                level1_independent=False,
                detector_type="RandomForest",
                interval=None,
                node_percentage=0.9):
        self.seed(seed)

        for k in override:
            try:
                getattr(self, k)
                setattr(self, k, override[k])
                print("Update {} to {}".format(k, override[k]))
            except Exception as e:
                pass

        if interval:
            self.INTERVAL = interval
            print("updating INTERVAL to ", interval)

        if custom_max_step:
            self.MAX_TIME_STEP = custom_max_step
            self.INTERVALS = list(range(self.INTERVAL, self.MAX_TIME_STEP+1000*self.INTERVAL, self.INTERVAL))

        if reward_shaping:
            self.REWARD_SHAPING = reward_shaping

        self.node_percentage=node_percentage
        self.MODEL_PATH = './detector/{}Classifier_TRAM_lengthNone.joblib'.format(detector_type)
        self.level1_independent = level1_independent
        self.graph_algorithm = graph_algorithm
        self.walk_p = walk_p
        self.walk_q = walk_q
        self.node_embed_dim = node_embed_dim
        self.flg_detection = flg_detection
        self.PROB_RETWEET = probs
        self.MODE = graph_feature

        self.DEBUG = debug
        self.model_type = model_type
        self.validation = validation
        self.validation_graphs = validation_graphs
        self.discrete_history = discrete_history
        self.random_stimulation = random_stimulation

        self.n_fake_users = num_bots
        self.initialize()
        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)


    """ def update_avail_actions(self):
        activated_idx = np.array(list(self.seed_nodes.keys()))
        self.action_mask = np.array([1] * self.max_avail_actions) ## all actions are available
        if len(activated_idx) > 0:
            self.action_mask[activated_idx] = 0

        disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        self.action_mask[disable_node] = 0

        if self.action_mask.sum() == 0: # if there is no valid action => open all
            self.action_mask = np.array([1] * self.max_avail_actions)
            if len(activated_idx) > 0:
                self.action_mask[activated_idx] = 0 """

    """ def update_avail_actions(self):  # Set the value of k as needed
        k=self.k
        activated_idx = np.array(list(self.seed_nodes.keys()))
        self.action_mask = np.array([1] * self.max_avail_actions)  # Initialize the action mask to all 1s
        
        # Find the top k nodes with the highest PageRank that have not been activated
        remaining_nodes = set(self.pagerank.keys()) - set(activated_idx)
        sorted_remaining_nodes = sorted(remaining_nodes, key=lambda x: self.pagerank[x], reverse=True)
        top_k_nodes = sorted_remaining_nodes[:k]
        
        # Set all nodes to 0 in the action mask
        self.action_mask[:] = 0
        
        # Set the top k nodes to 1 in the action mask
        self.action_mask[top_k_nodes] = 1
        
        # Disable nodes with out-degree less than or equal to OUT_DEGREE_MIN
        disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        self.action_mask[disable_node] = 0
        
        # If there are no valid actions, reset the action mask to allow all nodes except the activated ones
        if self.action_mask.sum() == 0:
            self.action_mask = np.array([1] * self.max_avail_actions)
            if len(activated_idx) > 0:
                self.action_mask[activated_idx] = 0 """
    def visualize(self, last_k_entropy_nodes, last_k_pagerank_nodes, disable_node):
        G_nx = nx.DiGraph(self.G)
        pos = nx.spring_layout(G_nx, seed=42, k=0.15)
        nx.draw_networkx_edges(G_nx, pos, edge_color='lightgray', alpha=0.5)

        # Define colors for different node types
        default_color = 'lightblue'
        entropy_color = 'purple'
        pagerank_color = 'green'
        disable_color = 'orange'

        # Prepare colors and sizes for all nodes, with default color and size
        node_colors = [default_color for _ in range(G_nx.number_of_nodes())]
        node_sizes = [50 for _ in range(G_nx.number_of_nodes())]

        # Highlight the last k nodes with the highest structural entropy
        for node in last_k_entropy_nodes:
            node_colors[node] = entropy_color
            node_sizes[node] = 100

        # Highlight the last k nodes with the highest PageRank
        for node in last_k_pagerank_nodes:
            node_colors[node] = pagerank_color
            node_sizes[node] = 100

        # Highlight the disabled nodes
        for node in disable_node:
            node_colors[node] = disable_color
            node_sizes[node] = 100

        # Draw the nodes with the specified colors and sizes
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=node_sizes)

        # Save and show the plot
        current_thread = threading.current_thread()
        thread_id = current_thread.ident
        plt.savefig(f"img/0415/image_{thread_id}.png")
        plt.show()
        input("Press any key to continue...")

    def visualize_sep(self, last_k_entropy_nodes, last_k_pagerank_nodes, disable_node):
        print(last_k_entropy_nodes,"\n---\n",last_k_pagerank_nodes,"\n---\n",disable_node)
        G_nx = nx.DiGraph(self.G)
        pos = nx.spring_layout(G_nx, seed=42, k=0.15)
        
        plt.figure(figsize=(20, 5))

        # Original graph
        plt.subplot(1, 4, 1)
        nx.draw_networkx_edges(G_nx, pos, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=50)
        plt.title("Original Graph")
        
        # Graph with last k highest structural entropy nodes
        plt.subplot(1, 4, 2)
        nx.draw_networkx_edges(G_nx, pos, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=50)
        nx.draw_networkx_nodes(G_nx, pos, nodelist=last_k_entropy_nodes, node_color='purple', node_size=100)
        plt.title("Last k highest structural entropy nodes")

        # Graph with last k highest PageRank nodes
        plt.subplot(1, 4, 3)
        nx.draw_networkx_edges(G_nx, pos, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=50)
        nx.draw_networkx_nodes(G_nx, pos, nodelist=last_k_pagerank_nodes, node_color='green', node_size=100)
        plt.title("Last k highest PageRank nodes")

        # Graph with disabled nodes
        plt.subplot(1, 4, 4)
        nx.draw_networkx_edges(G_nx, pos, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_nodes(G_nx, pos, node_color='lightblue', node_size=50)
        nx.draw_networkx_nodes(G_nx, pos, nodelist=disable_node, node_color='orange', node_size=100)
        plt.title("Disabled nodes")

        # Save and show the plot
        current_thread = threading.current_thread()
        thread_id = current_thread.ident
        plt.savefig(f"img/0415/image_{thread_id}.png")
        plt.show()
        input("Press any key to continue...")

    def update_avail_actions(self):
        k=self.k
        activated_idx = np.array(list(self.seed_nodes.keys()))
        self.action_mask = np.array([1] * self.max_avail_actions)  # Initialize the action mask to all 1s
        

        # Find the top k nodes with the highest PageRank that have not been activated
        remaining_nodes = set(self.pagerank.keys()) - set(activated_idx)
        sorted_remaining_nodes = sorted(remaining_nodes, key=lambda x: self.pagerank[x], reverse=True)
        top_k_nodes = sorted_remaining_nodes[:k]
        last_k_pagerank_nodes=sorted_remaining_nodes[k-len(self.G):]

        # Find the top k nodes with the highest structural entropy that have not been activated
        remaining_nodes = set(self.structural_entropy.keys()) - set(activated_idx)
        logging.debug(f"remaining_nodes: {remaining_nodes}")
        sorted_remaining_nodes = sorted(remaining_nodes, key=lambda x: self.structural_entropy[x], reverse=True)
        top_k_nodes = sorted_remaining_nodes[:k]
        logging.debug(f"top_k_nodes: {top_k_nodes}")
        last_k_entropy_nodes= sorted_remaining_nodes[k-len(self.G):]
        logging.debug(f"last_k_entropy_nodes: {last_k_entropy_nodes}")
        # visualize for presentation
        disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        
        logger = logging.getLogger()
        if logger.level == logging.DEBUG:
            self.visualize_sep(last_k_entropy_nodes, last_k_pagerank_nodes, disable_node)


        # Set all nodes to 0 in the action mask
        self.action_mask[:] = 0
        
        # Set the top k nodes to 1 in the action mask
        self.action_mask[top_k_nodes] = 1
        self.action_mask[disable_node] = 0
        # Disable nodes with out-degree less than or equal to OUT_DEGREE_MIN
        #disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        #self.action_mask[disable_node] = 0
        
        # If there are no valid actions, reset the action mask to allow all nodes except the activated ones
        if self.action_mask.sum() == 0:
            self.action_mask = np.array([1] * self.max_avail_actions)
            if len(activated_idx) > 0:
                self.action_mask[activated_idx] = 0




    def vectorize_graph(self, g, mode="gcn"):
        if mode == "gcn":
            rt = np.stack(get_embeds(g, 
                    node_embed_dim=self.node_embed_dim,
                    alg=self.graph_algorithm,
                    p=self.walk_p, 
                    q=self.walk_q))

        elif mode == "out_degree":
            rt = self.out_degree/len(self.G)

        elif mode == "rank":
            rt = ss.rankdata(self.out_degree)/len(self.G)

        return rt


    def best_reward(self):
        idx = np.argsort(self.out_degree)[::-1][:self.MAX_TIME_STEP]
        cur_reward = self.compute_influence(self.G, list(idx), prob=self.PROB_RETWEET)
        return cur_reward


    def next_best_greedy(self):
        idx = np.argsort(self.out_degree)[::-1]
        for i in idx:
            if i not in self.seed_nodes:
                return i
        return np.random.choice(i)

    def construct_adj_matrix(self):
        n = len(self.G)
        adj_matrix = np.zeros((n, n))

        for node_i in self.G.nodes():
            for node_j in self.G.nodes():
                if self.G.has_edge(node_i, node_j):
                    adj_matrix[node_i, node_j] = self.G[node_i][node_j].get('weight', 1)

        return adj_matrix

    def calculate_structural_entropies(self):
        adj_matrix = self.construct_adj_matrix()  # This should be a method that returns the adjacency matrix
        logging.debug("building tree")
        partition_tree = PartitionTree(adj_matrix)
        # 树高
        x = partition_tree.build_coding_tree(2)
        logging.debug("tree built")
        structural_entropies = {}
        total_nodes = len(adj_matrix)
        for node_id, node in partition_tree.tree_node.items():
            if node_id < total_nodes:
                node_se=0
                if node.parent is not None:
                    node_p = partition_tree.tree_node[node.parent]
                    node_vol = node.vol
                    node_g = node.g
                    node_p_vol = node_p.vol
                    if partition_tree.VOL != 0 and node_vol > 0 and node_p_vol > 0:
                        node_se = -(node_g / partition_tree.VOL) * math.log2(node_vol / node_p_vol)
                    structural_entropies[node_id] = node_se
        #print(structural_entropies)
        #input("pause for debug")
        return structural_entropies

    def initialize(self, reset_network=True):
        if self.PROB_RETWEET < 0:
            np.random.seed(int(str(time.time())[-5:]))
            self.PROB_RETWEET = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            print("SETTING PROB TWEET", self.PROB_RETWEET)

        if not (self.validation and len(self.validation_graphs)):
            self.G = randomize_graph(graph_name=self.ORIGIN_GRAPH, k=self.NUM_COMMUNITY, mode=3)
            total_nodes = len(self.G)
            self.k = int(total_nodes * self.node_percentage)
            for e in self.G.edges():
                self.G[e[0]][e[1]]['weight'] = self.PROB_RETWEET
        else:
            idx = np.random.choice(len(self.validation_graphs))
            self.G = self.validation_graphs[idx]
            print("EVALUATING REAL GRPAH...", len(self.G))

        self.out_degree = np.array([a[1] for a in list(self.G.out_degree(list(range(len(self.G)))))])
        # print("OUT_DEGREE", np.sort(self.out_degree)[::-1][:10])
        #self.pagerank = nx.pagerank(self.G)
        # Construct the adjacency matrix
        adj_matrix = self.construct_adj_matrix()

        # Compute the structural entropy
        #partition_tree = PartitionTree(adj_matrix)
        entropies=self.calculate_structural_entropies()
        self.structural_entropy = {node: entropy for node, entropy in enumerate(entropies)}
        self.pagerank = nx.pagerank(self.G)
        self.n_legit_users = len(self.G)
        self.max_avail_actions = self.n_legit_users
        self.state = ""
        self.seed_nodes = {}
        self.following = {}
        if not self.validation:
            self.current_interval = 0
        else:
            self.current_interval = 1

        # self.activated_idx = np.array([1]*self.n_legit_users)
        self.level1_reward = 0.0
        self.level2_reward = 0.0
        self.done = 0
        self.current_t = 0
        self.previous_reward = 0
        self.previous_rewards = []
        self.last_undetect = 0
        self.heuristic_optimal_reward = self.best_reward()
        self.action_mask = np.array([1] * self.max_avail_actions)

        self.G_obs = self.vectorize_graph(self.G, mode=self.MODE)
        
        self.action_dim = self.node_embed_dim
        random_state = np.random.RandomState(seed=7777)
        self.action_assignments = random_state.normal(0, 1, (self.max_avail_actions, self.action_dim)).reshape(self.max_avail_actions, self.action_dim)
        
        # if not self.validation:
        self.update_avail_actions()

        self.level1_action_space = gym.spaces.Discrete(len(self.ACTION))
        self.level1_observation_space = gym.spaces.Box(low=0, high=1, shape=self.pack_observation("level1").shape)
        self.level2_action_space = gym.spaces.Discrete(self.max_avail_actions)

        temp_obs = self.pack_observation("level2")
        if self.model_type == "FCN":
            self.level2_observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
                "avail_actions": Box(-5, 5, shape=(self.max_avail_actions, self.action_dim)),
                "advbot":  gym.spaces.Box(low=-10, high=10, shape=temp_obs['advbot'].shape),
            })

        else:
            self.level2_observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
                "avail_actions": Box(-5, 5, shape=(self.max_avail_actions, self.action_dim)),
                "advbot":  gym.spaces.Box(low=-100, high=100, shape=temp_obs['advbot'].shape),
                "activated": gym.spaces.Box(low=0, high=1, shape=temp_obs['activated'].shape),
                "history": gym.spaces.Box(low=0, high=1, shape=temp_obs['history'].shape),
            })


    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def pack_observation(self, agent):
        state = self.state
        num_tweets = state.count('T')
        num_replies = state.count('A')
        num_retweets = state.count('R')
        num_mentions = state.count('M')
        bot_degree = len(self.following)

        if self.level1_independent:
            history = np.array([num_tweets, num_replies, num_retweets, num_mentions]).reshape(1, -1)
        else:
            history = np.array([num_tweets, num_replies, num_retweets, num_mentions, bot_degree]).reshape(1, -1)
        history = history / max(1, history.sum()) #dynamic

        if agent == "level1":
            obs = history
            # obs = np.concatenate((history, activated), 1)

        elif agent == "level2":
            network = self.G_obs
            activated_idx = np.array(list(self.seed_nodes.keys()))
            activated = np.array([0]*self.n_legit_users)
            if len(activated_idx):
                target_degree = np.array([a[1] for a in list(self.G.degree(activated_idx))])
                activated[activated_idx] = target_degree
            activated = activated.reshape(1, -1)
            activated = (1+len(self.following))/((1+activated))
            activated = activated / max(1, activated.sum())
                
            if self.model_type == "FCN":
                network = network.flatten().reshape(1, -1)
                if self.MODE not in ["rank", "gcn"]:
                    if len(activated_idx):
                        network[0,activated_idx] = 0
                advbot = np.concatenate((history, network, activated), 1)
                obs = {
                    "action_mask": self.action_mask,
                    "avail_actions": self.action_assignments,
                    "advbot": advbot
                }

            else:
                obs = {
                    "action_mask": self.action_mask,
                    "avail_actions": self.action_assignments,
                    "advbot": network.reshape(network.shape[0], network.shape[1], 1),
                    "activated": activated,
                    "history": history
                }

        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        obs = self.pack_observation("level1")
        return {"level1": obs}


    def compute_influence(self, graph, seed_nodes, prob, n_iters=10):
        total_spead = 0
        for i in range(n_iters):
            np.random.seed(i)
            active = seed_nodes[:]
            new_active = seed_nodes[:]
            while new_active:
                activated_nodes = []
                for node in new_active:
                    neighbors = list(graph.neighbors(node))
                    success = np.random.uniform(0, 1, len(neighbors)) < prob
                    activated_nodes += list(np.extract(success, neighbors))
                new_active = list(set(activated_nodes) - set(active))
                active += new_active
            total_spead += len(active)
        return total_spead / n_iters


    def cal_rewards(self, test=False, seeds=None, action=None, specific_action=False, reward_shaping=1):
        if not seeds:
            seeds = list(self.seed_nodes.keys())

        if not test:
            if not specific_action:
                cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward
            else:
                assert action >= 0
                cur_reward = self.compute_influence(self.G, [action], prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward
                
        else:
            # print("SEEDS", seeds, len(seeds))
            print("out_degree", self.out_degree[seeds][:5], len(self.seed_nodes))
            cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
            reward = cur_reward

        if reward_shaping:
            reward = 1.0 * reward/self.best_reward()

        return reward


    def render(self, mode=None):
        pass



    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "level1" in action_dict:
            return self._step_level1(action_dict["level1"])
        else:
            return self._step_level2(action_dict["level2"])


    def _step_level1(self, action):
        self.current_t += 1
        self.state += self.ACTION[action]
        detection_reward = 0.1
        
        if self.flg_detection:
            try:
                if len(self.state) > self.INTERVALS[self.current_interval]:
                # if self.current_t % self.INTERVAL == 0 and self.current_t > 0:
                    # print("CHECKING DETECTION", self.current_interval)
                    pred = self.detector.predict(self.state)[0]
                    if pred >= 0.5:
                        self.done = self.current_t
                    else:
                        detection_reward += (self.current_t - self.last_undetect)/self.MAX_TIME_STEP
                        self.last_undetect = self.current_t

                    self.current_interval += 1
            except:
                print(self.INTERVALS, len(self.INTERVALS))
                print(self.current_interval)

        if not self.validation:
            if len(self.state) >= self.MAX_TIME_STEP:
            # if self.current_t >= self.MAX_TIME_STEP:
                self.done = self.current_t
        else:
            if len(self.seed_nodes) >= self.MAX_TIME_STEP:
                self.done = self.current_t
            elif (self.current_t > self.UPPER_TIME_LIMIT):
                self.done = self.MAX_TIME_STEP

        self.level1_reward = detection_reward

        if self.ACTION[action] == "T":
            global_obs1 = self.pack_observation(agent="level1")
            if not self.done:
                return {"level1": global_obs1}, \
                        {"level1": 0.1 * self.level1_reward}, \
                        {"__all__":False}, \
                        {"level1": {}} 

            else:
                influence_reward = self.cal_rewards(specific_action=False, reward_shaping=self.REWARD_SHAPING)
                global_obs2 = self.pack_observation(agent="level2")
                if "R" in self.state or "A" in self.state or "M" in self.state:
                    return {"level1": global_obs1, "level2": global_obs2}, \
                            {"level1": influence_reward, "level2": influence_reward}, \
                            {"__all__":self.done}, \
                            {"level1": {}, "level2": {}} 
                else:
                    return {"level1": global_obs1}, \
                            {"level1": influence_reward}, \
                            {"__all__":self.done}, \
                            {"level1": {}} 

        else:
            global_obs2 = self.pack_observation(agent="level2")
            return {"level2": global_obs2}, \
                    {"level2": self.level2_reward}, \
                    {"__all__":False}, \
                    {"level2": {}}       


    def _step_level2(self, action):
        if self.validation:
            if len(self.seed_nodes) >= self.MAX_TIME_STEP:
                print("DONE HERE", len(self.seed_nodes), self.MAX_TIME_STEP)
                self.done = self.current_t
        
        if not (self.validation and self.done):
            self.seed_nodes[action] = 1

        self.following[action] = 1
        bot_degree = len(self.following)
        target_degree = self.G.degree(action)
        ratio = (1+bot_degree)/(1+target_degree)
        ratio = np.clip(ratio, a_min=0.0, a_max=1.0)

        if np.random.binomial(1, p=1-ratio):
            self.state += self.state[-1]
            self.state += self.state[-1]
            self.state += self.state[-1]

        # if not self.validation:
        self.update_avail_actions()
        
        global_obs1 = self.pack_observation(agent="level1")

        if not self.done:
            influence_reward = self.cal_rewards(action=action, specific_action=True, reward_shaping=self.REWARD_SHAPING)
            self.level2_reward = influence_reward
            
            return {"level1": global_obs1}, \
                {"level1": influence_reward}, \
                {"__all__": False}, \
                {"level1": {}}
        else:
            influence_reward = self.cal_rewards(specific_action=False, reward_shaping=self.REWARD_SHAPING)
            global_obs2 = self.pack_observation(agent="level2")
            return {"level1": global_obs1, "level2": global_obs2}, \
                {"level1": influence_reward, "level2": influence_reward}, \
                {"__all__": self.done}, \
                {"level1": {}, "level2": {}} 

    def close(self):
        self.reset()


import copy
import math
import heapq
import numba as nb
import numpy as np
import networkx as nx


def get_id():
    i = 0
    while True:
        yield i
        i += 1
def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = {}
    VOL = 0
    node_vol = []
    for i in range(g_num_nodes):
        n_v = 0
        adj = set()
        for j in range(g_num_nodes):
            if adj_matrix[i,j] != 0:
                n_v += adj_matrix[i,j]
                VOL += adj_matrix[i,j]
                adj.add(j)
        adj_table[i] = adj
        node_vol.append(n_v)
    return g_num_nodes,VOL,node_vol,adj_table

@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2):
    c12 = 0
    for i in range(len(p1)):
        for j in range(len(p2)):
            c = adj_matrix[p1[i],p2[j]]
            if c != 0:
                c12 += c
    return c12

def LayerFirst(node_dict,start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
                                 g=g, vol=v,child_h= child_h,child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break



def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth+=1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1,p_node):
    a = node1.child_cut
    v1 = node1.vol
    v2 = p_node.vol
    return a * math.log(v2 / v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2
    return ((v1 - g1) * math.log(v12 / v1,2) + (v2 - g2) * math.log(v12 / v2,2) - 2 * cut_v * math.log(g_vol / v12,2)) / g_vol



class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children:set = None,parent = None,child_h = 0, child_cut = 0):
        self.ID = ID
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h #不包括该节点的子树高度
        self.child_cut = child_cut

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

class PartitionTree():

    def __init__(self,adj_matrix):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(adj_matrix)
        self.id_g = get_id()
        self.leaves = []
        self.build_leaves()



    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = next(self.id_g)
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v)
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)


    def build_sub_leaves(self,node_list,p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            log_value = self.tree_node[vertex].vol / p_vol
            if log_value > 0:
                ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                        * math.log2(self.tree_node[vertex].vol / p_vol)
            else:
                # Handle the error or print a warning
                print(f"Warning: Encountered non-positive value ({log_value}) for logarithm")
            
            sub_n = set()
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex,vertex_n]
                if c != 0:
                    vol += c
                    sub_n.add(vertex_n)
            sub_leaf = PartitionTreeNode(ID=vertex,partition=[vertex],g=vol,vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict,ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2(node.vol / g_vol)
            new_n = set()
            for nei in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.add(nei)
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(ID=node_id,partition=node.partition,vol=node.vol,g = node.g,children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en


    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            if node.parent is not None:
                node_p = node_dict[node.parent]
                node_vol = node.vol
                node_g = node.g
                node_p_vol = node_p.vol
                ent += - (node_g / self.VOL) * math.log2(node_vol / node_p_vol)
        return ent


    def __build_k_tree(self,g_vol,nodes_dict:dict,k = None,):
        min_heap = []
        cmp_heap = []
        nodes_ids = nodes_dict.keys()
        new_id = None
        for i in nodes_ids:
            for j in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    if len(n1.partition) == 1 and len(n2.partition) == 1:
                        cut_v = self.adj_matrix[n1.partition[0],n2.partition[0]]
                    else:
                        cut_v = cut_volume(self.adj_matrix,p1 = np.array(n1.partition),p2=np.array(n2.partition))
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        while unmerged_count > 1:
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = next(self.id_g)
            merge(new_id, id1, id2, cut_v, nodes_dict)
            self.adj_table[new_id] = self.adj_table[id1].union(self.adj_table[id2])
            for i in self.adj_table[new_id]:
                self.adj_table[i].add(new_id)
            #compress delta
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id1],nodes_dict[new_id]),id1,new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id2],nodes_dict[new_id]),id2,new_id])
            unmerged_count -= 1

            for ID in self.adj_table[new_id]:
                if not nodes_dict[ID].merged:
                    n1 = nodes_dict[ID]
                    n2 = nodes_dict[new_id]
                    cut_v = cut_volume(self.adj_matrix,np.array(n1.partition), np.array(n2.partition))

                    new_diff = CombineDelta(nodes_dict[ID], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, ID, new_id, cut_v))
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = next(self.id_g)
            new_node = PartitionTreeNode(ID=new_id,partition=list(nodes_ids),children=unmerged_nodes,
                                         vol=g_vol,g = 0,child_h=new_child_h)
            nodes_dict[new_id] = new_node

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue
                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break
                for e in cmp_heap:
                    if e[1] == p_id:
                        if child_tree_deepth(nodes_dict, p_id) > k:
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[e[2]])
                    if e[1] in children:
                        if nodes_dict[e[1]].child_h == 0:
                            continue
                        if child_tree_deepth(nodes_dict, e[1]) > k:
                            e[2] = p_id
                            e[0] = CompressDelta(nodes_dict[e[1]], nodes_dict[p_id])
                heapq.heapify(cmp_heap)
        return root


    def check_balance(self,node_dict,root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict,c)

    def single_up(self,node_dict,node_id):
        new_id = next(self.id_g)
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                      children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for i in self.adj_table[node_id]:
            self.adj_table[i].add(new_id)



    def root_down_delta(self):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0 , None , None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol,nodes_dict=subgraph_node_dict,k=2)
        self.check_balance(subgraph_node_dict,new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self,sub_node_dict,sub_root_id,node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict,sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2(node.vol / node_p.vol)
        return ent

    def leaf_up(self):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict,ori_ent = self.build_sub_leaves(sub_nodes,candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,nodes_dict=subgraph_node_dict,k = 2)
                self.check_balance(subgraph_node_dict,sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict,sub_root,node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta,id_mapping,h1_new_child_tree

    def leaf_up_update(self,id_mapping,leaf_up_dict):
        for node_id,h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node,i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1


    def root_down_update(self, new_id , root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2'):
        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k = k)
        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k = 2)
            self.check_balance(self.tree_node,self.root_id)

            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2


            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta,id_mapping,leaf_up_dict = self.leaf_up()
                    root_down_delta, new_id , root_down_dict = self.root_down_delta()

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up()
                elif flag == 2:
                    root_down_delta, new_id , root_down_dict = self.root_down_delta()
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    # print('root down')
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id,root_down_dict)

                else:
                    # leaf up update
                    # print('leave up')
                    flag = 1
                    # print(self.tree_node[self.root_id].child_h)
                    self.leaf_up_update(id_mapping,leaf_up_dict)
                    # print(self.tree_node[self.root_id].child_h)


                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[root_down_id].children
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        assert len(self.tree_node) == count


def load_graph(dname):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('datasets/%s/%s.txt' % (dname, dname.replace('-', '')), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                g.add_node(j, tag=row[0])
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
            else:
                node_features = None

            assert len(g) == n
            g_list.append({'G': g, 'label': l})
    print("# data: %d\tlabel:%s" % (len(g_list), len(label_dict)))
    return g_list


if __name__ == "__main__":
    undirected_adj = [[0, 3, 5, 8, 0], [3, 0, 6, 4, 11],
                      [5, 6, 0, 2, 0], [8, 4, 2, 0, 10],
                      [0, 11, 0, 10, 0]]

    undirected_adj = [[0, 1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0]]
    undirected_adj = np.array(undirected_adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_coding_tree(2)
    for k, v in y.tree_node.items():
        print(k, v.__dict__)