__version__ = "0.2"

"""
MultiXrank
========
Universal multilayer Exploration by Random Walk with Restart
See https://multixrank.readthedocs.com for complete documentation.
"""

import numpy
import os
import copy
import pandas
import pathlib
import shutil
import sys
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Union, Optional
import scipy

from multixrank import constants
from multixrank.BipartiteAll import BipartiteAll
from multixrank.ConfigParser import ConfigParser
from multixrank.Output import Output
from multixrank.PathManager import PathManager
from multixrank.TransitionMatrix import TransitionMatrix
from multixrank.logger_setup import logger


class Multixrank(object):
    """Main class to run the random walk with restart in universal multiplex networks"""

    def __init__(self, config: str, wdir: str, pr=None):
        """
        Constructs an object for the random walk with restart.

        Args:
            config (str): Path to the configuration file in YML format. Paths will be used relative to the wdir path variable below
            wdir (str): Path to the working directory that will be as starting point to the paths in the config file.
        """

        #######################################################################
        #
        # Read ConfigPath
        #
        #######################################################################

        config_parser_obj = ConfigParser(config=config, wdir=wdir)
        config_parser_obj.parse()
        self.pr = pr
        self.wdir = os.path.join(os.getcwd(), wdir)

        if not os.path.isdir(self.wdir):
            logger_setup.logger.error('This input config_path is NOT a directory: {}'.format(self.wdir))
            sys.exit(1)

        #######################################################################
        #
        # paramater object from config_parser and properties
        #
        #######################################################################
        parameter_obj = config_parser_obj.parameter_obj
        self.r = parameter_obj.r
        self.lamb = parameter_obj.lamb
        logger.debug("Parameter 'lambda' is equal to: {}".format(self.lamb))

        #######################################################################
        #
        # multiplexall object from config_parser and properties
        #
        #######################################################################

        multiplexall_obj = config_parser_obj.multiplexall_obj
        self.multiplexall_obj = multiplexall_obj

        #######################################################################
        #
        # bipartite object from config_parser and properties
        #
        #######################################################################

        self.bipartiteall_obj = config_parser_obj.bipartitelist_obj

        self.multiplex_layer_count_list = [len(multiplexone_obj.layer_tuple) for multiplexone_obj in
                                           multiplexall_obj.multiplex_tuple]

        self.multiplexall_node_list2d = [multiplexone_obj.nodes for multiplexone_obj in
                                         multiplexall_obj.multiplex_tuple]

        # self. N nb of nodes in each multiplex
        # self.N = list()
        self.multiplexall_node_count_list = [len(x) for x in self.multiplexall_node_list2d]

        #######################################################################
        #
        # seed object from config_parser and properties
        #
        #######################################################################
        if type(self.pr) == type(None):
            self.seed_obj = config_parser_obj.seed_obj
        else:
            N = copy.deepcopy(self.multiplexall_node_count_list)
            N.insert(0, 0)
            L = self.multiplex_layer_count_list
            temp = [numpy.repeat((self.pr[numpy.sum(N[:i]):numpy.sum(N[:i + 1])] / L[i - 1]), L[i - 1]) for i in
                    range(1, len(L) + 1)]
            self.pr = numpy.concatenate(temp)

        ###
        self.node_to_multiplex_layer = self.__create_node_to_multiplex_layer_mapping()
        self.node_to_index, self.layer_index_map = self.__create_node_index_mapping()

    # 1.3.3 :
    def __random_walk_restart(self, prox_vector, transition_matrixcoo, r):
        """

        Function that realize the RWR and give back the steady probability distribution for
        each multiplex in a dataframe.

        self.results (list) : A list of ndarray. Each ndarray correspond to the probability distribution of the
            nodes of the multiplex.

        """
        rwr_result_lst = list()
        threshold = 1e-10
        residue = 1
        itera = 1
        prox_vector_norm = prox_vector / (sum(prox_vector))
        restart_vector = prox_vector_norm
        while residue >= threshold:
            old_prox_vector = prox_vector_norm
            prox_vector_norm = (1 - r) * (transition_matrixcoo.dot(prox_vector_norm)) + r * restart_vector
            residue = numpy.sqrt(sum((prox_vector_norm - old_prox_vector) ** 2))
            itera += 1
        for k in range(len(self.multiplex_layer_count_list)):
            start = sum(
                numpy.array(self.multiplexall_node_count_list[:k]) * numpy.array(self.multiplex_layer_count_list[:k]))
            end = start + self.multiplexall_node_count_list[k] * self.multiplex_layer_count_list[k]
            data = numpy.array(prox_vector_norm[start:end])
            rwr_result_lst.append(data)
        return rwr_result_lst

    def __get_node_multiplex_and_layer(self, node: int, to_layer_index: Optional[int] = None) -> \
            Tuple[Optional[str], Optional[str]]:
        """
        Get the multiplex and layer of a given node.

        Args:
            node (int): The index of the node.
            to_layer_index (Optional[int]): Index of the layer to retrieve. Default is None.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the multiplex and layer keys.
                If the node is not found, returns (None, None).
                If to_layer_index is provided and valid, returns the multiplex and layer keys at that index.
                If to_layer_index is not provided, returns a random choice of multiplex and layer keys for the node.
        """
        multiplex_layer_tuples = self.node_to_multiplex_layer.get(node, [])

        if multiplex_layer_tuples:
            # If to_layer_index is provided, we only need to return the multiplex and layer at that index
            if to_layer_index is not None:
                return multiplex_layer_tuples[to_layer_index]
            return random.choice(multiplex_layer_tuples)

        else:
            return None, None

    def __create_node_to_multiplex_layer_mapping(self) -> Dict[int, List[Tuple[str, str]]]:
        """
        Create mapping of nodes to their corresponding multiplex and layer.

        Returns:
            Dict[int, List[Tuple[str, str]]]: A dictionary mapping node indices to lists of tuples.
                Each tuple contains the key of the multiplex and the key of the layer the node belongs to.
        """
        node_to_multiplex_layer = {}
        for i, multiplex in enumerate(self.multiplexall_obj.multiplex_tuple):
            for layer in multiplex.layer_tuple:
                for node in layer.networkx.nodes:
                    node = int(node)
                    if node in node_to_multiplex_layer:
                        node_to_multiplex_layer[node].append((multiplex.key, layer.key))
                    else:
                        node_to_multiplex_layer[node] = [(multiplex.key, layer.key)]

        return node_to_multiplex_layer

    def __create_node_index_mapping(self) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        """
        Create mapping of nodes ID to their indices and layers to their indices.

        Returns:
            Tuple[Dict[int, List[int]], Dict[str, int]]: A tuple containing two dictionaries:
                - First dictionary maps node ID to lists of their indices across layers.
                - Second dictionary maps layer keys to their indices.
        """
        node_index_map = {}
        layer_index_map = {}
        index_counter = 0
        layer_counter = 0

        for multiplex in self.multiplexall_obj.multiplex_tuple:
            for layer in multiplex.layer_tuple:
                if layer not in layer_index_map:
                    layer_index_map[layer.key] = layer_counter
                    layer_counter += 1
                for node in multiplex.nodes:
                    node = int(node)
                    if node not in node_index_map:
                        node_index_map[node] = []
                    node_index_map[node].append(index_counter)
                    index_counter += 1
            layer_counter = 0
        return node_index_map, layer_index_map

    def __simulated_random_walk_with_restart(self, transition_matrixcoo: scipy.sparse.coo_matrix, seeds: List[object],
                                             r: float, num_walks: int = 100, max_steps: int = 100) -> pandas.DataFrame:
        """
        Simulate a random walk with restart from the given seed nodes and logs each step.

        Args:
            transition_matrixcoo (scipy.sparse.coo_matrix): The transition matrix in COO format.
            seeds (list): List of node indices representing seed nodes.
            r (float): Restart probability.
            num_walks (int): Number of random walks to perform. Default is 100.
            max_steps (int): Maximum number of steps for each walk. Default is 100.

        Returns:
            pandas.DataFrame: DataFrame logging each walk's step with columns:
                ['walk_id', 'step', 'from_node', 'from_multiplex', 'from_layer', 'to_node', 'to_multiplex', 'to_layer'].
        """

        def single_walk(walk_id: int) -> List[List[Union[int, str]]]:
            """
            Simulate a single random walk with restart from a given seed node and logs each step.

            Args:
                walk_id (int): ID of the walk.

            Returns:
                List[List[Union[int, str]]]: List of lists, each sublist containing the details of a step in the
                walk.
                Each sublist has the format: [walk_id, step, from_node, from_multiplex, from_layer, to_node,
                to_multiplex, to_layer].
            """
            walk_logs = []
            # Invert mapping to get node ID to from given index, one node ID can have multiple index by layer
            index_to_node = {index: node for node, indices in self.node_to_index.items() for index in
                             indices}

            # Each walk starts from a random seed node, randomly select its layer
            current_node = int(numpy.random.choice(seeds))
            current_multiplex, current_layer = self.__get_node_multiplex_and_layer(current_node)

            # we need layer index to get the index of the node in the layer
            current_layer_index = self.layer_index_map[current_layer]
            current_index = self.node_to_index[current_node][current_layer_index]

            for step in range(max_steps):
                # Restart with probability r and select a seed node with random layer
                if numpy.random.random() < r:
                    to_node = int(numpy.random.choice(seeds))
                    to_multiplex, to_layer = self.__get_node_multiplex_and_layer(to_node)
                    to_layer_index = self.layer_index_map[to_layer]
                    to_index = self.node_to_index[to_node][to_layer_index]
                else:
                    # Determine the next node based on transition probabilities
                    possible_transitions = [(i, j, val) for i, j, val in zip(transition_matrixcoo.row,
                                                                             transition_matrixcoo.col,
                                                                             transition_matrixcoo.data) if
                                            i == current_index]
                    # restart if no outgoing edges (possible transitions)
                    if not possible_transitions:
                        to_node = int(numpy.random.choice(seeds))
                        to_multiplex, to_layer = self.__get_node_multiplex_and_layer(to_node)
                        to_layer_index = self.layer_index_map[to_layer]
                        to_index = self.node_to_index[to_node][to_layer_index]
                    else:
                        # select index of the next node based on the transition probabilities, use index to find node ID
                        to_index = int(random.choices(population=[j for _, j, _ in possible_transitions],
                                                      weights=[val for _, _, val in possible_transitions], k=1)[0])
                        to_node = index_to_node[to_index]
                        node_indices = self.node_to_index.get(to_node)
                        if node_indices is not None:
                            # Find the index of the given index within the node indices
                            to_layer_index = node_indices.index(to_index)

                        # find the matching multiplex and layer determined by the index
                        to_multiplex, to_layer = self.__get_node_multiplex_and_layer(to_node, current_layer,
                                                                                     current_multiplex, to_layer_index)
                # log current step
                walk_logs.append([walk_id, step, current_node, current_multiplex, current_layer,
                                  to_node, to_multiplex, to_layer])
                # Move to next node
                current_node, current_multiplex, current_layer, current_layer_index, current_index, = to_node, \
                                                                                                      to_multiplex, \
                                                                                                      to_layer, \
                                                                                                      to_layer_index, \
                                                                                                      to_index

            return walk_logs

        # Perform random walks in parallel to speed up the process
        with ThreadPoolExecutor() as executor:
            results = executor.map(single_walk, range(num_walks))

        flat_results = [item for sublist in results for item in sublist]

        # Convert logs to DataFrame
        walk_logs_df = pandas.DataFrame(flat_results, columns=['walk_id', 'step', 'from_node', 'from_multiplex',
                                                               'from_layer', 'to_node', 'to_multiplex', 'to_layer'])
        return walk_logs_df

    ###########################################################################
    # 2 :Analysis func##############################tions
    ###########################################################################

    # 2.1 :
    ###########################################################################

    # 2.1.1 :
    def random_walk_rank(self) -> pandas.DataFrame:
        """
        Function that carries ous the full random walk with restart from a list of seeds.

        Returns :
                rwr_ranking_df (pandas.DataFrame) : A pandas Dataframe with columns: multiplex, node, layer, score
        """
        bipartite_matrix = self.bipartiteall_obj.bipartite_matrix
        transition_matrix_obj = TransitionMatrix(multiplex_all=self.multiplexall_obj, bipartite_matrix=bipartite_matrix,
                                                 lamb=self.lamb)
        transition_matrixcoo = transition_matrix_obj.transition_matrixcoo

        # Get initial seed probability distribution
        if type(self.pr) == type(None):
            prox_vector, seed_score = self.seed_obj.get_seed_scores(transition=transition_matrixcoo)
        else:
            prox_vector = self.pr
        # Run RWR algorithm
        rwr_ranking_lst = self.__random_walk_restart(prox_vector, transition_matrixcoo, self.r)
        rwr_ranking_df = self.__random_walk_rank_lst_to_df(rwr_result_lst=rwr_ranking_lst)

        return rwr_ranking_df

    def simulate_random_walk_with_restart(self, max_walk: int = 100, max_step: int = 100) -> pandas.DataFrame:
        """
        Simulate random walks with restart and return a DataFrame with the walk details.

        Args:
            max_walk (int): Maximum number of random walks to perform. Default is 100.
            max_step (int): Maximum number of steps for each walk. Default is 100.

        Returns:
            pandas.DataFrame: DataFrame logging each walk's step with columns:
                ['walk_id', 'step', 'from_node', 'from_multiplex', 'from_layer', 'to_node', 'to_multiplex', 'to_layer'].
        """
        bipartite_matrix = self.bipartiteall_obj.bipartite_matrix
        transition_matrix_obj = TransitionMatrix(multiplex_all=self.multiplexall_obj, bipartite_matrix=bipartite_matrix,
                                                 lamb=self.lamb)
        transition_matrixcoo = transition_matrix_obj.transition_matrixcoo

        seeds = self.seed_obj.seed_list

        random_walk_df = self.__simulated_random_walk_with_restart(transition_matrixcoo, seeds, self.r, max_walk,
                                                                   max_step)
        return random_walk_df

    def write_ranking(self, random_walk_rank: pandas.DataFrame, path: str, top: int = None, aggregation: str = "gmean",
                      degree: bool = False):
        """Writes the 'random walk results' to a subnetwork with the 'top' nodes as a SIF format (See Cytoscape documentation)

        Args:
            rwr_ranking_df (pandas.DataFrame) : A pandas Dataframe with columns: multiplex, node, layer, score, which is the output of the random_walk_rank function
            path (str): Path to the SIF file
            top (int): Top nodes based on the random walk score to be included in the SIF file
            aggregation (str): One of "nomean", "gmean", "hmean", "mean", or "sum"
        """

        if not (aggregation in ['nomean', 'gmean', 'hmean', 'mean', 'sum']):
            logger.error(
                'Aggregation parameter must take one of these values: "nomean", "gmean", "hmean", "mean", or "sum". '
                'Current value: {}'.format(aggregation))
            sys.exit(1)

        output_obj = Output(random_walk_rank, self.multiplexall_obj, top=top, top_type="layered",
                            aggregation=aggregation)
        output_obj.to_tsv(outdir=path, degree=degree)

    def to_sif(self, random_walk_rank: pandas.DataFrame, path: str, top: int = None, top_type: str = 'layered',
               aggregation: str = 'gmean'):
        """Writes the 'random walk results' to a subnetwork with the 'top' nodes as a SIF format (See Cytoscape documentation)

        Args:
            rwr_ranking_df (pandas.DataFrame) : A pandas Dataframe with columns: multiplex, node, layer, score, which is the output of the random_walk_rank function
            path (str): Path to the TSV file with the random walk results
            top (int): Top nodes based on the random walk score to be included in the TSV file
            top_type (str): "per layer" (top nodes for each layer) or "all" (top nodes any layer)
            aggregation (str): One of "none", "geometric mean" or "sum"
        """

        if not (aggregation in ['nomean', 'gmean', 'hmean', 'mean', 'sum']):
            logger.error(
                'Aggregation parameter must take one of these values: "nomean", "gmean", "hmean", "mean", or "sum". '
                'Current value: {}'.format(aggregation))
            sys.exit(1)

        if not (top_type in ['layered', 'all']):
            logger.error('top_type parameter must take one of these values: "layered" or "all". '
                         'Current value: {}'.format(top_type))
            sys.exit(1)

        output_obj = Output(random_walk_rank, self.multiplexall_obj, top=top, top_type=top_type,
                            aggregation=aggregation)
        pathlib.Path(os.path.dirname(path)).mkdir(exist_ok=True, parents=True)
        output_obj.to_sif(path=path, bipartiteall=self.bipartiteall_obj)

    def __random_walk_rank_lst_to_df(self, rwr_result_lst) -> pandas.DataFrame:
        rwrrestart_df = pandas.DataFrame({'multiplex': [], 'node': [], 'layer': [], 'score': []})
        for i, multiplex in enumerate(self.multiplexall_obj.multiplex_tuple):
            multiplex_label_lst = [multiplex.key] * len(multiplex.nodes) * len(
                multiplex.layer_tuple)
            nodes = [item for subl in [multiplex.nodes] * len(multiplex.layer_tuple) for item in
                     subl]
            layer_lst = [item for subl in
                         [[layer.key] * len(multiplex.nodes) for layer in multiplex.layer_tuple] for
                         item in subl]
            if type(self.pr) == type(None):
                score = list(rwr_result_lst[i].T[0])
            else:
                score = list(rwr_result_lst[i].T)
            rwrrestart_df = pandas.concat([rwrrestart_df, pandas.DataFrame(
                {'multiplex': multiplex_label_lst, 'node': nodes, 'layer': layer_lst, 'score': score})], axis=0)
        return rwrrestart_df


class Example:

    def __init__(self):
        """Initiates example class"""
        self.package_path = PathManager.get_package_path()
        self.airport_input_path = os.path.join(self.package_path, 'data_example', 'airport')

    def write(self, path):
        """Writes file tree of working example to 'path' directory

        Args:
            path (str): Path to the output directory
        """
        try:
            shutil.copytree(self.airport_input_path, path)
        except FileExistsError:
            logger.error('Directory exists: {}'.format(path))
