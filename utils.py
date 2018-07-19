#!/usr/bin/env/python
import numpy as np
import tensorflow as tf
import queue
import threading
import pickle
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict, deque
import os
import heapq
import planarity
import sascorer
from rdkit.Chem import Crippen
from rdkit.Chem import QED

SMALL_NUMBER = 1e-7
LARGE_NUMBER= 1e10

geometry_numbers=[3, 4, 5, 6] # triangle, square, pentagen, hexagon

# bond mapping
bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
number_to_bond= {0: Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE, 
                 2: Chem.rdchem.BondType.TRIPLE, 3:Chem.rdchem.BondType.AROMATIC}

def dataset_info(dataset): #qm9, zinc, cep
    if dataset=='qm9':
        return { 'atom_types': ["H", "C", "N", "O", "F"],
                 'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                 'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                 'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
               }
    elif dataset=='zinc':
        return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
                 'bucket_sizes': np.array([28,31,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,55,58,84])
               }
    
    elif dataset=="cep":
        return { 'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                 'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                 'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                 'bucket_sizes': np.array([25,28,29,30, 32, 33,34,35,36,37,38,39,43,46])
               }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)

# add one edge to adj matrix
def add_edge_mat(amat, src, dest, e, considering_edge_type=True):
    if considering_edge_type:
        amat[e, dest, src] = 1
        amat[e, src, dest] = 1
    else:
        amat[src, dest] = 1
        amat[dest, src] = 1 

def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True, considering_edge_type=True):
    if considering_edge_type:
        amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e)
    else:
        amat = np.zeros((max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e, considering_edge_type=False)
    return amat

def check_edge_prob(dataset):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    for ep, epl in zip(edge_prob, edge_prob_label):
        print("prediction")
        print(ep)
        print("label")
        print(epl)

# check whether a graph is planar or not
def is_planar(location, adj_list, is_dense=False):
    if is_dense:
        new_adj_list=defaultdict(list)
        for x in range(len(adj_list)):
            for y in range(len(adj_list)):
                if adj_list[x][y]==1:
                    new_adj_list[x].append((y,1))
        adj_list=new_adj_list
    edges=[]
    seen=set()
    for src, l in adj_list.items():
        for dst, e in l:
            if (dst, src) not in seen:
                edges.append((src,dst))
                seen.add((src,dst))
    edges+=[location, (location[1], location[0])]
    return planarity.is_planar(edges)

def check_edge_type_prob(filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    for ep, epl in zip(edge_type_prob, edge_type_label):
        print("prediction")
        print(ep)
        print("label")
        print(epl)    

def check_mean(dataset, filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(mean.tolist()[:40])

def check_variance(dataset, filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(np.exp(logvariance).tolist()[:40])

def check_node_prob(filter=None):
    print(dataset)
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(node_symbol_prob[0])
    print(node_symbol[0])
    print(node_symbol_prob.shape)

def check_qed(filter=None):
    with open('intermediate_results_%s' % dataset, 'rb') as f:
        adjacency_matrix, edge_type_prob, edge_type_label, node_symbol_prob, node_symbol, edge_prob, edge_prob_label, qed_prediction, qed_labels,mean, logvariance=pickle.load(f)
    print(qed_prediction)
    print(qed_labels[0])
    print(np.mean(np.abs(qed_prediction-qed_labels[0])))

def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z
    
def generate_empty_adj_matrix(maximum_vertice_num):
    return np.zeros((1, 3, maximum_vertice_num, maximum_vertice_num))

# standard normal with shape [a1, a2, a3]
def generate_std_normal(a1, a2, a3):
    return np.random.normal(0, 1, [a1, a2, a3])
    
def check_validity(dataset):       
    with open('generated_smiles_%s' % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    count=0
    for smiles in all_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            count+=1
    return len(all_smiles), count

# Get length for each graph based on node masks
def get_graph_length(all_node_mask):
    all_lengths=[]
    for graph in all_node_mask:
        if 0 in graph:
            length=np.argmin(graph)
        else:
            length=len(graph)
        all_lengths.append(length)
    return all_lengths

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('made directory %s' % path)

# sample node symbols based on node predictions
def sample_node_symbol(all_node_symbol_prob, all_lengths, dataset):
    all_node_symbol=[]
    for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
        node_symbol=[]
        for node_idx in range(all_lengths[graph_idx]):
            symbol=np.random.choice(np.arange(len(dataset_info(dataset)['atom_types'])), p=graph_prob[node_idx])
            node_symbol.append(symbol)
        all_node_symbol.append(node_symbol)
    return all_node_symbol

def dump(file_name, content):
    with open(file_name, 'wb') as out_file:        
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)
        
def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)    

# generate a new feature on whether adding the edges will generate more than two overlapped edges for rings
def get_overlapped_edge_feature(edge_mask, color, new_mol):
    overlapped_edge_feature=[]
    for node_in_focus, neighbor in edge_mask:
        if color[neighbor] == 1:
            # attempt to add the edge
            new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[0])
            # Check whether there are two cycles having more than two overlap edges
            try:
                ssr = Chem.GetSymmSSSR(new_mol)
            except:
                ssr = []
            overlap_flag = False
            for idx1 in range(len(ssr)):
                for idx2 in range(idx1+1, len(ssr)):
                    if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                        overlap_flag=True
            # remove that edge
            new_mol.RemoveBond(int(node_in_focus), int(neighbor))
            if overlap_flag:
                overlapped_edge_feature.append((node_in_focus, neighbor))
    return overlapped_edge_feature

# adj_list [3, v, v] or defaultdict. bfs distance on a graph
def bfs_distance(start, adj_list, is_dense=False):
    distances={}
    visited=set()
    queue=deque([(start, 0)])
    visited.add(start)
    while len(queue) != 0:
        current, d=queue.popleft()
        for neighbor, edge_type in adj_list[current]:
            if neighbor not in visited:
                distances[neighbor]=d+1
                visited.add(neighbor)
                queue.append((neighbor, d+1))
    return [(start, node, d) for node, d in distances.items()]

def get_initial_valence(node_symbol, dataset):
    return [dataset_info(dataset)['maximum_valence'][s] for s in node_symbol]

def add_atoms(new_mol, node_symbol, dataset):
    for number in node_symbol:
        if dataset=='qm9' or dataset=='cep':
            idx=new_mol.AddAtom(Chem.Atom(dataset_info(dataset)['number_to_atom'][number]))
        elif dataset=='zinc':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])            
            charge_num=int(dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            new_mol.AddAtom(new_atom)

def visualize_mol(path, new_mol):
    AllChem.Compute2DCoords(new_mol)
    print(path)
    Draw.MolToFile(new_mol,path)

def get_idx_of_largest_frag(frags):
    return np.argmax([len(frag) for frag in frags])

def remove_extra_nodes(new_mol):
    frags=Chem.rdmolops.GetMolFrags(new_mol)
    while len(frags) > 1:
        # Get the idx of the frag with largest length
        largest_idx = get_idx_of_largest_frag(frags)
        for idx in range(len(frags)):
            if idx != largest_idx:
                # Remove one atom that is not in the largest frag
                new_mol.RemoveAtom(frags[idx][0])
                break
        frags=Chem.rdmolops.GetMolFrags(new_mol)

def novelty_metric(dataset):
    with open('all_smiles_%s.pkl' % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f)) 
    with open('generated_smiles_%s' % dataset, 'rb') as f:
        generated_all_smiles=set(pickle.load(f))
    total_new_molecules=0
    for generated_smiles in generated_all_smiles:
        if generated_smiles not in all_smiles:
            total_new_molecules+=1
    
    return float(total_new_molecules)/len(generated_all_smiles)

def count_edge_type(dataset, generated=True):
    if generated:
        filename='generated_smiles_%s' % dataset
    else:
        filename='all_smiles_%s.pkl' % dataset
    with open(filename, 'rb') as f:
        all_smiles=set(pickle.load(f))

    counter=defaultdict(int)
    edge_type_per_molecule=[]
    for smiles in all_smiles:
        nodes, edges=to_graph(smiles, dataset)
        edge_type_this_molecule=[0]* len(bond_dict)
        for edge in edges:
            edge_type=edge[1]
            edge_type_this_molecule[edge_type]+=1
            counter[edge_type]+=1
        edge_type_per_molecule.append(edge_type_this_molecule)
    total_sum=0
    return len(all_smiles), counter, edge_type_per_molecule

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def check_planar(dataset):
    with open("generated_smiles_%s" % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    total_non_planar=0
    for smiles in all_smiles:
        try:
            nodes, edges=to_graph(smiles, dataset)
        except:
            continue
        edges=[(src, dst) for src, e, dst in edges]
        if edges==[]:
            continue

        if not planarity.is_planar(edges):
            total_non_planar+=1
    return len(all_smiles), total_non_planar

def count_atoms(dataset):
    with open("generated_smiles_%s" % dataset, 'rb') as f:
        all_smiles=set(pickle.load(f))
    counter=defaultdict(int)
    atom_count_per_molecule=[] # record the counts for each molecule
    for smiles in all_smiles:
        try:
            nodes, edges=to_graph(smiles, dataset)
        except:
            continue
        atom_count_this_molecule=[0]*len(dataset_info(dataset)['atom_types'])
        for node in nodes:
            atom_type=np.argmax(node)
            atom_count_this_molecule[atom_type]+=1
            counter[atom_type]+=1
        atom_count_per_molecule.append(atom_count_this_molecule)
    total_sum=0

    return len(all_smiles), counter, atom_count_per_molecule


def to_graph(smiles, dataset):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
        assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        if dataset=='qm9' or dataset=="cep":
            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types'])))
        elif dataset=='zinc': # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)
            
            if atom_str not in dataset_info(dataset)['atom_types']:
                print('unrecognized atom type %s' % atom_str)
                return [], []

            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))

    return nodes, edges

def check_uniqueness(dataset):
    with open('generated_smiles_%s' % dataset, 'rb') as f:
        all_smiles=pickle.load(f)
    original_num = len(all_smiles)
    all_smiles=set(all_smiles)
    new_num = len(all_smiles)
    return new_num/original_num

def shape_count(dataset, remove_print=False, all_smiles=None):
    if all_smiles==None:
        with open('generated_smiles_%s' % dataset, 'rb') as f:
            all_smiles=set(pickle.load(f)) 

    geometry_counts=[0]*len(geometry_numbers)
    geometry_counts_per_molecule=[] # record the geometry counts for each molecule
    for smiles in all_smiles:
        nodes, edges = to_graph(smiles, dataset)
        if len(edges)<=0:
            continue
        new_mol=Chem.MolFromSmiles(smiles)
        
        ssr = Chem.GetSymmSSSR(new_mol)
        counts_for_molecule=[0] * len(geometry_numbers)
        for idx in range(len(ssr)):
            ring_len=len(list(ssr[idx]))
            if ring_len in geometry_numbers:
                geometry_counts[geometry_numbers.index(ring_len)]+=1
                counts_for_molecule[geometry_numbers.index(ring_len)]+=1
        geometry_counts_per_molecule.append(counts_for_molecule)

    return len(all_smiles), geometry_counts, geometry_counts_per_molecule

def check_adjacent_sparse(adj_list, node, neighbor_in_doubt):
    for neighbor, edge_type in adj_list[node]:
        if neighbor == neighbor_in_doubt:
            return True, edge_type
    return False, None

def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int=2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()

# Implements multilayer perceptron
class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden

class Graph():
 
    def __init__(self, V, g):
        self.V = V
        self.graph  = g
 
    def addEdge(self, v, w):
        # Add w to v ist.
        self.graph[v].append(w) 
        # Add v to w list.
        self.graph[w].append(v) 
 
    # A recursive function that uses visited[] 
    # and parent to detect cycle in subgraph 
    # reachable from vertex v.
    def isCyclicUtil(self, v, visited, parent):
 
        # Mark current node as visited
        visited[v] = True
 
        # Recur for all the vertices adjacent 
        # for this vertex
        for i in self.graph[v]:
            # If an adjacent is not visited, 
            # then recur for that adjacent
            if visited[i] == False:
                if self.isCyclicUtil(i, visited, v) == True:
                    return True
 
            # If an adjacent is visited and not 
            # parent of current vertex, then there 
            # is a cycle.
            elif i != parent:
                return True
 
        return False
 
    # Returns true if the graph is a tree, 
    # else false.
    def isTree(self):
        # Mark all the vertices as not visited 
        # and not part of recursion stack
        visited = [False] * self.V
 
        # The call to isCyclicUtil serves multiple 
        # purposes. It returns true if graph reachable 
        # from vertex 0 is cyclcic. It also marks 
        # all vertices reachable from 0.
        if self.isCyclicUtil(0, visited, -1) == True:
            return False
 
        # If we find a vertex which is not reachable
        # from 0 (not marked by isCyclicUtil(), 
        # then we return false
        for i in range(self.V):
            if visited[i] == False:
                return False
 
        return True

# whether whether the graphs has no cycle or not 
def check_cyclic(dataset, generated=True):
    if generated:
        with open("generated_smiles_%s" % dataset, 'rb') as f:
            all_smiles=set(pickle.load(f))
    else:
        with open("all_smiles_%s.pkl" % dataset, 'rb') as f:
            all_smiles=set(pickle.load(f))        
    
    tree_count=0
    for smiles in all_smiles:
        nodes, edges=to_graph(smiles, dataset)
        edges=[(src, dst) for src, e, dst in edges]
        if edges==[]:
            continue
        new_adj_list=defaultdict(list)

        for src, dst in edges:
            new_adj_list[src].append(dst)
            new_adj_list[dst].append(src)
        graph=Graph(len(nodes), new_adj_list)
        if graph.isTree():
            tree_count+=1
    return len(all_smiles), tree_count

def check_sascorer(dataset):
    with open('generated_smiles_%s' % dataset, 'rb') as f:   
        all_smiles=set(pickle.load(f))     
    sa_sum=0
    total=0
    sa_score_per_molecule=[]
    for smiles in all_smiles:
        new_mol=Chem.MolFromSmiles(smiles)
        try:
            val = sascorer.calculateScore(new_mol)
        except:
            continue
        sa_sum+=val
        sa_score_per_molecule.append(val)
        total+=1
    return sa_sum/total, sa_score_per_molecule

def check_logp(dataset):
    with open('generated_smiles_%s' % dataset, 'rb') as f:   
        all_smiles=set(pickle.load(f))
    logp_sum=0
    total=0
    logp_score_per_molecule=[]
    for smiles in all_smiles:
        new_mol=Chem.MolFromSmiles(smiles)
        try:
            val = Crippen.MolLogP(new_mol)
        except:
            continue
        logp_sum+=val
        logp_score_per_molecule.append(val)
        total+=1
    return logp_sum/total, logp_score_per_molecule

def check_qed(dataset):
    with open('generated_smiles_%s' % dataset, 'rb') as f:   
        all_smiles=set(pickle.load(f))
    qed_sum=0
    total=0
    qed_score_per_molecule=[]
    for smiles in all_smiles:
        new_mol=Chem.MolFromSmiles(smiles)
        try:
            val = QED.qed(new_mol)
        except:
            continue
        qed_sum+=val
        qed_score_per_molecule.append(val)
        total+=1
    return qed_sum/total, qed_score_per_molecule

def sssr_metric(dataset):
    with open('generated_smiles_%s' % dataset, 'rb') as f:   
        all_smiles=set(pickle.load(f))
    overlapped_molecule=0
    for smiles in all_smiles:
        new_mol=Chem.MolFromSmiles(smiles)
        ssr = Chem.GetSymmSSSR(new_mol)
        overlap_flag=False
        for idx1 in range(len(ssr)):
            for idx2 in range(idx1+1, len(ssr)):
                if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                    overlap_flag=True
        if overlap_flag:
            overlapped_molecule+=1
    return overlapped_molecule/len(all_smiles)

# select the best based on shapes and probs
def select_best(all_mol):
    # sort by shape
    all_mol=sorted(all_mol)
    best_shape=all_mol[-1][0]
    all_mol=[(p, m) for s, p, m in all_mol if s==best_shape]
    # sort by probs
    all_mol=sorted(all_mol)
    return all_mol[-1][1]


# a series util function converting sparse matrix representation to dense 

def incre_adj_mat_to_dense(incre_adj_mat, num_edge_types, maximum_vertice_num):
    new_incre_adj_mat=[] 
    for sparse_incre_adj_mat in incre_adj_mat:
        dense_incre_adj_mat=np.zeros((num_edge_types, maximum_vertice_num,maximum_vertice_num))
        for current, adj_list in sparse_incre_adj_mat.items():
            for neighbor, edge_type in adj_list:
                dense_incre_adj_mat[edge_type][current][neighbor]=1
        new_incre_adj_mat.append(dense_incre_adj_mat)
    return new_incre_adj_mat # [number_iteration,num_edge_types,maximum_vertice_num, maximum_vertice_num]

def distance_to_others_dense(distance_to_others, maximum_vertice_num):
    new_all_distance=[]
    for sparse_distances in distance_to_others:
        dense_distances=np.zeros((maximum_vertice_num), dtype=int)
        for x, y, d in sparse_distances:
            dense_distances[y]=d
        new_all_distance.append(dense_distances)
    return new_all_distance  # [number_iteration, maximum_vertice_num]

def overlapped_edge_features_to_dense(overlapped_edge_features, maximum_vertice_num):
    new_overlapped_edge_features=[]
    for sparse_overlapped_edge_features in overlapped_edge_features:
        dense_overlapped_edge_features=np.zeros((maximum_vertice_num), dtype=int)
        for node_in_focus, neighbor in sparse_overlapped_edge_features:
            dense_overlapped_edge_features[neighbor]=1
        new_overlapped_edge_features.append(dense_overlapped_edge_features)
    return new_overlapped_edge_features  # [number_iteration, maximum_vertice_num]

def node_sequence_to_dense(node_sequence,maximum_vertice_num):
    new_node_sequence=[]
    for node in node_sequence:
        s=[0]*maximum_vertice_num
        s[node]=1
        new_node_sequence.append(s)
    return new_node_sequence # [number_iteration, maximum_vertice_num]

def edge_type_masks_to_dense(edge_type_masks, maximum_vertice_num, num_edge_types):
    new_edge_type_masks=[]
    for mask_sparse in edge_type_masks:
        mask_dense=np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in mask_sparse:
            mask_dense[bond][neighbor]=1
        new_edge_type_masks.append(mask_dense)
    return new_edge_type_masks #[number_iteration, 3, maximum_vertice_num]

def edge_type_labels_to_dense(edge_type_labels, maximum_vertice_num,num_edge_types):
    new_edge_type_labels=[]
    for labels_sparse in edge_type_labels:
        labels_dense=np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in labels_sparse:
            labels_dense[bond][neighbor]= 1/float(len(labels_sparse)) # fix the probability bug here.
        new_edge_type_labels.append(labels_dense)
    return new_edge_type_labels #[number_iteration, 3, maximum_vertice_num]

def edge_masks_to_dense(edge_masks, maximum_vertice_num):
    new_edge_masks=[]
    for mask_sparse in edge_masks:
        mask_dense=[0] * maximum_vertice_num
        for node_in_focus, neighbor in mask_sparse:
            mask_dense[neighbor]=1
        new_edge_masks.append(mask_dense)
    return new_edge_masks # [number_iteration, maximum_vertice_num]

def edge_labels_to_dense(edge_labels, maximum_vertice_num):
    new_edge_labels=[]
    for label_sparse in edge_labels:
        label_dense=[0] * maximum_vertice_num
        for node_in_focus, neighbor in label_sparse:
            label_dense[neighbor]=1/float(len(label_sparse))
        new_edge_labels.append(label_dense)
    return new_edge_labels # [number_iteration, maximum_vertice_num]