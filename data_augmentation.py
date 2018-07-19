from utils import *
from copy import deepcopy
import random

# Generate the mask based on the valences and adjacent matrix so far
# For a (node_in_focus, neighbor, edge_type) to be valid, neighbor's color < 2 and 
# there is no edge so far between node_in_focus and neighbor and it satisfy the valence constraint
# and node_in_focus != neighbor 
def generate_mask(valences, adj_mat, color, real_n_vertices, node_in_focus, check_overlap_edge, new_mol):
    edge_type_mask=[]
    edge_mask=[]
    for neighbor in range(real_n_vertices):
        if neighbor != node_in_focus and color[neighbor] < 2 and \
            not check_adjacent_sparse(adj_mat, node_in_focus, neighbor)[0]:
            min_valence = min(valences[node_in_focus], valences[neighbor], 3)
            # Check whether two cycles have more than two overlap edges here
            # the neighbor color = 1 and there are left valences and 
            # adding that edge will not cause overlap edges.
            if check_overlap_edge and min_valence > 0 and color[neighbor] == 1:
                # attempt to add the edge
                new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[0])
                # Check whether there are two cycles having more than two overlap edges
                ssr = Chem.GetSymmSSSR(new_mol)
                overlap_flag = False
                for idx1 in range(len(ssr)):
                    for idx2 in range(idx1+1, len(ssr)):
                        if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                            overlap_flag=True
                # remove that edge
                new_mol.RemoveBond(int(node_in_focus), int(neighbor))
                if overlap_flag:
                    continue
            for v in range(min_valence):
                assert v < 3
                edge_type_mask.append((node_in_focus, neighbor, v))
            # there might be an edge between node in focus and neighbor
            if min_valence > 0:
                edge_mask.append((node_in_focus, neighbor))
    return edge_type_mask, edge_mask

# when a new edge is about to be added, we generate labels based on ground truth
# if an edge is in ground truth and has not been added to incremental adj yet, we label it as positive
def generate_label(ground_truth_graph, incremental_adj, node_in_focus, real_neighbor, real_n_vertices, params):
    edge_type_label=[]
    edge_label=[]
    for neighbor in range(real_n_vertices):
        adjacent, edge_type = check_adjacent_sparse(ground_truth_graph, node_in_focus, neighbor)
        incre_adjacent, incre_edge_type = check_adjacent_sparse(incremental_adj, node_in_focus, neighbor)
        if not params["label_one_hot"] and adjacent and not incre_adjacent:
            assert edge_type < 3
            edge_type_label.append((node_in_focus, neighbor, edge_type))
            edge_label.append((node_in_focus, neighbor))
        elif params["label_one_hot"] and adjacent and not incre_adjacent and neighbor==real_neighbor:
            edge_type_label.append((node_in_focus, neighbor, edge_type))
            edge_label.append((node_in_focus, neighbor))    
    return edge_type_label, edge_label

# add a incremental adj with one new edge
def genereate_incremental_adj(last_adj, node_in_focus, neighbor, edge_type):
    # copy last incremental adj matrix
    new_adj= deepcopy(last_adj)
    # Add a new edge into it
    new_adj[node_in_focus].append((neighbor, edge_type))
    new_adj[neighbor].append((node_in_focus, edge_type))
    return new_adj

def update_one_step(overlapped_edge_features, distance_to_others,node_sequence, node_in_focus, neighbor, edge_type, edge_type_masks, valences, incremental_adj_mat,
                    color, real_n_vertices, graph, edge_type_labels, local_stop, edge_masks, edge_labels, local_stop_label, params,
                    check_overlap_edge, new_mol, up_to_date_adj_mat,keep_prob):
    # check whether to keep this transition or not
    if params["sample_transition"] and random.random()> keep_prob:
        return
    # record the current node in focus
    node_sequence.append(node_in_focus)
    # generate mask based on current situation
    edge_type_mask, edge_mask=generate_mask(valences, up_to_date_adj_mat,
                               color,real_n_vertices, node_in_focus, check_overlap_edge, new_mol)
    edge_type_masks.append(edge_type_mask)
    edge_masks.append(edge_mask)
    if not local_stop_label:
        # generate the label based on ground truth graph
        edge_type_label, edge_label=generate_label(graph, up_to_date_adj_mat, node_in_focus, neighbor,real_n_vertices, params)
        edge_type_labels.append(edge_type_label)
        edge_labels.append(edge_label)
    else:
        edge_type_labels.append([])
        edge_labels.append([])
    # update local stop 
    local_stop.append(local_stop_label)                            
    # Calculate distance using bfs from the current node to all other node
    distances = bfs_distance(node_in_focus, up_to_date_adj_mat)
    distances = [(start, node, params["truncate_distance"]) if d > params["truncate_distance"] else (start, node, d) for start, node, d in distances]
    distance_to_others.append(distances)
    # Calculate the overlapped edge mask
    overlapped_edge_features.append(get_overlapped_edge_feature(edge_mask, color, new_mol))
    # update the incremental adj mat at this step
    incremental_adj_mat.append(deepcopy(up_to_date_adj_mat))

def construct_incremental_graph(dataset, edges, max_n_vertices, real_n_vertices, node_symbol, params, initial_idx=0):
    # avoid calculating this if it is just for generating new molecules for speeding up
    if params["generation"]:
        return [], [], [], [], [], [], [], [], []
    # avoid the initial index is larger than real_n_vertices:
    if initial_idx >= real_n_vertices:
        initial_idx=0
    # Maximum valences for each node
    valences=get_initial_valence([np.argmax(symbol) for symbol in node_symbol], dataset)
    # Add backward edges
    edges_bw=[(dst, edge_type, src) for src, edge_type, dst in edges]
    edges=edges+edges_bw
    # Construct a graph object using the edges
    graph=defaultdict(list)
    for src, edge_type, dst in edges:
        graph[src].append((dst, edge_type))
    # Breadth first search over the molecule 
    # color 0: have not found 1: in the queue 2: searched already
    color = [0] * max_n_vertices
    color[initial_idx] = 1
    queue=deque([initial_idx])
    # create a adj matrix without any edges
    up_to_date_adj_mat=defaultdict(list)
    # record incremental adj mat
    incremental_adj_mat=[]
    # record the distance to other nodes at the moment
    distance_to_others=[]
    # soft constraint on overlapped edges
    overlapped_edge_features=[]
    # the exploration order of the nodes
    node_sequence=[]
    # edge type masks for nn predictions at each step
    edge_type_masks=[]
    # edge type labels for nn predictions at each step
    edge_type_labels=[]
    # edge masks for nn predictions at each step
    edge_masks=[]
    # edge labels for nn predictions at each step
    edge_labels=[]
    # local stop labels
    local_stop=[]
    # record the incremental molecule
    new_mol = Chem.MolFromSmiles('')
    new_mol = Chem.rdchem.RWMol(new_mol)
    # Add atoms
    add_atoms(new_mol, sample_node_symbol([node_symbol], [len(node_symbol)], dataset)[0], dataset)
    # calculate keep probability
    sample_transition_count= real_n_vertices + len(edges)/2
    keep_prob= float(sample_transition_count)/((real_n_vertices + len(edges)/2) * params["bfs_path_count"])   # to form a binomial distribution
    while len(queue) > 0:
        node_in_focus=queue.popleft()
        current_adj_list=graph[node_in_focus]
        # sort (canonical order) it or shuffle (random order) it 
        if not params["path_random_order"]:
            current_adj_list=sorted(current_adj_list)
        else:
            random.shuffle(current_adj_list)
        for neighbor, edge_type in current_adj_list:
            # Add this edge if the color of neighbor node is not 2
            if color[neighbor]<2:
                update_one_step(overlapped_edge_features, distance_to_others,node_sequence, node_in_focus, neighbor, edge_type,  
                         edge_type_masks, valences, incremental_adj_mat, color, real_n_vertices, graph, 
                         edge_type_labels, local_stop, edge_masks, edge_labels, False, params, params["check_overlap_edge"], new_mol, 
                         up_to_date_adj_mat,keep_prob)
                # Add the edge and obtain a new adj mat
                up_to_date_adj_mat=genereate_incremental_adj(
                                   up_to_date_adj_mat, node_in_focus, neighbor, edge_type)
                # suppose the edge is selected and update valences after adding the 
                valences[node_in_focus]-=(edge_type + 1)
                valences[neighbor]-=(edge_type + 1)
                # update the incremental mol
                new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[edge_type])
            # Explore neighbor nodes
            if color[neighbor]==0:
                queue.append(neighbor)
                color[neighbor]=1
        # local stop here. We move on to another node for exploration or stop completely
        update_one_step(overlapped_edge_features, distance_to_others,node_sequence, node_in_focus, None, None, edge_type_masks, 
                        valences, incremental_adj_mat, color, real_n_vertices, graph,
                        edge_type_labels, local_stop, edge_masks, edge_labels, True, params, params["check_overlap_edge"], new_mol, up_to_date_adj_mat,keep_prob)
        color[node_in_focus]=2
        
    return incremental_adj_mat,distance_to_others,node_sequence,edge_type_masks,edge_type_labels,local_stop, edge_masks, edge_labels, overlapped_edge_features
