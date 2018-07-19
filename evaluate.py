#!/usr/bin/env/python
"""
Usage:
    evaluate.py --dataset zinc|qm9|cep

Options:
    -h --help                Show this screen.
    --dataset NAME           Dataset name: zinc, qm9, cep
"""

import utils
from utils import dataset_info
import numpy as np 
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)
    dataset=args.get('--dataset')
    logpscorer, logp_score_per_molecule=utils.check_logp(dataset)
    qedscorer, qed_score_per_molecule=utils.check_qed(dataset)
    novelty=utils.novelty_metric(dataset)
    total, nonplanar=utils.check_planar(dataset)
    total, atom_counter, atom_per_molecule =utils.count_atoms(dataset)
    total, edge_type_counter, edge_type_per_molecule=utils.count_edge_type(dataset)
    total, shape_count, shape_count_per_molecule=utils.shape_count(dataset)
    total, tree_count=utils.check_cyclic(dataset)    
    sascorer, sa_score_per_molecule=utils.check_sascorer(dataset)
    total, validity=utils.check_validity(dataset)

    print("------------------------------------------")
    print("Metrics")
    print("------------------------------------------")
    print("total molecule")
    print(total)
    print("------------------------------------------")
    print("percentage of nonplanar:")
    print(nonplanar/total)
    print("------------------------------------------")
    print("avg atom:")
    for atom_type, c in atom_counter.items():
        print(dataset_info(dataset)['atom_types'][atom_type])
        print(c/total)
    print("standard deviation")
    print(np.std(atom_per_molecule, axis=0))
    print("------------------------------------------")
    print("avg edge_type:")
    for edge_type, c in edge_type_counter.items():
        print(edge_type+1)
        print(c/total)
    print("standard deviation")
    print(np.std(edge_type_per_molecule, axis=0))
    print("------------------------------------------")
    print("avg shape:")
    for shape, c in zip(utils.geometry_numbers, shape_count):
        print(shape)
        print(c/total)
    print("standard deviation")
    print(np.std(shape_count_per_molecule, axis=0))
    print("------------------------------------------")
    print("percentage of tree:")
    print(tree_count/total)
    print("------------------------------------------")
    print("percentage of validity:")
    print(validity/total)
    print("------------------------------------------")
    print("avg sa_score:")
    print(sascorer)
    print("standard deviation")
    print(np.std(sa_score_per_molecule))
    print("------------------------------------------")
    print("avg logp_score:")
    print(logpscorer)
    print("standard deviation")
    print(np.std(logp_score_per_molecule))
    print("------------------------------------------")
    print("percentage of novelty:")
    print(novelty)
    print("------------------------------------------")
    print("avg qed_score:")
    print(qedscorer)
    print("standard deviation")
    print(np.std(qed_score_per_molecule))
    print("------------------------------------------")
    print("uniqueness")
    print(utils.check_uniqueness(dataset))
    print("------------------------------------------")
    print("percentage of SSSR")
    print(utils.sssr_metric(dataset))
