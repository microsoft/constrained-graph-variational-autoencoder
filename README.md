# Constrained Graph Variational Autoencoders for Molecule Design

This repository contains our implementation of [Constrained Graph Variational Autoencoders for Molecule Design](https://arxiv.org/abs/1805.09076) (CGVAE). 

```
@article{liu2018constrained,
  title={Constrained Graph Variational Autoencoders for Molecule Design},
  author={Liu, Qi and Allamanis, Miltiadis and Brockschmidt, Marc and Gaunt, Alexander L.},
  journal={The Thirty-second Conference on Neural Information Processing Systems},
  year={2018}
}
```

# Requirements

This code was tested in Python 3.5 with Tensorflow 1.3. `conda`, `docopt` and `rdkit` are also necessary. A Bash script is provided to install all these requirements.

```
source ./install.sh
```

To evaluate SAS scores, use `get_sascorer.sh` to download the SAS implementation from [rdkit](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score)

# Data Extraction

Three datasets (QM9, ZINC and CEPDB) are in use. For downloading CEPDB, please refer to [CEPDB](http://cleanenergy.molecularspace.org/).

For downloading QM9 and ZINC, please go to `data` directory and run `get_qm9.py` and `get_zinc.py`, respectively.

```
python get_qm9.py

python get_zinc.py
```

# Running CGVAE

We provide two settings of CGVAE. The first setting samples one breadth first search path for each molecule. The second setting samples transitions from multiple breadth first search paths for each molecule. 

To train and generate molecules using the first setting, use

```
python CGVAE.py --dataset qm9|zinc|cep
```

To avoid training and generate molecules with a pretrained model, use

```
python CGVAE.py --dataset qm9|zinc|cep --restore pretrained_model --config '{"generation": true}'
```

To train and generate molecules using the second setting, use

```
python CGVAE.py --dataset qm9|zinc|cep --config '{"sample_transition": true, "multi_bfs_path": true, "path_random_order": true}'
```

To use optimization in the latent space, set `optimization_step` to a positive number

```
python CGVAE.py --dataset qm9|zinc|cep --restore pretrained_model --config '{"generation": true, "optimization_step": 50}'
```

More configurations can be found at function `default_params` in `CGVAE.py`

# Evaluation

To evaluate the generated molecules, use

```
python evaluate.py --dataset qm9|zinc|cep
```

# Pretrained Models and Generated Molecules
<!--
We provide pretrained models and generated molecules for both settings. The following files are pretrained models

```
pretrained/qm9_setting1
pretrained/qm9_setting2
pretrained/zinc_setting1
pretrained/zinc_setting2
pretrained/cep_setting1
pretrained/cep_setting2
```

The following files are generated molecules

```
molecules/generated_smiles_qm9_setting1
molecules/generated_smiles_qm9_setting2
molecules/generated_smiles_zinc_setting1
molecules/generated_smiles_zinc_setting2
molecules/generated_smiles_cep_setting1
molecules/generated_smiles_cep_setting2
```
-->

Generated molecules can be obtained upon request.

A program in folder `molecules` is provided to read and visualize the molecules

```
python visualize.py molecule_file output_file
```

# Questions/Bugs

Please submit a Github issue or contact [qiliu@u.nus.edu](mailto:qiliu@u.nus.edu).

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
