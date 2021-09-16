# BNS
Codes for "Blocking-based Neighbor Sampling for Large-scale Graph Neural Networks" (IJCAI 2021). 

## Requirements
* torch >= 1.8.1
* torch-geometric >= 1.6.3
* torch-scatter >= 2.0.8
* torch-sparse >= 0.6.11
* ogb >= 1.2.6
* hydra-core
* pandas
* numba

Running the codes with other versions of the above library may be ok with slight modification in codes.

## Usage
All configurations are placed in 'conf'. If you want to add configurations of different sampling methods, models and datasets, or you want to change the settings, please feel free to modify corresponding files.

`
python main_ogb_fast.py
`

Before running with ogbn-papers100M, please run the 'preprocess.py' first. If you have any questions, please open an issue, I will reply as soon as possible.

## Notes
1. We optimize the implementations of BNS and improve the results of BNS, so there are slight differences from the original version in our published paper. We recommend you to cite results achieved by this improved version.
* We speedup the sampling process of BNS (so as other baselines).
* We use GraphNorm as the optional normalization operation.
2. Some codes may be merged from other works, but I did not record where they are from (really sorry about this). Please feel free to tell me the sources of some parts in our implementation, and I will cite them in the implementation.

## Citation
<pre>
@inproceedings{DBLP:conf/ijcai/YaoL21,  
  author    = {Kai-Lang Yao and Wu-Jun Li},  
  title     = {Blocking-based Neighbor Sampling for Large-scale Graph Neural Networks},  
  booktitle = {International Joint Conference on Artificial Intelligence},  
  year      = {2021},  
}
</pre>


