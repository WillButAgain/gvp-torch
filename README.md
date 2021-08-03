# gvp-torch
implementation of geometric vector perceptrons in pytorch. 

existing implementations are - 
https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/__init__.py#L279

which does not seem to allow training/inference with .PDB files, which is weird

and 
https://github.com/lucidrains/geometric-vector-perceptron -

which uses atom-wise representations of proteins

mine allows training/inference with .PDB files, and it uses per-residue representations of proteins 

if you used this work for commercial purposes, consider donating to your local homeless shelter
