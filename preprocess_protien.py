import torch
import prody as pr
import numpy as np

AA_identities = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U', 'B', 'Z', 'X', 'J']
AA_identity_vocab = {word:ind+1 for ind, word in enumerate(AA_identities)} 
AA_identity_vocab['pad'] = 0

def _get_atom_coord_tensor(chain, device='cpu'):
    # residue atoms are C-1, N, CA, C, N+1, CA+1
    # ret is (n_residues, (C-1, N, CA, C, N+1, CA+1), (x, y, z)) / shape (n_residues, 5, 3)
    residue_coords = []
    numResidues = chain.numResidues()
    for i, residue in enumerate(chain.iterResidues()):
        # handle start of sequence
        if not i:
            residue_coords.append([
                3*[np.nan],
                residue['N']._getCoords(),
                residue['CA']._getCoords(),
                residue['C']._getCoords(), 
                residue.getNext()['N']._getCoords(),
                residue.getNext()['CA']._getCoords(),
                3*[np.nan],
            ])
            continue
        # handle end of sequence
        if i+1==numResidues:
            residue_coords.append([
                residue.getPrev()['C']._getCoords(), 
                residue['N']._getCoords(),
                residue['CA']._getCoords(), 
                residue['C']._getCoords(),
                3*[np.nan],
                3*[np.nan],
                residue.getPrev()['CA']._getCoords(),
            ])
            break

        residue_coords.append([
            residue.getPrev()['C']._getCoords(), 
            residue['N']._getCoords(), 
            residue['CA']._getCoords(), 
            residue['C']._getCoords(), 
            residue.getNext()['N']._getCoords(),
            residue.getNext()['CA']._getCoords(),
            residue.getPrev()['CA']._getCoords(),
        ])
    return torch.tensor(residue_coords, device=device)

def get_dihedral(c1, c2, c3, c4):    
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3
    
    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1, u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )

def get_dihedral_tensor(atom_coord_tensor):
    # get Psi dihedrals - N, CA, C, N+1
    psi_dihedrals = get_dihedral(*torch.unbind(atom_coord_tensor[: , 1:5], dim=1))*57.2958
    # get Phi dihedrals - C-1, N, CA, C
    phi_dihedrals = get_dihedral(*torch.unbind(atom_coord_tensor[: , :-3], dim=1))*57.2958
    # get Phi dihedrals - CA, C, N+1, CA+1
    omega_dihedrals = get_dihedral(*torch.unbind(atom_coord_tensor[: , 1:-2], dim=1))*57.2958
    
    psi_phi_omega = torch.stack([psi_dihedrals, phi_dihedrals, omega_dihedrals]).T.to(atom_coord_tensor.device)
    
    # make this return sin(angles), cos(angles) for ret.shape=(n_residues, 6) instead of (n_residues, 3)
    return torch.cat([torch.sin(psi_phi_omega), torch.cos(psi_phi_omega)], dim=-1)

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def get_knn_indices(distances, k=30):
    topk = torch.topk(distances, k=k, dim=1, largest=False)
    node_indices = torch.repeat_interleave(torch.arange(0, topk.indices.size(-2), device=distances.device), topk.indices.size(-1)).unsqueeze(-2)
    per_node_indices = topk.indices.view(1, -1)
    return  torch.cat([node_indices, per_node_indices], dim=-2)

def compute_AA_orientation(atom_coord_tensor):
    # see third bullet point in the discussion of node features here-
    # https://arxiv.org/pdf/2009.01411.pdf (GVP paper)
    n = atom_coord_tensor[:, 1] - atom_coord_tensor[:, 2] # N-CA
    c = atom_coord_tensor[:, 3] - atom_coord_tensor[:, 2]
    
    nxc = torch.cross(n, c, dim=-1)
    nxc *= 0.577/torch.norm(nxc, dim=-1, keepdim=True) # coef is sqrt(1/3)
    
    second_term = ((n+c)*0.816)/torch.norm(n+c, dim=-1, keepdim=True) # coef is sqrt(2/3)
    
    return nxc - second_term

def _positional_embeddings(edge_index, num_embeddings=None, period_range=[2, 1000], device='cpu'):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def embed_sequence(chain, vocab, device):
    return torch.Tensor([vocab[residue.getSequence()[0]] for residue in chain.iterResidues()]).to(device)

def get_distances(atom_coord_tensor):
    # euchlidean distance, diagonal will have large value since that is usually important 
    almost_distances = atom_coord_tensor[:, 2].unsqueeze(0) - atom_coord_tensor[:, 2].unsqueeze(1) # distance between alpha carbons CA
    distances = torch.sqrt((almost_distances**2).sum(-1))
    distances += distances.max()*torch.eye(distances.size(0), device=atom_coord_tensor.device)
    return distances


def parse_pdb(pdb_path, device='cpu', inject_noise=False):
    chain = pr.parsePDB(pdb_path, chain='A', model=1)
    # dim is (n_residues, 7, 3)
    atom_coord_tensor = _get_atom_coord_tensor(chain, device)[1:-1].float() # residue atoms are C-1, N, CA, C, N+1, CA+1, CA-1
    if inject_noise: # this is used for de-noising training
        atom_coord_tensor += (atom_coord_tensor.max() - atom_coord_tensor.min()) * torch.rand_like(atom_coord_tensor) / 3
    # == construct node feature set ==
    # get dihedral angles (TODO - raise these to the circle, currently just psi, phi, omega)
    dihedrals = get_dihedral_tensor(atom_coord_tensor)
    
    # get idx of AA based on vocab 
    idx_tensor = embed_sequence(chain, vocab=AA_identity_vocab, device=device)[1:-1]
        
    # forward and backward unit vectors in the direction of CA -> CA+1, and CA -> CA-1 respectively 
    forward = atom_coord_tensor[:, -2] - atom_coord_tensor[:, 2]
    backward = atom_coord_tensor[:, 2] - atom_coord_tensor[:, -1]
    forward /= torch.norm(forward, dim=-1, keepdim=True)
    backward /= torch.norm(backward, dim=-1, keepdim=True)
    
    AA_orientation_vec = compute_AA_orientation(atom_coord_tensor)
    # dihedrals - first 6 elemeents - scalar features
    # AA_identity_embeddings - subsequent 16 elemeents - scalar features
    # forward/backward - subsequent 6 elements, vector features
    # AA_orientation_vec - subsequent 3 elements, vector features
    node_feats = torch.cat([idx_tensor.unsqueeze(-1), dihedrals, forward, backward, AA_orientation_vec], dim=-1)
    
    # == construct edge feature set ==
    distances = get_distances(atom_coord_tensor)
    # neighbors are defined by the nearest k=30 nodes
    edge_idx = get_knn_indices(distances, k=30)
    rbf_distances = _rbf(distances[edge_idx[0], edge_idx[1]], D_min=distances.min(), D_max=distances.max())
    
    # get unit vectors in the direction of a parent node alpha carbon to a child node alpha carbon Cai -> Caj
    directional_vectors = atom_coord_tensor[edge_idx[0], 3] - atom_coord_tensor[edge_idx[1], 3] # recall: idx=3 is alpha carbons
    directional_vectors /= torch.norm(directional_vectors, dim=-1, keepdim=True)
    
    pos_embedding = _positional_embeddings(edge_idx, num_embeddings=4, device=device)
    # rbf_distances - first 16 elemeents - scalar features
    # pos_embedding, subsequent "num_embeddings" elements, scalar features
    # directional_vectors - subsequent 3 elemeents - vector features
    edge_feats = torch.cat([rbf_distances, pos_embedding, directional_vectors], dim=-1)
    
    return node_feats.float(), edge_feats.float(), edge_idx, atom_coord_tensor


if __name__ == '__main__':
    from functools import partial
    from glob import glob
    from tqdm import tqdm
    
    paths = glob("/data/home/will/drugs-vae/data/big_files/PDB_dump/*.pdb")[:1000]
    
    func = partial(parse_pdb, inject_noise=False, device='cpu')
#     all_data = pool.map(func, paths)
    all_data = []
    for path in tqdm(paths, total=len(paths)):
        all_data.append(parse_pdb(path, inject_noise=False, device='cpu'))    
    
    batch_size = 500
    batches = [all_data[i : i + batch_size] for i in range(0, len(all_data), batch_size)]

    for i, batch in enumerate(batches):
        with open(f'data/tensors-{i}.pkl', 'wb') as f:
            pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)    
    
    
    
    
    
    
    
#     import matplotlib.pyplot as plt
#     chain = pr.parsePDB('hA3G_alphafold2_zinc_MSA.pdb', chain='A', model=1)
#     protein_residue_coords = _get_atom_coord_tensor(chain)
#     distances = get_distances(protein_residue_coords)

#     plt.imshow(distances.numpy(), interpolation='nearest')
#     plt.savefig('distogram_test')

#     embedding_rule = torch.nn.Embedding(len(AA_identity_vocab), embedding_dim=16, padding_idx=0)
#     node_feats, edge_feats, edge_idx, coords = parse_pdb('tests/hA3G_alphafold2_zinc_MSA.pdb', embedding_rule)
#     print('node_feats shape:', node_feats.shape)
#     print('node_feats sample:', node_feats[:10, :10])