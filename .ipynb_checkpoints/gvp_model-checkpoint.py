import torch
from torch import nn

from preprocess_protien import AA_identity_vocab


def exists(thing): return thing is not None

# get/post because that is the terminology for get/opposite of get in an API
# it made sense at the time of writing
# EDIT 3 days later - it does not make sense and is stupid but too lazy to change
def get_scalar_vec_feats(node_feats, n_vecs=2):
    return node_feats[..., :-3*n_vecs], node_feats[..., -3*n_vecs:].view(-1, n_vecs, 3)
def post_scalar_vec_feats(s, v, n_vecs=2):
    return torch.cat([s, v.view(-1, n_vecs*3)], -1)

# there is probably a better way to do this using torch.gather() or something, but this is guaranteed to execute well on cuda
class MessagePassing(torch.nn.Module):  
    def __init__(self):
        super().__init__()    
        
    def _sum_over_neighbors(self, edge_attrs, attr_idx):
        '''
        inputs - edge_attrs, attr_idx
        edge_attrs is the edge attributes, shape is (number of edges, number of edge features)
        attr_idx is the edge information, first column is parent node, second column is child node, shape is (number of edges, number of edge features)

        order is kept specific in both tensors - they should map 1:1 edge identification in attr_idx and edge features in edge_attrs
        '''
        n_edge_feats = edge_attrs.size(-1)
        n_edges = edge_attrs.size(0)
        
        # edge attributes in form that floats the boat of torch.sparse_coo_tensor
        flat_edges = edge_attrs.contiguous().view(-1, 1)

        # shape of these idx tensors will be n_edge_feats*n_edges or attr_idx.size(0)*attr_idx.size(1)
        attr_idx = attr_idx.long().repeat(1, n_edge_feats) # repeat tensor of edge idx to expand computation to each feature
        attr_idx2 = torch.arange(0, n_edge_feats, dtype=torch.long, device=edge_attrs.device).repeat_interleave(n_edges).unsqueeze(0) 
        indices = torch.cat([attr_idx, attr_idx2], dim=0)
        
        adjacency_attrs = torch.sparse_coo_tensor(indices, flat_edges.squeeze(), (indices.max()+1, indices.max()+1, n_edge_feats))

        # analog of adjacency matrix, except each entry is edge information
        # aggregate across n_nodes dimension to aggregate across neighbors
        # ret.shape = (n_nodes, edge_attr_dim)
        return torch.sparse.sum(adjacency_attrs, -2).to_dense()
    
def GVP_layernorm(V):
    # V.shape = (n_nodes, v, 3)
    # scale the row vectors of V such that their root-mean-square norm is one
    assert torch.norm(V, keepdim=True)/V.size(-2) > 0, 'GVP layernorm encountered negative value or zero in denominator'
    return V/torch.sqrt(torch.norm(V, keepdim=True)/V.size(-2)) #.size(-2) is v    

class Layernorm_combined(torch.nn.Module):
    def __init__(self):
        super().__init__(x_out_scalar_channels)
        self.layernorm = nn.LayerNorm(x_out_scalar_channels)
    
    def forward(self, s, V):
        assert s.max() < 1e21, 'your values being passed to layernorm are too large, they will return nan'
        return self.layernorm(s), GVP_layernorm(V)
        

class GVP(nn.Module):
    '''    
    Bowen Jing et. al. (2021)
    Learning from Protein Structure with Geometric Vector Perceptrons
    https://arxiv.org/pdf/2009.01411.pdf
    '''
    def __init__(self, v_dim, μ_dim, n_dim, m_dim, h_dim):
        super().__init__()
        self.v_dim = v_dim
        self.μ_dim = μ_dim
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.h_dim = h_dim
        
        # usually, node_feats_dim == h_dim, for all but first layer 
        if not exists(h_dim):
            h_dim = v_dim
        
        # for vector transformations
        self.Wh = nn.Parameter(torch.normal(mean=0, std=0.25, size=(h_dim, v_dim)))
        self.Wμ = nn.Parameter(torch.normal(mean=0, std=0.25, size=(μ_dim, h_dim)))
        
        # for scalar transformations
        self.Wm = nn.Sequential(
            nn.Linear(h_dim+n_dim, m_dim),
            nn.ReLU(),
        )
        
        self.σplus = nn.ReLU()

    def forward(self, x, split_output=False):
        s, V = get_scalar_vec_feats(x, n_vecs = self.v_dim)
        Vh = self.Wh @ V
        Vμ = self.Wμ @ Vh
        
        sh = torch.norm(Vh, dim=-1) # V=vector, this is norm along euchlidean vector channel (x,y,z) -> L
        vμ = torch.norm(Vμ, dim=-1, keepdim=True)
        
        shn = torch.cat([s, sh], dim=-1)
        
        s_ = self.Wm(shn)
        
        V_ = self.σplus(vμ) * Vμ
        
        if split_output:
            return s_, V_
        
        return post_scalar_vec_feats(s_, V_, n_vecs = self.μ_dim)
    
    
class GVP_MPNN(MessagePassing):
    '''    
    Message passing layer as defined in the GVP paper;
    
    Bowen Jing et. al. (2021)
    Learning from Protein Structure with Geometric Vector Perceptrons
    https://arxiv.org/pdf/2009.01411.pdf
    '''
    def __init__(
            self, # no, there is not a better way to do this, probably
            x_out_vector_channels,
            x_out_scalar_channels,
            edge_out_vector_channels,
            edge_out_scalar_channels,
            hidden_dim, 
            dropout_p=0.1,
            residual=True):
        super().__init__()
        
        self.ve_dim = edge_out_vector_channels
        self.μe_dim = edge_out_vector_channels
        ne_dim = edge_out_scalar_channels
        me_dim = edge_out_scalar_channels
        vx_dim = x_out_vector_channels
        self.μx_dim = x_out_vector_channels
        nx_dim = x_out_scalar_channels
        mx_dim = x_out_scalar_channels
        h_dim = hidden_dim
        self.residual = residual
        
        self.g_v = GVP(v_dim=vx_dim, n_dim=nx_dim, μ_dim=self.μx_dim, h_dim=h_dim, m_dim=mx_dim) # for nodes
        self.g_ve = GVP(v_dim=self.μx_dim+self.ve_dim, n_dim=mx_dim+ne_dim, μ_dim=self.μx_dim, h_dim=h_dim, m_dim=mx_dim) # for nodes and edges
        self.g_e = GVP(v_dim=self.μx_dim+self.ve_dim, n_dim=mx_dim+ne_dim, μ_dim=self.μe_dim, h_dim=h_dim, m_dim=me_dim) # for nodes and edges
        
        self.ln = nn.LayerNorm(48)
        self.d = nn.Dropout(dropout_p)
        self.d2 = nn.Dropout2d(dropout_p)
        
    def norm_and_dropout(self, s, V): # you can only combine them when your model is NOT residual
        return self.ln(self.d(s)), GVP_layernorm(self.d2(V))
    
    def forward(self, node_attrs, edge_attrs, edge_idx):
        
        # == equation 5 in the paper, except not residual ==
        # perform the node update first - project node features to hidden dimension 
        s_node, V_node = self.g_v(node_attrs, split_output=True)
        
        if self.residual:
            s_node_before, V_node_before = get_scalar_vec_feats(node_attrs, n_vecs=self.g_v.μ_dim)
            s_node += s_node_before
            V_node += V_node_before
        s_node, V_node = self.norm_and_dropout(s_node, V_node)
        node_attrs = post_scalar_vec_feats(s_node, V_node, n_vecs = self.g_v.μ_dim)
        
        # == equation 3 in the paper ===
        # get scalar and vector features for all edges
        s_node, V_node = get_scalar_vec_feats(node_attrs[edge_idx[1]], n_vecs=self.g_v.μ_dim) # edge_idx[1] refers to the SOURCE nodes
        s_edge, V_edge = get_scalar_vec_feats(edge_attrs, n_vecs=self.ve_dim) # 1 comes from data preprocessing
        
        # concatenate - information required for edge update is all edge and all node scalar+vector features
        s = torch.cat([s_node, s_edge], dim=-1)
        V = torch.cat([V_node, V_edge], dim=-2)
        attr_and_edge_tensor = post_scalar_vec_feats(s, V, n_vecs=self.μx_dim+self.μe_dim)
        
        # get updated edge representation
        hij = self.g_ve(attr_and_edge_tensor)
        
        # == equation 4 in the paper ==
        node_update = self._sum_over_neighbors(hij, edge_idx) / 30 # 30 here is because we manually enforce 30 edges per node
        
        s_node, V_node = get_scalar_vec_feats(node_update, n_vecs=self.g_ve.μ_dim)
        if self.residual:
            s_node_before, V_node_before = get_scalar_vec_feats(node_attrs, n_vecs=self.g_v.μ_dim)
            s_node += s_node_before
            V_node += V_node_before
        s_node, V_node = self.norm_and_dropout(s_node, V_node)
        node_update = post_scalar_vec_feats(s_node, V_node, n_vecs=self.g_ve.μ_dim)
        
        # == not done in paper ==
        # to aggregate and add the messages to the nodes, they need to be the same shape, and thus the edge/node information
        # ends up being the same shape. here, we pass the edges through another GVP sequence to get them into different shapes
        hij = self.g_e(attr_and_edge_tensor)
        
        return node_attrs + node_update, hij 
        
class GVP_GNN(MessagePassing):
    '''    
    Message passing layer as defined in the GVP paper;
    
    Bowen Jing et. al. (2021)
    Learning from Protein Structure with Geometric Vector Perceptrons
    https://arxiv.org/pdf/2009.01411.pdf
    '''
    def __init__(
            self, # no, there is not a better way to do this, probably
            x_out_vector_channels,
            x_out_scalar_channels,
            edge_out_vector_channels,
            edge_out_scalar_channels,
            hidden_dim, 
            edge_in_scalar_channels=22,
            edge_in_vector_channels=1,
            x_in_vector_channels=3,
            x_in_scalar_channels=22,
            dropout_p=0.1,
            n_layers=3,
        ):
        super().__init__()
        self.gcnn_layers = torch.nn.ModuleList()
        
        # == these layers will be used to project nodes/edges to the dimensions usable by the MPNN layers ==
        self.g_v = GVP( # for nodes
            v_dim=x_in_vector_channels, 
            n_dim=x_in_scalar_channels,
            μ_dim=x_out_vector_channels,
            m_dim=x_out_scalar_channels,
            h_dim=hidden_dim,
        )
        self.g_ve = GVP( # for nodes and edges
            v_dim=edge_in_vector_channels,
            n_dim=edge_in_scalar_channels, 
            μ_dim=edge_out_vector_channels,
            h_dim=hidden_dim,
            m_dim=edge_out_scalar_channels
        )
        
        self.layernorm = nn.LayerNorm(x_out_scalar_channels)
        self.d = nn.Dropout(dropout_p)
        
        self.embed = torch.nn.Embedding(len(AA_identity_vocab), embedding_dim=16, padding_idx=0)
        
        for _ in range(n_layers):
            self.gcnn_layers.append(GVP_MPNN(
                x_out_vector_channels=x_out_vector_channels,
                x_out_scalar_channels=x_out_scalar_channels,
                edge_out_vector_channels=edge_out_vector_channels,
                edge_out_scalar_channels=edge_out_scalar_channels,
                hidden_dim=hidden_dim,
            ))
        
    def forward(self, node_attrs, edge_attrs, edge_idx, coords):
        # this just embeds amino acids with the embedding defined in __init__ :(
        node_attrs = self.embed_node_features(node_attrs)
        
        # project node and edge attributes to the dimensions used by the rest of the model
        node_attrs = self.g_v(node_attrs, split_output=False)        
        edge_attrs = self.g_ve(edge_attrs)
        
        for layer in self.gcnn_layers:
            node_attrs, edge_attrs = layer(node_attrs, edge_attrs, edge_idx)
        # TODO - maybe edges still contain information at final layer? latent = torch.cat([node_attrs.mean(-2), edge_attrs.mean(-2)], dim=-1)
        latent = node_attrs.mean(-2)
        return node_attrs, latent
    
    def embed_node_features(self, node_attrs):
        # i accidently wrote this in such a way that it only works if idx_col = 0
        idx_col = 0
        assert (node_attrs[:, idx_col].int().float() == node_attrs[:, idx_col]).all(), f'you have probably the column which holds your amino acid identity indices at the incorrect location, idx tensor looks like this - {node_attrs[:, idx_col]}'
        AA_embeddings = self.embed(node_attrs[:, idx_col].long())
        return torch.cat([AA_embeddings, node_attrs[:, 1:]], dim=-1)

if __name__ == '__main__':
    test = GVP_GNN(
        x_in_vector_channels=3,
        x_out_vector_channels=4,
        x_in_scalar_channels=22,
        x_out_scalar_channels=48,
        edge_in_vector_channels=1,
        edge_out_vector_channels=2,
        edge_in_scalar_channels=20,
        edge_out_scalar_channels=16,
        hidden_dim=16,
        n_layers=3
    )
    from preprocess_protein import parse_pdb
    embedding_rule = torch.nn.Embedding(len(AA_identity_vocab), embedding_dim=16, padding_idx=0)
    node_feats, edge_feats, edge_idx, coords = parse_pdb('tests/hA3G_alphafold2_zinc_MSA.pdb',embedding_rule)
    print('node_feats out shape:', node_feats.shape)
    print('node_feats out sample:', node_feats[:10, :10])