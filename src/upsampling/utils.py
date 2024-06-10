import torch


def calc_strands_similarity(patch_world_displ):
    
    patch_size = patch_world_displ.shape[0]
    
    def cosine_similarity(a, b):
        dot_product = torch.sum(a * b, dim=-1)
        norm_a = torch.norm(a, dim=-1)
        norm_b = torch.norm(b, dim=-1)
        return dot_product / (norm_a * norm_b + 1e-20)


    diff_patch_world_displ_shifted_x = torch.roll(patch_world_displ, shifts=-1, dims=1)
    diff_patch_world_displ_shifted_y = torch.roll(patch_world_displ, shifts=-1, dims=0)

    # Compute the cosine similarities
    cos_sim_x = cosine_similarity(patch_world_displ[:, :-1], diff_patch_world_displ_shifted_x[:, :-1]).mean(-1)
    cos_sim_y = cosine_similarity(patch_world_displ[:-1, :], diff_patch_world_displ_shifted_y[:-1, :]).mean(-1)

    cos_sim = torch.zeros(patch_size, patch_size, device='cuda')

    cos_sim[-1, :] = cos_sim_y[-1, :]
    cos_sim[:, -1] = cos_sim_x[:, -1]

    cos_sim[:-1, :-1] = torch.maximum(cos_sim_x[:patch_size-1, :patch_size-1].reshape(-1, 1), cos_sim_y[:patch_size-1, :patch_size-1].reshape(-1, 1)).reshape(patch_size-1, patch_size-1)
    
    
    cos_sim[:-1, :-1] = (cos_sim_x[:patch_size-1, :patch_size-1] + cos_sim_y[:patch_size-1, :patch_size-1]) / 2
    
    return cos_sim