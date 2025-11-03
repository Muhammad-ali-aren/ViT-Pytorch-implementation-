import torch 
import torch.nn as nn
import math

def patcher(image,image_size=384):
    if image.shape[1] != image_size:
        raise ValueError(" size of image isn't 384 x 384")
    patch_size = int(image_size / math.sqrt(16))
    patches = []
    for patchx in range(0,image.shape[1], patch_size):
        for patchy in range(0,image.shape[1], patch_size):
            patch = image[:,patchx: patchx + patch_size, patchy: patchy + patch_size]
            patches.append(patch)
    return patches


def PatchEmbeddings(patches,number_of_patches=16,out_features=768):
    batch_size = len(patches)
    patch_dim = len(torch.flatten(patches[0][0]))
    class_token = nn.Parameter(torch.zeros(batch_size,1,out_features))
    linear_embedding = nn.Linear(in_features=patch_dim, out_features=out_features)
    postional_embedding = nn.Parameter(torch.zeros(batch_size,number_of_patches + 1, out_features))
    batch_embeddings = torch.tensor([])
    for batch_idx in range(batch_size):
        image_embeddings = []
        for patch in patches[batch_idx]:
            flattened_patch = torch.flatten(patch)
            embedded_patch = linear_embedding(flattened_patch)
            image_embeddings.append(embedded_patch)
        image_embeddings = torch.stack(image_embeddings).unsqueeze(0)
        batch_embeddings = torch.cat((batch_embeddings,image_embeddings),dim=0)
    l_embeddings_With_Class_tokken = torch.cat((class_token,batch_embeddings),dim=1)
    embedded_patches = torch.add(l_embeddings_With_Class_tokken, postional_embedding)
    return embedded_patches