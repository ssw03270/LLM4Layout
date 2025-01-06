import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
from retriever_dataloader import RetrieverDataset
import faiss
from accelerate.utils import set_seed


def build_faiss_index(latent_vectors, use_gpu=False):
    d = latent_vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
    faiss.normalize_L2(latent_vectors)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(latent_vectors)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def perform_retrieval(index, query_vectors, query_file_paths, train_file_paths, k=5, is_train=False):
    retrieval_dict = {}

    # Normalize query vectors for cosine similarity
    faiss.normalize_L2(query_vectors)

    D, I = index.search(query_vectors, k + 1 if is_train else k)  # k+1 for train to exclude self

    for idx, (distances, indices) in enumerate(zip(D, I)):
        query_path = query_file_paths[idx]
        retrieved_paths = []

        for neighbor_rank, neighbor_idx in enumerate(indices):
            if is_train and neighbor_idx == idx:
                continue  # Skip self for train retrieval
            retrieved_paths.append(train_file_paths[neighbor_idx])
            if len(retrieved_paths) == k:
                break

        retrieval_dict[query_path] = retrieved_paths

    return retrieval_dict

def encode_dataset(dataset, dataloader, model, device):
    latent_vectors = []
    file_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {dataset} images"):
            pkl_files, image_masks = batch
            image_masks = image_masks.to(device)

            features = model(image_masks)
            features = features.view(features.size(0), -1)
            features_np = features.cpu().numpy()
            latent_vectors.append(features_np)
            file_paths.extend(pkl_files)

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    print(f'Extracted latent vectors for {dataset} set shape: {latent_vectors.shape}')
    print(f"File paths length for {dataset} set: {len(file_paths)}")
    return latent_vectors, file_paths


def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument("--root_dir", type=str, default="../../datasets/threed_front_diningroom",
                        help="Dataset root directory")
    args = parser.parse_args()

    set_seed(42)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and modify ResNet-18
    resnet18 = models.resnet18(pretrained=True)
    modules = list(resnet18.children())[:-1]  # Remove the last FC layer
    model = torch.nn.Sequential(*modules)
    model = model.to(device)
    model.eval()

    # Initialize DataLoaders for train, val, and test
    splits = ['train', 'val', 'test']
    batch_size = 1024
    num_workers = 4
    dataloaders = {}
    file_paths_dict = {}
    latent_vectors_dict = {}

    for split in splits:
        dataset = RetrieverDataset(root_dir=args.root_dir, data_type=split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        dataloaders[split] = dataloader

    # Encode the train set and build the FAISS index
    train_latent, train_file_paths = encode_dataset('train', dataloaders['train'], model, device)
    index = build_faiss_index(train_latent, use_gpu=True)  # Assuming GPU is available

    # Optionally, save the train FAISS index
    index_path = 'faiss_index_train.bin'
    faiss.write_index(faiss.index_gpu_to_cpu(index) if faiss.StandardGpuResources else index, index_path)
    print(f"FAISS index for train set saved to {index_path}")

    # Perform retrieval for each split
    retrieval_results = {}
    for split in splits:
        latent, file_paths = encode_dataset(split, dataloaders[split], model, device)
        is_train = split == 'train'
        retrieval = perform_retrieval(index, latent, file_paths, train_file_paths, k=5, is_train=is_train)
        retrieval_results[split] = retrieval

        # Save retrieval results for each split
        dict_save_path = f'retrieval_{split}.pkl'
        with open(dict_save_path, 'wb') as f:
            pickle.dump(retrieval, f)
        print(f"Retrieval results for {split} set saved to {dict_save_path}")

    # Display a sample from each retrieval dictionary
    for split in splits:
        sample_key = list(retrieval_results[split].keys())[0]
        sample_retrieved = retrieval_results[split][sample_key]
        print(f"\nSample retrieval for {split} set - {sample_key}:")
        for rank, path in enumerate(sample_retrieved, start=1):
            print(f"  Rank {rank}: {path}")


if __name__ == "__main__":
    main()
