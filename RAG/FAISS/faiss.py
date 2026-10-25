from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

class FaissMemoryBank:
    """
    Standard reproducible component for building and querying a FAISS-supported 
    population memory bank for Retrieval-Augmented Multimodal Inference.
    
    References:
        - Original Paper (Meta AI): https://arxiv.org/abs/2401.08281
        - Original Repository: https://github.com/facebookresearch/faiss
        
    Implementation & Use Case:
        In multimodal clinical prediction pipelines (such as PRISM-MedGemma), raw patient 
        records are encoded into dense representations by modality-specific towers (e.g., EHR 
        or ECG encoders). To help the model infer clinical trajectories from historical precedent, 
        this component constructs an index over a training population memory bank using FAISS 
        (Facebook AI Similarity Search). 
        
        During inference or dataset preprocessing, a patient's embedding queries the FAISS 
        index to perform Approximate Nearest Neighbor (ANN) search. The top-k similar patient 
        embeddings are retrieved and subsequently transformed via soft token projectors to inject 
        historical population context directly into the language model's input sequence.
    """

    def __init__(self, dimension: int, index_type: str = "Flat", use_gpu: bool = False):
        if not _FAISS_AVAILABLE:
            raise ImportError("FAISS library is not installed. Please install via `pip install faiss-cpu` or `faiss-gpu`.")
        
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index: Optional[faiss.Index] = None
        
    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Builds and trains the FAISS index using the provided cohort embeddings.

        Args:
            embeddings (np.ndarray): A 2D numpy array of shape (num_samples, dimension) 
                                     containing the population features.
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"

        if self.index_type == "Flat":
            # Exact search using L2 (Euclidean) distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Quantized inverted file index for faster search on large cohorts
            nlist = min(4096, max(1, embeddings.shape[0] // 39))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            if not self.index.is_trained:
                self.index.train(embeddings)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the population memory bank for the top-k nearest neighbors.

        Args:
            query_embeddings (np.ndarray): A 2D numpy array of shape (num_queries, dimension).
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - distances: Array of shape (num_queries, top_k) representing vector distances.
                - indices: Array of shape (num_queries, top_k) containing dataset row indices of matches.
        """
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded yet. Call `build_index` first.")
            
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save(self, filepath: str) -> None:
        """Saves the FAISS index to disk."""
        if self.index is not None:
            cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
            faiss.write_index(cpu_index, filepath)

    def load(self, filepath: str) -> None:
        """Loads a pre-computed FAISS index from disk."""
        self.index = faiss.read_index(filepath)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)