from __future__ import annotations
import torch
import torch.nn as nn

class SoftTokenProjector(nn.Module):
    """
    A generic, reusable soft token projector that maps any input representation 
    tensor of shape (Batch, d_in) into a sequence of soft tokens of shape 
    (Batch, n_tokens, d_llm) suitable for LLM prefix or token-level fusion.

    Corresponding to the main README diagram: Foundation Model-Specific Projector.
    """

    def __init__(
        self, 
        d_in: int,                 # Input feature dimension from your upstream encoder (e.g., EHRTower output dim)[cite: 1]
        d_llm: int = 2560,         # Target LLM hidden dimension (default configured for MedGemma specs)[cite: 1]
        n_tokens: int = 4,         # Number of soft token slots to generate per sample[cite: 1]
        hidden_dim: int = 1024,    # Internal bottleneck expansion dimension for the MLP[cite: 1]
        dropout: float = 0.1       # Dropout probability for regularization
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_llm = d_llm
        self.n_tokens = n_tokens
        
        # Define the MLP non-linear mapping head
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),                    # Normalizes input representation variance
            nn.Linear(d_in, hidden_dim),           # First linear mapping to hidden expansion space
            nn.GELU(),                             # Smooth non-linear activation function
            nn.Dropout(dropout),                   # Regularization layer
            nn.Linear(hidden_dim, n_tokens * d_llm) # Projects flat features directly into the total size required for reshaping
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass to project and reshape input representations.
        
        Args:
            x: A 2D PyTorch tensor of shape (Batch_Size, d_in)[cite: 1].
            
        Returns:
            A 3D PyTorch tensor of shape (Batch_Size, n_tokens, d_llm) ready for LLM injection[cite: 1].
        """
        batch_size = x.size(0)  # Extract dynamic batch size
        
        # 1. Project through the MLP layers -> Output shape: (Batch_Size, n_tokens * d_llm)
        projected = self.net(x)
        
        # 2. Reshape into the 3D multi-token sequence format expected by the LLM
        return projected.view(batch_size, self.n_tokens, self.d_llm)


def build_soft_token_projector(
    d_in: int, 
    d_llm: int = 2560, 
    n_tokens: int = 4, 
    hidden_dim: int = 1024
) -> SoftTokenProjector:
    """
    Factory function to instantiate and return the SoftTokenProjector.
    Matches the clean builder pattern seen in modular codebases[cite: 1].
    """
    return SoftTokenProjector(
        d_in=d_in, 
        d_llm=d_llm, 
        n_tokens=n_tokens, 
        hidden_dim=hidden_dim
    )