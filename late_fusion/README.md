# Token-Level Fusion

Token-level fusion is a multimodal learning approach in which separate encoders process different data modalities, and their representations are projected into a shared foundation model's expected embedding space. The resulting modality-specific tokens are concatenated into a shared token sequence and processed jointly by the foundation model, which can learn interactions between modalities through self-attention.

In clinical research, for example, a tabular encoder may process structured EHR data (e.g., laboratory results and demographics), a text encoder may process clinical notes, and a vision encoder may process medical images. Each encoder's output is first transformed by a generic projector into an intermediate representation. A foundation-model-specific projector then maps these representations into the embedding space expected by the pretrained foundation model. The resulting tokens are concatenated, optionally alongside special tokens or textual prompts, and passed to the foundation model for joint processing.

Token-level fusion enables the model to learn fine-grained cross-modal interactions, allowing information from one modality to influence how it interprets another. However, it requires compatible token representations and can increase computational and memory requirements, particularly when processing high-resolution images or long clinical notes.

## Multimodal Foundation Model Workflow

```mermaid
flowchart TD
    A[Clinical Data Sources] --> B1[Structured EHR Data]
    A --> B2[Clinical Notes]
    A --> B3[Medical Images]

    B1 --> C1[Tabular Encoder]
    B2 --> C2[Text Encoder]
    B3 --> C3[Vision Encoder]

    C1 --> D1[Generic Projector]
    C2 --> D2[Generic Projector]
    C3 --> D3[Generic Projector]

    D1 --> E1[Tabular Representations]
    D2 --> E2[Text Representations]
    D3 --> E3[Image Representations]

    E1 --> F[Foundation-Model-Specific Projector]
    E2 --> F
    E3 --> F

    F --> G[Foundation-Compatible Tokens]
    G --> H[Token Concatenation]
    H --> I[Shared Token Sequence]
    I --> J[Pretrained Foundation Model]
    J --> K[Task-Specific Output Head]
    K --> L[Clinical Prediction]
