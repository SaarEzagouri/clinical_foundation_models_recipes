The current dir corrsponds to the diagram's *Foundation-Model-Specific Projector*.

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

    style F fill:#fff2cc,stroke:#ff9900,stroke-width:4px,color:#000000
