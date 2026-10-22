# Late Fusion

Late fusion is a multimodal learning approach in which separate models or encoders process different data modalities independently, and their predictions or learned representations are combined at a later stage.

In clinical research, for example, separate models may process structured EHR data (e.g., laboratory results and demographics), clinical notes, and medical images. Their outputs are then combined using a fusion method, such as averaging predicted probabilities, a weighted sum, or a trainable meta-model, to produce a final prediction.

Late fusion allows each modality to be processed with a model suited to its characteristics and can facilitate the integration of heterogeneous data sources. However, it may capture fewer cross-modal interactions than approaches that combine information earlier in the modeling pipeline.

## Standard Generic Workflow

flowchart TD
    A[Clinical Data Sources] --> B1[Structured EHR Data]
    A --> B2[Clinical Notes]
    A --> B3[Medical Images]

    B1 --> C1[Tabular Encoder]
    B2 --> C2[Text Encoder]
    B3 --> C3[Vision Encoder]

    C1 --> D1[Projection Head]
    C2 --> D2[Projection Head]
    C3 --> D3[Projection Head]

    D1 --> E[Multimodal Representations]
    D2 --> E
    D3 --> E

    E --> F[Fusion / Integration]
    F --> G[Pretrained Foundation Model]
    G --> H[Task-Specific Output Head]
    H --> I[Clinical Prediction / Task Output]

    classDef data fill:#E8F1FA,stroke:#3973AC,color:#111
    classDef encoder fill:#E8F5E9,stroke:#4A8C53,color:#111
    classDef projection fill:#FFF3D6,stroke:#C58A15,color:#111
    classDef model fill:#F0E5FA,stroke:#8052A0,color:#111
    classDef output fill:#FCE8E6,stroke:#C45B50,color:#111

    class B1,B2,B3 data
    class C1,C2,C3 encoder
    class D1,D2,D3 projection
    class F,G model
    class H,I output

