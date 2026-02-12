# 🏗️ YOLO11x-RGBT-Gated-V3 System Architecture

**Documentation Date**: February 11, 2026
**System Status**: Production Training (Phase 1)
**Model Type**: Dual-Stream Mid-Fusion (RGBT)

---

## 1. High-Level Architecture

The system is a **Multi-Modal Mid-Fusion** architecture based on YOLO11x. It processes RGB and Thermal (IR) images in parallel using separate backbones, fuses the features at three scales (P3, P4, P5) using a custom **Gated Spatial Fusion** mechanism, and then decodes the fused features using a standard YOLO11 head.

### Core Components

1. **Dual Backbone**: Two identical CSPDarknet backbones (one for RGB, one for IR) extract features independently.
2. **Gated Fusion Bridge**: Three `GatedSpatialFusion_V3` modules selectively merge features based on uncertainty and illumination.
3. **Unified Head**: A standard YOLO11 PANet + Detect head processes the fused feature maps to predict bounding boxes and classes.

---

## 2. Visual Data Flow (Mermaid)

```mermaid
graph TD
    classDef rgb fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef ir fill:#fbe9e7,stroke:#bf360c,stroke-width:2px;
    classDef fuse fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef head fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    %% INPUTS
    Input[RGBT Input (4ch)] --> Split{Split}
    Split --> RGB_Img[RGB Image (3ch)]:::rgb
    Split --> IR_Img[IR Image (1ch)]:::ir

    %% BACKBONES
    subgraph "Dual Stream Backbone"
        direction TB
        RGB_Img --> RGB_P1
        RGB_P1 --> RGB_P2
        RGB_P2 --> RGB_P3[RGB P3 (256ch)]:::rgb
        RGB_P3 --> RGB_P4[RGB P4 (512ch)]:::rgb
        RGB_P4 --> RGB_P5[RGB P5 (1024ch)]:::rgb

        IR_Img --> IR_P1
        IR_P1 --> IR_P2
        IR_P2 --> IR_P3[IR P3 (256ch)]:::ir
        IR_P3 --> IR_P4[IR P4 (512ch)]:::ir
        IR_P4 --> IR_P5[IR P5 (1024ch)]:::ir
    end

    %% FUSION
    subgraph "Gated Fusion Bridge (P3, P4, P5)"
        direction TB
        %% P3 Fusion
        RGB_P3 & IR_P3 --> Fusion_P3[GatedFusion V3]:::fuse
        RGB_Img -.-> Fusion_P3

        %% P4 Fusion
        RGB_P4 & IR_P4 --> Fusion_P4[GatedFusion V3]:::fuse
        RGB_Img -.-> Fusion_P4

        %% P5 Fusion
        RGB_P5 & IR_P5 --> Fusion_P5[GatedFusion V3]:::fuse
        RGB_Img -.-> Fusion_P5
    end

    %% HEAD
    subgraph "YOLO11 Head"
        direction TB
        Fusion_P5 --> SPPF[SPPF]
        SPPF & Fusion_P4 --> PANet_P4
        PANet_P4 & Fusion_P3 --> PANet_P3
        
        PANet_P3 --> Detect_small
        PANet_P4 --> Detect_medium
        PANet_P5 --> Detect_large
    end
```

---

## 3. Component Deep Dive

### A. The Fusion Module: `GatedSpatialFusion_V3`

Located in: `ultralytics/nn/modules/block.py`

This is the brain of the system. It decides *how much* of each modality to use for every pixel.

**Key Mechanisms:**

1. **Stochastic Modality Dropout**: During training, randomly replaces 30% of RGB or IR features with **Learnable Tokens** (Magnitude 0.3). This forces the network to handle missing info.
2. **Uncertainty Quantification**: Uses **MC-Dropout (n=20)** to estimate pixel-wise uncertainty/variance. High variance = Low Confidence.
3. **Illumination Awareness**: A CNN estimator predicts scene brightness (0.0 - 1.0).
    * **Night**: Suppresses RGB weights based on darkness.
    * **Day**: Balances RGB and IR.
4. **Gating Logic**:

    ```python
    # Simplified Logic
    RGB_Conf = 1 / (1 + RGB_Uncertainty)
    IR_Conf = 1 / (1 + IR_Uncertainty)
    
    Gate_RGB = Network(Feats) * RGB_Conf * Illumination
    Gate_IR = Network(Feats) * IR_Conf * (1 - Illumination * Scale)
    
    Fused = (RGB * Gate_RGB) + (IR * Gate_IR)
    ```

### B. The Loss Function: `GatedDetectionLoss`

Located in: `gate_supervision.py`

Wrapper around standard YOLOv8 loss.

* **Primary Task**: Box Regression (CIoU) + Classification (BCE) + DFL.
* **Auxiliary Task**: **Gate Supervision** (Weight 0.05).
  * Forces the gate network to output weights consistent with illumination.
  * Example: If Illumination < 0.1 (Pitch Black), forces RGB Gate Weight -> 0.1.
  * **Safety**: Clears cached gate weights immediately to prevent `deepcopy` crashes.

### C. Training Pipeline: `MCFTrainer`

Located in: `train_near_model_gated.py`

* **Optimizer Hook**: Implements **Gradient Clipping** (max_norm=10.0) before every step to prevent exploding gradients in the gate network.
* **Callbacks**:
  * `on_train_start`: Registers gradient clipping.
  * `on_train_epoch_end`: Visualizes gate weights every 5 epochs using `visualize_gates.py`.

---

## 4. Source Code Mapping

| Component | Source File | Description |
| :--- | :--- | :--- |
| **Model Config** | `yolo11x-RGBT-gated-v3.yaml` | Defines dual-backbone and fusion connections. |
| **Fusion Block** | `ultralytics/nn/modules/block.py` | Implementation of `GatedSpatialFusion_V3`. |
| **Loss Function** | `gate_supervision.py` | `GatedDetectionLoss` wrapper. |
| **Trainer** | `train_near_model_gated.py` | Main training script & `MCFTrainer`. |
| **Visualization** | `visualize_gates.py` | Utility to plot gate weights over images. |
| **Verification** | `verify_gated_v3_final.py` | Test suite for stability and logic. |

---

## 5. Critical Engineering Decisions (Why it works)

1. **Mid-Fusion Strategy**: We fuse at the end of the backbone (P3/P4/P5) rather than the input. This allows the network to learn rich, modality-specific features (edges, textures in RGB; heat signatures in IR) *before* trying to combine them.
2. **Uncertainty-Weighted Gating**: We don't just "add" features. We trust the modality that is "confident" (low variance). If RGB is noisy (night), its uncertainty spikes, and its weight drops.
3. **Learnable Tokens**: Zeroing out missing modalities is bad for gradients. We use learnable tokens so the network has a canonical "I don't know" vector that still facilitates backpropagation.
