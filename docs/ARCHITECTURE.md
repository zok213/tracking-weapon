# 🎯 Vietnam Weapon Detection System - Complete System Architecture v7.0

## System Overview

This document provides the **complete, production-ready architecture** for the Vietnam Drone Weapon Detection System with all hardware components including the **Gimbal Hub**.

---

## 1. Complete Hardware Architecture

```mermaid
graph TB
    subgraph "Drone Platform"
        DRONE["🚁 DJI Mavic Mini 2<br/>or equivalent"]
    end

    subgraph "Sensor Suite"
        RGB["📷 RGB Camera<br/>1920×1080 @ 30fps"]
        THM["🌡️ Gremsy Thermal Camera<br/>640×512 @ 30fps"]
        LASER["📏 Gremsy Laser<br/>Distance Meter"]
        IMU["⚖️ IMU/GPS<br/>Position + Attitude"]
    end

    subgraph "Gimbal System"
        GIMBAL["🎯 Gremsy VIO F1<br/>3-Axis Gimbal"]
        HUB["🔌 Gimbal Hub<br/>Power + Data Multiplexer"]
        CTRL["🎮 Gimbal Controller<br/>MAVLink Protocol"]
    end

    subgraph "Processing Unit"
        JETSON["💻 Jetson Orin NX<br/>100 TOPS | 8GB RAM | 25W"]
    end

    subgraph "Communication"
        LTE["📡 LTE Modem<br/>Cloud Upload"]
        RC["📻 RC Link<br/>Operator Control"]
    end

    subgraph "Storage"
        SD["💾 SD Card<br/>Local Evidence"]
    end

    DRONE --> GIMBAL
    RGB --> HUB
    THM --> HUB
    LASER --> HUB
    HUB --> GIMBAL
    GIMBAL --> CTRL
    CTRL <--> JETSON
    IMU --> JETSON
    JETSON --> LTE
    JETSON --> SD
    RC --> JETSON

    style HUB fill:#ff9800,color:#000
    style GIMBAL fill:#4caf50,color:#fff
    style JETSON fill:#2196f3,color:#fff
```

---

## 2. Gimbal Hub Architecture (NEW)

The **Gimbal Hub** is the central data multiplexer connecting all sensors to the gimbal controller.

```mermaid
flowchart TB
    subgraph Sensors["Sensor Inputs"]
        RGB["📷 RGB Camera<br/>HDMI/USB3"]
        THM["🌡️ Thermal Camera<br/>USB/Serial"]
        LASER["📏 Laser Rangefinder<br/>Serial"]
    end

    subgraph Hub["Gimbal Hub (Gremsy Hub)"]
        PWR["⚡ Power Distribution<br/>12V/5V Rails"]
        MUX["🔀 Signal Multiplexer<br/>Data Routing"]
        SYNC["🔄 Frame Synchronizer<br/>Timestamp Alignment"]
        UART["📡 UART Bridge<br/>RS-232/TTL"]
    end

    subgraph Outputs["Hub Outputs"]
        GMBL["🎯 To Gimbal<br/>PWM Control"]
        JTSN["💻 To Jetson<br/>USB3 + Serial"]
        TELEM["📊 Telemetry<br/>MAVLink Stream"]
    end

    RGB --> MUX
    THM --> MUX
    LASER --> UART
    
    MUX --> SYNC
    UART --> SYNC
    PWR --> RGB & THM & LASER
    
    SYNC --> GMBL
    SYNC --> JTSN
    SYNC --> TELEM

    style Hub fill:#ff9800,color:#000
    style SYNC fill:#4caf50,color:#fff
```

### Gimbal Hub Functions

| Function | Description | Specification |
| -------- | ----------- | ------------- |
| **Power Distribution** | Provides regulated power to all sensors | 12V @ 3A, 5V @ 2A |
| **Signal Multiplexing** | Routes video/data streams | RGB HDMI, Thermal USB |
| **Frame Synchronization** | Aligns RGB + Thermal timestamps | ±1ms accuracy |
| **UART Bridge** | Connects laser rangefinder | 115200 baud |
| **Telemetry Output** | MAVLink stream to Jetson | 50 Hz update rate |

---

## 3. Complete Software Pipeline Architecture

```mermaid
graph TB
    subgraph "Layer 1: Hardware Interface"
        CAM_DRV["Camera Driver<br/>V4L2 / GStreamer"]
        THM_DRV["Thermal Driver<br/>USB Serial"]
        HUB_DRV["Hub Interface<br/>PySerial"]
        GMB_DRV["Gimbal MAVLink<br/>pymavlink"]
    end

    subgraph "Layer 2: Data Acquisition"
        SYNC["Frame Synchronizer<br/>Timestamp Matching"]
        PREPROC["Preprocessor<br/>Resize + Normalize"]
        TELEM["Telemetry Parser<br/>GPS + IMU + Gimbal"]
    end

    subgraph "Layer 3: AI Processing"
        YOLO["YOLO26n<br/>Detection"]
        CBAM["CBAM Fusion<br/>RGB + Thermal"]
        BYTE["ByteTrack<br/>Tracking"]
        LSTM["Thermal LSTM<br/>Verification"]
    end

    subgraph "Layer 4: Control & Output"
        GMB_CTRL["Gimbal Controller<br/>Center + Zoom"]
        GPS_TRANS["GPS Transformer<br/>Pixel → WGS84"]
        ALERT["Alert Generator<br/>Confidence → Priority"]
        EVIDENCE["Evidence Collector<br/>Video + JSON"]
    end

    subgraph "Layer 5: Communication"
        DASH["Operator Dashboard<br/>WebSocket UI"]
        CLOUD["Cloud Uploader<br/>LTE Async"]
        LOCAL["Local Storage<br/>SD Card"]
    end

    CAM_DRV --> SYNC
    THM_DRV --> SYNC
    HUB_DRV --> TELEM
    GMB_DRV --> TELEM

    SYNC --> PREPROC
    PREPROC --> YOLO
    YOLO --> CBAM
    CBAM --> BYTE
    BYTE --> LSTM

    LSTM --> GMB_CTRL
    LSTM --> GPS_TRANS
    LSTM --> ALERT
    
    TELEM --> GPS_TRANS
    GMB_CTRL --> GMB_DRV
    
    GPS_TRANS --> ALERT
    ALERT --> EVIDENCE
    EVIDENCE --> DASH & CLOUD & LOCAL

    style YOLO fill:#ff6b6b,color:#fff
    style BYTE fill:#4ecdc4,color:#fff
    style LSTM fill:#45b7d1,color:#fff
    style GMB_CTRL fill:#96ceb4,color:#000
```

---

## 4. Complete Data Flow Diagram

```mermaid
sequenceDiagram
    box Hardware Layer
        participant CAM as RGB Camera
        participant THM as Thermal Camera
        participant HUB as Gimbal Hub
        participant GMB as Gimbal
        participant GPS as Drone GPS/IMU
    end
    
    box Processing Layer
        participant JET as Jetson Orin NX
        participant DET as YOLOv12n
        participant TRK as ByteTrack
        participant VER as Thermal LSTM
    end
    
    box Output Layer
        participant ALT as Alert System
        participant OPR as Operator
    end

    Note over CAM,OPR: Frame Processing Loop (30 FPS = 33ms/frame)

    CAM->>HUB: RGB Frame (HDMI)
    THM->>HUB: Thermal Frame (USB)
    HUB->>HUB: Synchronize timestamps
    HUB->>JET: Synced frames + metadata
    GPS->>JET: Position + Attitude
    GMB->>JET: Gimbal angles (MAVLink)

    JET->>DET: Preprocess 640×640
    DET->>DET: RGB backbone + Thermal backbone
    DET->>DET: CBAM fusion
    DET->>TRK: Final Detections (NMS-free)


    TRK->>TRK: IoU matching (Stage 1 + 2)
    TRK->>VER: Tracked objects + IDs

    VER->>VER: Extract F1-F5 features
    VER->>VER: LSTM inference (30-frame buffer)
    VER->>ALT: Combined confidence

    alt High Confidence (≥0.65)
        ALT->>JET: Compute gimbal command
        JET->>GMB: Center + Zoom (MAVLink)
        JET->>JET: GPS transformation
        ALT->>OPR: Alert notification
        Note over ALT,OPR: Evidence package created
    end

    Note over CAM,OPR: Total: 18-25ms latency (YOLO26 optimized)
```

---

## 5. Gimbal Control with Hub Integration

```mermaid
flowchart TB
    subgraph Input["Detection System"]
        DET["Tracked Weapon<br/>bbox, confidence, class"]
    end

    subgraph Hub["Gimbal Hub"]
        HUB_RX["Receive:<br/>Gimbal telemetry"]
        HUB_TX["Transmit:<br/>Control commands"]
        HUB_SYNC["Sync:<br/>Sensor timestamps"]
    end

    subgraph Controller["Gimbal Controller"]
        CALC["Calculate:<br/>Centering error"]
        ZOOM["Decide:<br/>Zoom level"]
        CMD["Generate:<br/>MAVLink command"]
    end

    subgraph Gimbal["Gremsy VIO F1"]
        PITCH["Pitch Motor<br/>-90° nadir"]
        YAW["Yaw Motor<br/>0-360°"]
        CAM_MOUNT["Camera Mount<br/>RGB + Thermal"]
    end

    subgraph Feedback["Motion Compensation"]
        RATE["Angular Rate<br/>pitch_rate, yaw_rate"]
        SHIFT["Image Shift<br/>dx, dy pixels"]
        COMP["Compensate<br/>ByteTrack input"]
    end

    DET --> CALC
    CALC --> ZOOM
    ZOOM --> CMD
    CMD --> HUB_TX
    HUB_TX --> PITCH & YAW
    
    PITCH & YAW --> CAM_MOUNT
    CAM_MOUNT --> HUB_SYNC
    
    PITCH --> RATE
    YAW --> RATE
    RATE --> HUB_RX
    HUB_RX --> SHIFT
    SHIFT --> COMP

    style Hub fill:#ff9800,color:#000
    style Gimbal fill:#4caf50,color:#fff
```

---

## 6. Evidence Collection Pipeline

```mermaid
flowchart LR
    subgraph Trigger["Alert Trigger"]
        ALERT["High Confidence<br/>Detection"]
    end

    subgraph Collect["Data Collection"]
        RGB_F["RGB Frame<br/>1920×1080"]
        THM_F["Thermal Patch<br/>100×100"]
        BBOX["Detection Bbox<br/>[x1,y1,x2,y2]"]
        TRACK["Track Info<br/>ID, age, history"]
        GPS_C["GPS Coords<br/>lat, lon, alt"]
        GMB_A["Gimbal Angles<br/>pitch, yaw, roll"]
        LASER_D["Laser Distance<br/>meters"]
        TS["Timestamp<br/>milliseconds"]
    end

    subgraph Package["Evidence Package"]
        VIDEO["H.264 Video<br/>±10 sec clip"]
        META["JSON Metadata<br/>All sensor data"]
        SCREEN["Key Frames<br/>3-5 stills"]
    end

    subgraph Store["Storage"]
        SD["SD Card<br/>Local backup"]
        CLOUD["AWS S3<br/>Cloud storage"]
        DB["Database<br/>Searchable"]
    end

    ALERT --> RGB_F & THM_F & BBOX & TRACK & GPS_C & GMB_A & LASER_D & TS
    RGB_F --> VIDEO
    THM_F --> META
    BBOX --> META
    TRACK --> META
    GPS_C --> META
    GMB_A --> META
    LASER_D --> META
    TS --> META
    VIDEO --> SCREEN

    VIDEO --> SD
    META --> SD
    SCREEN --> SD
    
    SD --> CLOUD
    CLOUD --> DB

    style ALERT fill:#d32f2f,color:#fff
    style VIDEO fill:#4caf50,color:#fff
```

---

## 7. System Wiring Diagram

```mermaid
graph LR
    subgraph Drone["Drone Body"]
        BAT["🔋 Battery<br/>22.2V LiPo"]
        FC["Flight Controller<br/>Pixhawk/DJI"]
    end

    subgraph Power["Power System"]
        BEC1["BEC 12V<br/>Gimbal + Sensors"]
        BEC2["BEC 5V<br/>Jetson + Hub"]
    end

    subgraph Sensors["Sensors"]
        RGB["RGB Cam"]
        THM["Thermal Cam"]
        LASER["Laser"]
    end

    subgraph Gimbal_Sys["Gimbal System"]
        HUB["Gimbal Hub"]
        GMB["Gremsy VIO F1"]
    end

    subgraph Compute["Compute"]
        JET["Jetson Orin NX"]
        SD["SD Card"]
        LTE["LTE Modem"]
    end

    BAT --> BEC1 & BEC2
    BEC1 --> GMB & HUB
    BEC2 --> JET
    
    RGB -->|HDMI| HUB
    THM -->|USB| HUB
    LASER -->|Serial| HUB
    
    HUB -->|PWM| GMB
    HUB -->|USB3| JET
    HUB -->|Serial| JET
    
    GMB -->|MAVLink| JET
    FC -->|MAVLink| JET
    
    JET --> SD
    JET --> LTE

    style HUB fill:#ff9800,color:#000
    style JET fill:#2196f3,color:#fff
```

---

## 8. Module Dependency Graph

```mermaid
graph TB
    subgraph Core["Core Modules"]
        UTILS["utils/<br/>Common utilities"]
    end

    subgraph Detection_Mod["Detection"]
        DET_MOD["detection/<br/>detector.py"]
        CBAM_MOD["detection/<br/>cbam.py"]
    end

    subgraph Tracking_Mod["Tracking"]
        TRK_MOD["tracking/<br/>tracker.py"]
        KALMAN["tracking/<br/>kalman.py"]
    end

    subgraph Thermal_Mod["Thermal"]
        THM_FEAT["thermal/<br/>features.py"]
        THM_LSTM["thermal/<br/>verifier.py"]
    end

    subgraph Gimbal_Mod["Gimbal"]
        GMB_HUB["gimbal/<br/>hub.py"]
        GMB_CTRL["gimbal/<br/>controller.py"]
    end

    subgraph GPS_Mod["GPS"]
        GPS_TRANS["gps/<br/>transformer.py"]
    end

    subgraph Evidence_Mod["Evidence"]
        EVD_COLL["evidence/<br/>collector.py"]
        EVD_STORE["evidence/<br/>storage.py"]
    end

    subgraph Main["Main Pipeline"]
        PIPE["pipeline.py"]
    end

    UTILS --> DET_MOD & TRK_MOD & THM_FEAT & GMB_HUB & GPS_TRANS & EVD_COLL
    
    CBAM_MOD --> DET_MOD
    KALMAN --> TRK_MOD
    THM_FEAT --> THM_LSTM
    GMB_HUB --> GMB_CTRL
    
    DET_MOD --> PIPE
    TRK_MOD --> PIPE
    THM_LSTM --> PIPE
    GMB_CTRL --> PIPE
    GPS_TRANS --> PIPE
    EVD_COLL --> PIPE
    EVD_STORE --> EVD_COLL

    style PIPE fill:#ff6b6b,color:#fff
    style GMB_HUB fill:#ff9800,color:#000
```

---

## 9. Performance Specifications

### Latency Breakdown

| Stage | Time | Cumulative |
| ----- | ---- | ---------- |
| Hub → Jetson transfer | 2ms | 2ms |
| Preprocessing | 3ms | 5ms |
| YOLOv12n backbone | 7ms | 12ms |
| Thermal branch | 2ms | 14ms |
| CBAM fusion | 1ms | 15ms |
| Detection head | 2ms | 17ms |
| NMS + filter | 1ms | 18ms |
| ByteTrack | 2ms | 20ms |
| Thermal LSTM | 2ms | 22ms |
| **Total inference** | **22ms** | - |
| Gimbal command | 3ms | 25ms |
| GPS transform | 2ms | 27ms |
| **Total pipeline** | **27ms** | ✅ |

### Hardware Resources

| Resource | Allocation | Headroom |
| -------- | ---------- | -------- |
| GPU Memory | 2.5GB | 5.5GB free |
| CPU | 60% | 40% free |
| Power | 18W | 7W under TDP |
| Temperature | 60°C | 25°C to throttle |

---

## 10. Deployment Topology

```mermaid
graph TB
    subgraph Drone["Drone (Edge)"]
        JET["Jetson Orin NX"]
        HUB["Gimbal Hub"]
        GMB["Gremsy Gimbal"]
        SENS["Sensors"]
        LTE["LTE Modem"]
    end

    subgraph Ground["Ground Station"]
        OP_PC["Operator PC<br/>Dashboard"]
        RC_TX["RC Controller"]
    end

    subgraph Cloud["AWS Cloud"]
        S3["S3 Bucket<br/>Evidence Storage"]
        RDS["RDS Database<br/>Metadata"]
        API["API Gateway<br/>REST Interface"]
    end

    subgraph Law["Law Enforcement"]
        PORTAL["Evidence Portal<br/>Web Interface"]
        MOBILE["Mobile App<br/>Field Access"]
    end

    SENS --> HUB --> GMB
    HUB --> JET
    JET <--> LTE
    
    LTE <-->|4G/5G| S3
    LTE <-->|4G/5G| API
    
    RC_TX -->|RC Link| JET
    JET -->|WiFi| OP_PC
    
    S3 --> RDS
    RDS --> API
    API --> PORTAL
    API --> MOBILE

    style JET fill:#2196f3,color:#fff
    style Cloud fill:#ff9800,color:#000
```

---

## Quick Reference Tables

### Class Detection Thresholds

| Class | Min Confidence | Alert Level |
| ----- | -------------- | ----------- |
| knife_machete | 0.68 | MEDIUM |
| knife_machete | 0.72 | HIGH |
| metal_rod | 0.65 | MEDIUM |
| metal_rod | 0.72 | HIGH |
| any + zoom confirm | 0.82 | VERY_HIGH |

### Module File Mapping

| Component | File | Key Class |
| --------- | ---- | --------- |
| Detection | `src/detection/detector.py` | `WeaponDetector` |
| CBAM | `src/detection/detector.py` | `CBAMFusion` |
| Tracking | `src/tracking/tracker.py` | `ByteTracker` |
| Thermal | `src/thermal/verifier.py` | `ThermalVerifier` |
| Gimbal Hub | `src/gimbal/hub.py` | `GimbalHub` |
| Gimbal Ctrl | `src/gimbal/controller.py` | `GimbalController` |
| GPS | `src/gps/transformer.py` | `GPSTransformer` |
| Evidence | `src/evidence/collector.py` | `EvidenceCollector` |
| Pipeline | `src/pipeline.py` | `WeaponDetectionPipeline` |

---

**Document Version**: v7.0  
**Last Updated**: January 9, 2026  
**Status**: ✅ Complete with Gimbal Hub
