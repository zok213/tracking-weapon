# 🔄 Vietnam Weapon Detection System - Complete Dataflow v7.0

## System Pipeline Overview

This document provides the complete end-to-end dataflow for detecting **persons carrying weapons** from nadir drone footage, with full **gimbal hub integration**.

---

## COMPLETE SYSTEM DATAFLOW

```
================================================================================
                        INPUT PIPELINE (Gimbal Hub)
================================================================================
    ↓
Video Stream: Nadir drone (50-70m altitude, 1920×1080 @ 30 FPS)
├─ RGB channel: Standard visible spectrum (HDMI → Gimbal Hub)
├─ Thermal channel: Synchronized IR (640×512 @ 30 FPS, Gremsy thermal → USB)
└─ Laser rangefinder: Distance measurement (Serial → Gimbal Hub)
    ↓
Gimbal Hub (Gremsy Hub) - Central Data Multiplexer:
├─ Power distribution: 12V @ 3A (gimbal/sensors), 5V @ 2A (logic)
├─ Signal multiplexing: Route RGB (HDMI), Thermal (USB), Laser (Serial)
├─ Frame synchronization: Align RGB + Thermal timestamps (±1ms tolerance)
├─ UART bridge: Convert laser serial to processable format (115200 baud)
└─ Output to Jetson: USB3 video streams + Serial data + MAVLink telemetry @ 50Hz
    ↓
================================================================================
                            PREPROCESSING
================================================================================
    ↓
├─ Receive synced frames from Gimbal Hub:
│  ├─ RGB frame: 1920×1080 with timestamp
│  ├─ Thermal frame: 640×512 with timestamp
│  ├─ Frame sync drift: Must be ≤1ms
│  └─ Sequence ID: Unique frame identifier
│
├─ Resize RGB: 1920×1080 → 640×640 (letterbox, maintain aspect ratio)
├─ Resize Thermal: 640×512 → 640×640 (bilinear interpolation, match RGB)
├─ Normalize: RGB [0-255] → [0-1], Thermal [0-255] → [0-1]
├─ Channel stack: RGB (3ch) + Thermal (1ch replicated to 3ch for backbone)
└─ Synchronize: Frame-level alignment verified (timestamp matching from Hub)
    ↓
Preprocessing output:
├─ rgb_tensor: [1, 3, 640, 640] float32
├─ thermal_tensor: [1, 3, 640, 640] float32
├─ frame_metadata: {timestamp_ms, seq_id, laser_distance, gimbal_angles}
└─ Latency: 3ms
    ↓
================================================================================
                    DETECTION BACKBONE (YOLO26n + CBAM Fusion)
================================================================================
    ↓
├─ RGB branch (Primary detection):
│  ├─ Input: 640×640 RGB image [1, 3, 640, 640]
│  ├─ Backbone: 8 convolutional stages (CSP-Darknet style)
│  │  ├─ Stage 1: Conv2d(3→16, k=3, s=2) → 320×320
│  │  ├─ Stage 2: C3k2 block → 160×160
│  │  ├─ Stage 3: C3k2 block → 80×80 (P3 features)
│  │  ├─ Stage 4: C3k2 block → 40×40 (P4 features)
│  │  ├─ Stage 5-8: C3k2 blocks → 20×20 (P5 features)
│  │  └─ Total params: ~5.5M (YOLOv12n lightweight)
│  │
│  ├─ FPN layers (Feature Pyramid Network):
│  │  ├─ P3: 80×80×256 (small objects - weapons at distance)
│  │  ├─ P4: 40×40×512 (medium objects - persons)
│  │  └─ P5: 20×20×1024 (large objects - vehicles)
│  │
│  └─ Output: Multi-scale feature maps for detection head
│
└─ Thermal branch (CBAM Early Fusion):
   ├─ Input: 640×640 thermal image [1, 3, 640, 640]
   ├─ Lightweight backbone: 5 conv stages (~1.5M params, 30% of RGB)
   │  ├─ Designed for thermal-specific patterns
   │  ├─ Focus on temperature gradients and cold spots
   │  └─ Output: Thermal features at P3 level only (small objects)
   │
   ├─ Fusion point: After backbone, at P3 layer (small object focus)
   ├─ CBAM Fusion mechanism:
   │  ├─ Channel attention (WHAT to focus on):
   │  │  ├─ Global avg pool + max pool → channel descriptors
   │  │  ├─ FC(256→16→256) → channel weights
   │  │  ├─ Learn: Which modality to trust per channel
   │  │  └─ Daytime: 70% RGB / 30% Thermal
   │  │      Night: 40% RGB / 60% Thermal
   │  │
   │  └─ Spatial attention (WHERE to focus):
   │     ├─ Channel-wise avg + max → spatial map
   │     ├─ Conv(2→1, k=7) → spatial weights
   │     ├─ Highlight cold spots (weapon metal signatures)
   │     └─ Suppress warm body regions (less useful for weapons)
   │
   └─ Output: Fused features [1, 256, 80, 80] ready for detection
    ↓
Backbone latency breakdown:
├─ RGB backbone: 7ms
├─ Thermal branch: 2ms
├─ CBAM fusion: 1ms
└─ Total: 10ms
    ↓
================================================================================
                    DETECTION HEAD (Person + Weapon Detection)
================================================================================
    ↓
├─ Input: Fused features from P3 (80×80), P4 (40×40), P5 (20×20) layers
│
├─ Detection classes (4 total):
│  ├─ Class 0: person (rider on motorcycle - primary target context)
│  ├─ Class 1: knife_machete (melee blades - hard class)
│  ├─ Class 2: metal_rod (crowbars, pipes - easier class)
│  └─ Class 3: motorcycle (context for rider detection)
│
├─ Prediction grid per scale:
│  ├─ P3 (stride 8): 80×80 grid = 6,400 cells → weapons (5-30 pixels)
│  ├─ P4 (stride 16): 40×40 grid = 1,600 cells → persons (30-100 pixels)
│  └─ P5 (stride 32): 20×20 grid = 400 cells → vehicles (100+ pixels)
│
├─ Outputs per grid cell:
│  ├─ Bounding box: 4 coords (x_center, y_center, width, height) normalized
│  ├─ Objectness score: Is there an object? [0-1]
│  ├─ Class probabilities: [person, knife, rod, motorcycle]
│  └─ Confidence: objectness × class_prob
│
├─ Person-Weapon Association:
│  ├─ Detect person bbox (rider on motorcycle)
│  ├─ Detect weapon bbox (knife/rod near person)
│  ├─ Associate if: IoU(person, weapon) > 0.1 OR
│  │              distance(person_center, weapon_center) < 50px
│  ├─ Result: weapon_owner_id linking weapon to specific person
│  └─ Benefit: Track "Person A carrying Knife" not just "Knife"
│
└─ Output: Raw detections [N × (x, y, w, h, objectness, class_probs[4])]
    ↓
Detection head latency: 3ms
    ↓
================================================================================
                POST-PROCESSING (Class-Specific Filtering)
================================================================================
    ↓
├─ Confidence thresholds (class-specific):
│  ├─ person: Keep if conf ≥ 0.50 (easier to detect)
│  ├─ knife_machete: Keep if conf ≥ 0.68 (stricter for hard class)
│  ├─ metal_rod: Keep if conf ≥ 0.65 (slightly easier than knife)
│  └─ motorcycle: Keep if conf ≥ 0.50 (context only)
│
├─ NMS-Free (YOLO26):
│  ├─ No IoU threshold needed
│  └─ End-to-End detection output
│
├─ Person-Weapon Linking:
│  ├─ For each weapon detection:
│  │  ├─ Find nearest person detection
│  │  ├─ Compute spatial relationship:
│  │  │  ├─ IoU overlap (weapon on person's body)
│  │  │  ├─ Distance (weapon near person)
│  │  │  └─ Position (weapon at hip/back/rack area)
│  │  ├─ If linked: weapon.owner_id = person.track_id
│  │  └─ If unlinked: weapon.owner_id = None (standalone weapon)
│  │
│  └─ Benefits:
│     ├─ Reduces false positives (weapons without persons may be bike parts)
│     ├─ Enables "Person A carrying Machete" alerts
│     └─ Supports legal evidence ("Suspect in red shirt with crowbar")
│
└─ Output: Filtered detections with owner associations
   [N × {bbox, class_id, confidence, owner_id}]
    ↓
Latency: <1ms
    ↓
================================================================================
            THERMAL CONFIDENCE FUSION (Physics-Based Verification)
================================================================================
    ↓
├─ For each weapon detection:
│  ├─ Extract thermal ROI (crop thermal frame at weapon bbox)
│  ├─ Pad bbox by 20% (capture surrounding context)
│  │
│  ├─ Compute 5 hand-crafted thermal features:
│  │  │
│  │  ├─ F1: Temperature Gradient (Edge Sharpness)
│  │  │  ├─ Formula: |∇T| = sqrt((dT/dx)² + (dT/dy)²)
│  │  │  ├─ Weapon signature: 10-30 °C/cm (sharp metal edges)
│  │  │  ├─ Non-weapon: 1-5 °C/cm (smooth gradients)
│  │  │  └─ Normalized score: gradient / 30.0
│  │  │
│  │  ├─ F2: Cold Spot Ratio
│  │  │  ├─ Formula: count(T < body_temp - 5°C) / total_pixels
│  │  │  ├─ Weapon signature: 30-60% cold pixels (metal absorbs cold)
│  │  │  ├─ Non-weapon: <20% cold pixels
│  │  │  └─ Normalized score: ratio / 0.6
│  │  │
│  │  ├─ F3: Temperature Contrast
│  │  │  ├─ Formula: max_temp - min_temp in ROI
│  │  │  ├─ Weapon signature: 15-35°C (cold metal vs warm body)
│  │  │  ├─ Non-weapon: <10°C contrast
│  │  │  └─ Normalized score: contrast / 35.0
│  │  │
│  │  ├─ F4: Spatial Concentration
│  │  │  ├─ Find center of mass of cold pixels
│  │  │  ├─ Compute variance of cold pixel positions
│  │  │  ├─ Weapon signature: Low variance (compact cold cluster)
│  │  │  ├─ Non-weapon: High variance (scattered)
│  │  │  └─ Normalized score: 1.0 - (variance / max_variance)
│  │  │
│  │  └─ F5: Temporal Consistency
│  │     ├─ Compare features with previous 5 frames
│  │     ├─ Compute cosine similarity
│  │     ├─ Weapon signature: >0.8 correlation (stable across frames)
│  │     ├─ Non-weapon: <0.5 correlation (changing patterns)
│  │     └─ Normalized score: correlation coefficient
│  │
│  ├─ 30-Frame Buffer for LSTM:
│  │  ├─ Stack features: [30 × 5] tensor per track
│  │  ├─ LSTM model: LSTM(input=5, hidden=32) → Dense(16) → Dense(1) → Sigmoid
│  │  ├─ Output: thermal_confidence ∈ [0, 1]
│  │  └─ Latency: 2ms
│  │
│  ├─ Confidence Fusion Decision:
│  │  ├─ Rule 1: If (RGB_conf > 0.7) AND (thermal_conf > 0.6):
│  │  │  └─ combined_conf = 0.95 (HIGH confidence, both agree)
│  │  │
│  │  ├─ Rule 2: If (RGB_conf > 0.7) AND (thermal_conf > 0.3):
│  │  │  └─ combined_conf = 0.75 + 0.15 × thermal_conf
│  │  │
│  │  ├─ Rule 3: If (thermal_conf < 0.3):
│  │  │  └─ combined_conf = RGB_conf × 0.85 (thermal doesn't support)
│  │  │
│  │  └─ Default: weighted_avg = 0.7 × RGB_conf + 0.3 × thermal_conf
│  │
│  └─ Confidence boost summary:
│     ├─ metal_rod (pipe/crowbar): +10-15% boost (strong thermal signature)
│     ├─ knife_machete: +5-8% boost (weaker thermal, smaller object)
│     └─ No thermal ROI available: Keep RGB confidence unchanged
│
└─ Output: Enhanced detections with thermal verification
   [N × {bbox, class_id, combined_conf, thermal_conf, owner_id}]
    ↓
Thermal fusion latency: 2ms
    ↓
================================================================================
                MULTI-OBJECT TRACKING (ByteTrack Algorithm)
================================================================================
    ↓
├─ Input: Filtered detections from frame N
│  [N × {bbox, class_id, combined_conf, owner_id}]
│
├─ ByteTrack Two-Stage Association:
│  │
│  ├─ STAGE 1: High-Confidence Association (conf ≥ 0.70)
│  │  ├─ Get all detections with confidence ≥ 0.70
│  │  ├─ Get all ACTIVE tracklets (matched in recent frames)
│  │  ├─ Compute IoU distance matrix:
│  │  │  cost[i,j] = 1.0 - IoU(track[i].bbox, detection[j].bbox)
│  │  ├─ Apply Hungarian algorithm (optimal linear assignment)
│  │  ├─ Match threshold: IoU ≥ 0.50 (cost ≤ 0.50)
│  │  ├─ Results:
│  │  │  ├─ Matched tracks: Update state with new detection
│  │  │  ├─ Unmatched tracks: Go to Stage 2
│  │  │  └─ Unmatched detections: Create new tracks
│  │  └─ Handle gimbal motion compensation (subtract gimbal shift from tracks)
│  │
│  └─ STAGE 2: Low-Confidence Recovery (0.30 ≤ conf < 0.70)
│     ├─ Purpose: Recover tracks through brief occlusions or missed detections
│     ├─ Get detections with 0.30 ≤ confidence < 0.70
│     ├─ Get UNMATCHED tracklets from Stage 1
│     ├─ Compute IoU distance matrix
│     ├─ Apply Hungarian algorithm
│     ├─ Results:
│     │  ├─ Matched: Recover track (occlusion handled!)
│     │  └─ Unmatched tracks: Keep in buffer for 30 frames
│     └─ KEY INSIGHT: Low-conf detections often occur during occlusion
│
├─ Track State Machine:
│  ├─ NEW → TENTATIVE: Created from unmatched high-conf detection
│  ├─ TENTATIVE → ACTIVE: After 3 consecutive frame matches
│  ├─ ACTIVE → ACTIVE: Matched in current frame
│  ├─ ACTIVE → LOST: Unmatched for 1+ frames (Kalman prediction)
│  ├─ LOST → ACTIVE: Re-matched within 30 frames
│  └─ LOST → DELETED: Unmatched for 30 consecutive frames
│
├─ Kalman Filter (per track):
│  ├─ State vector: [x, y, w, h, vx, vy, vw, vh]
│  ├─ Predict: Estimate next bbox using velocity
│  ├─ Update: Correct estimate when detection matches
│  ├─ Gimbal compensation: Subtract gimbal motion from velocity
│  └─ Handles: Brief detection dropouts, motion blur
│
├─ Person-Weapon Track Linking:
│  ├─ Maintain association from detection phase
│  ├─ If weapon.owner_id != None:
│  │  ├─ Link weapon track to person track
│  │  ├─ Inherit person's GPS trajectory
│  │  └─ Alert shows: "Person Track #42 carrying Machete Track #87"
│  └─ If person moves, weapon follows (spatial consistency check)
│
├─ Track output per frame:
│  ├─ track_id: Unique identifier (persistent across video)
│  ├─ bbox: Current position [x, y, w, h]
│  ├─ class_id: 0=person, 1=knife, 2=rod, 3=motorcycle
│  ├─ confidence: Smoothed over last 10 frames
│  ├─ age: Frames since track creation
│  ├─ owner_id: Person track ID (for weapons)
│  └─ state: ACTIVE, LOST, TENTATIVE
│
└─ Tracking output:
   [M × {track_id, bbox, class_id, confidence, age, owner_id, state}]
    ↓
Tracking latency: 2ms
    ↓
================================================================================
        GIMBAL CONTROL (Gremsy MAVLink Protocol via Gimbal Hub)
================================================================================
    ↓
├─ Gimbal Hub Interface:
│  ├─ Receive: Synced frames + telemetry from hub
│  ├─ Send: Gimbal commands through hub's UART bridge
│  ├─ Protocol: MAVLink v2 at 115200 baud
│  └─ Update rate: 50Hz (20ms command interval)
│
├─ Trigger Conditions for Gimbal Action:
│  │
│  ├─ Condition 1: Low confidence + small object
│  │  ├─ If (track_confidence < 0.65) AND (bbox_width < 25px):
│  │  │  └─ Object too small and uncertain → Zoom for clarity
│  │  ├─ Action: Center on target + Zoom 8×
│  │  └─ Goal: Get better resolution for re-detection
│  │
│  ├─ Condition 2: New high-value detection
│  │  ├─ If (class == knife_machete) AND (first detection):
│  │  │  └─ Knife is hard class, needs confirmation
│  │  ├─ Action: Center on target + Zoom 4×
│  │  └─ Goal: Verify blade shape in higher resolution
│  │
│  └─ Condition 3: Track losing confidence
│     ├─ If (confidence dropping 3 frames) AND (still ACTIVE):
│     │  └─ Object becoming occluded or blurred
│     ├─ Action: Center on last known position + Zoom 4×
│     └─ Goal: Maintain tracking through occlusion
│
├─ Centering Algorithm:
│  ├─ Target: Weapon bbox center (x_center, y_center)
│  ├─ Frame center: (320, 256) at 640×512 thermal / (960, 540) at 1920×1080 RGB
│  ├─ Error vector: error = (target - frame_center)
│  │
│  ├─ Pixel to Angle conversion:
│  │  ├─ Horizontal FOV: 60° (typical gimbal camera)
│  │  ├─ Vertical FOV: 45°
│  │  ├─ yaw_angle = error_x × (60° / frame_width)
│  │  ├─ pitch_angle = error_y × (45° / frame_height)
│  │  └─ Apply PID smoothing: Kp=0.5, Kd=0.1 (prevent oscillation)
│  │
│  ├─ MAVLink Command (via Gimbal Hub):
│  │  ├─ MAV_CMD_DO_MOUNT_CONTROL (205)
│  │  │  ├─ param1: pitch_angle (degrees)
│  │  │  ├─ param2: roll_angle (0, stabilized)
│  │  │  ├─ param3: yaw_angle (degrees)
│  │  │  └─ param7: MAV_MOUNT_MODE_MAVLINK_TARGETING
│  │  │
│  │  └─ Command flow: Jetson → Hub → Gimbal (via UART)
│  │
│  └─ Centering tolerance: ±2-3 pixels (stop adjusting when centered)
│
├─ Adaptive Zoom Control:
│  ├─ Zoom level selection based on target size:
│  │  ├─ bbox_width <  15px: Zoom 16× (maximum, very small object)
│  │  ├─ bbox_width <  30px: Zoom 8× (small object, typical for weapons)
│  │  ├─ bbox_width <  60px: Zoom 4× (medium object)
│  │  ├─ bbox_width < 100px: Zoom 2× (larger object)
│  │  └─ bbox_width ≥ 100px: Zoom 1× (no zoom needed)
│  │
│  ├─ Zoom command (MAVLink):
│  │  ├─ MAV_CMD_SET_CAMERA_ZOOM (531)
│  │  │  ├─ param1: ZOOM_TYPE_CONTINUOUS (0)
│  │  │  └─ param2: zoom_level (1.0 - 20.0)
│  │  │
│  │  └─ Dwell time: Wait 150-200ms for optical refocus
│  │
│  └─ Effective resolution at zoom:
│     ├─ 1×: 1920×1080 → weapon ~10-20px
│     ├─ 4×: 7680×4320 effective → weapon ~40-80px
│     ├─ 8×: 15360×8640 effective → weapon ~80-160px
│     └─ 16×: 30720×17280 effective → weapon ~160-320px (very clear!)
│
├─ Re-detection at Higher Zoom:
│  ├─ After zoom stabilizes (200ms delay):
│  │  ├─ Capture zoomed frame from Gimbal Hub
│  │  ├─ Preprocess: Crop center 640×640 (zoomed region only)
│  │  ├─ Run YOLOv12n inference on zoomed image
│  │  └─ Higher resolution = better feature visibility
│  │
│  ├─ Confirmation logic:
│  │  ├─ If re-detected with conf ≥ 0.70:
│  │  │  └─ Boost original confidence +20% (zoom confirmed!)
│  │  │  └─ Alert level: Upgrade to VERY_HIGH
│  │  │
│  │  ├─ If re-detected with conf < 0.50:
│  │  │  └─ Likely false positive, downgrade alert
│  │  │
│  │  └─ If not re-detected:
│  │     └─ Keep original confidence, continue tracking
│  │
│  └─ Benefit: Resolves ambiguous detections with higher resolution
│
└─ Gimbal Motion Compensation (for ByteTrack):
   ├─ Read gimbal telemetry from Hub (50Hz):
   │  ├─ GIMBAL_DEVICE_ATTITUDE (MAVLink message)
   │  │  ├─ pitch, yaw, roll (current angles in degrees)
   │  │  └─ pitch_rate, yaw_rate, roll_rate (angular velocity)
   │  │
   │  └─ Hub provides synchronized telemetry with frames
   │
   ├─ Compute image shift due to gimbal movement:
   │  ├─ dt = 33ms (frame interval)
   │  ├─ dx_pixels = yaw_rate × focal_length × dt / 57.3
   │  ├─ dy_pixels = pitch_rate × focal_length × dt / 57.3
   │  └─ shift_vector = (dx_pixels, dy_pixels)
   │
   ├─ Apply compensation in ByteTrack:
   │  ├─ For each predicted track bbox:
   │  │  predicted_bbox.x += dx_pixels
   │  │  predicted_bbox.y += dy_pixels
   │  ├─ Result: Track positions corrected for gimbal motion
   │  └─ Kalman filter learns TRUE object motion only
   │
   └─ Benefit: Prevents false ID switches when gimbal pans
    ↓
Gimbal control latency: 3ms (command generation, not including execution)
    ↓
================================================================================
                GPS COORDINATE TRANSFORMATION
================================================================================
    ↓
├─ Inputs (all synchronized via Gimbal Hub):
│  ├─ Detection bbox center: (u, v) pixels in 640×640 frame
│  ├─ Gimbal angles: pitch, yaw, roll (from Hub telemetry)
│  ├─ Gimbal offset: (dx=0, dy=0, dz=-0.1m) from drone center
│  ├─ Laser distance: D meters (from Gremsy laser via Hub, ±2.5m accuracy)
│  ├─ Drone GPS: (lat_drone, lon_drone, alt_drone) from MAVLink
│  ├─ Drone attitude: roll, pitch, yaw (from IMU via MAVLink)
│  └─ Timestamp: Microsecond-level sync across all sensors
│
├─ Step 1: Pixel → Camera Frame
│  ├─ Camera intrinsics (calibration):
│  │  ├─ fx, fy: 1000 pixels (focal length)
│  │  ├─ cx, cy: 320, 256 (principal point, image center)
│  │  └─ Distortion: k1, k2, p1, p2 (radial/tangential)
│  │
│  ├─ Undistort pixel if needed:
│  │  (u', v') = undistort(u, v, distortion_coeffs)
│  │
│  ├─ Normalized camera coordinates:
│  │  xc = (u' - cx) / fx
│  │  yc = (v' - cy) / fy
│  │  zc = 1.0 (unit depth)
│  │
│  └─ Ray direction in camera frame:
│     ray_cam = normalize([xc, yc, zc])
│
├─ Step 2: Camera Frame → Gimbal Frame
│  ├─ Gimbal angles from Hub telemetry:
│  │  ├─ pitch_g: -90° (nadir) to +30° (forward)
│  │  ├─ yaw_g: -180° to +180° (azimuth)
│  │  └─ roll_g: ~0° (stabilized by gimbal)
│  │
│  ├─ Rotation matrix R_gimbal:
│  │  R_gimbal = Rz(yaw_g) × Ry(pitch_g) × Rx(roll_g)
│  │
│  └─ Ray in gimbal frame:
│     ray_gimbal = R_gimbal × ray_cam
│
├─ Step 3: Gimbal Frame → Drone Body Frame
│  ├─ Gimbal mount offset (fixed installation):
│  │  ├─ dx: 0m (centered)
│  │  ├─ dy: 0m (centered)
│  │  └─ dz: -0.1m (below drone body)
│  │
│  ├─ Ray in drone body frame:
│  │  ray_body = ray_gimbal (no rotation, gimbal aligned with body)
│  │
│  └─ Apply gimbal offset:
│     ray_body_origin = drone_center + offset
│
├─ Step 4: Drone Body Frame → World Frame (NED)
│  ├─ Drone attitude from IMU:
│  │  ├─ roll_d: Typically ±5° during flight
│  │  ├─ pitch_d: Typically ±10° during flight
│  │  └─ yaw_d: 0-360° (heading)
│  │
│  ├─ Rotation matrix R_drone:
│  │  R_drone = Rz(yaw_d) × Ry(pitch_d) × Rx(roll_d)
│  │
│  ├─ Ray in NED (North-East-Down) frame:
│  │  ray_NED = R_drone × ray_body
│  │
│  ├─ Scale by laser distance:
│  │  point_NED = ray_NED × laser_distance_D
│  │
│  └─ Result: 3D offset from drone in NED coordinates
│     (north_offset, east_offset, down_offset) in meters
│
├─ Step 5: NED → WGS84 GPS Coordinates
│  ├─ Drone GPS (reference point):
│  │  lat_drone, lon_drone, alt_drone (WGS84)
│  │
│  ├─ Convert NED to lat/lon change:
│  │  ├─ Earth radius: R = 6,378,137m (WGS84 equatorial)
│  │  ├─ lat_change = north_offset / R × (180/π)
│  │  ├─ lon_change = east_offset / (R × cos(lat_drone)) × (180/π)
│  │  └─ alt_change = -down_offset (NED down is negative altitude)
│  │
│  └─ Target GPS coordinates:
│     lat_target = lat_drone + lat_change
│     lon_target = lon_drone + lon_change
│     alt_target = alt_drone + alt_change
│
├─ Accuracy Analysis:
│  ├─ Laser distance: ±2.5m (Gremsy spec)
│  ├─ Gimbal angles: ±0.1° (encoder precision)
│  ├─ Drone GPS: ±2m (RTK) or ±5m (standard)
│  ├─ Total horizontal: ±2.5m with RTK, ±5m without
│  └─ Vertical: ±1m (barometric + laser)
│
└─ Output: Single GPS point per track per frame
   {lat: float, lon: float, alt: float, accuracy_m: float, timestamp_ms: int}
    ↓
GPS transformation latency: 2ms
    ↓
================================================================================
                ALERT GENERATION & OPERATOR DASHBOARD
================================================================================
    ↓
├─ Alert Classification (Person + Weapon combined):
│  │
│  ├─ VERY HIGH PRIORITY (🔴 Immediate dispatch):
│  │  ├─ metal_rod ≥ 0.82 confidence
│  │  ├─ OR: Zoom re-detection confirmed (+20% boost applied)
│  │  ├─ OR: knife_machete ≥ 0.78 with thermal confirmation
│  │  ├─ Operator action: Dispatch law enforcement immediately
│  │  └─ Review time: 1-2 seconds
│  │
│  ├─ HIGH PRIORITY (🟠 Monitor closely):
│  │  ├─ knife_machete ≥ 0.72
│  │  ├─ OR: metal_rod ≥ 0.72
│  │  ├─ OR: Person-weapon association confirmed
│  │  ├─ Operator action: Monitor + prepare dispatch
│  │  └─ Review time: 2-3 seconds
│  │
│  ├─ MEDIUM PRIORITY (🟡 Investigate):
│  │  ├─ knife_machete ≥ 0.65
│  │  ├─ OR: Track persists ≥ 5 frames
│  │  ├─ OR: Thermal boost applied
│  │  ├─ Operator action: Request gimbal zoom, investigate
│  │  └─ Review time: 3-5 seconds
│  │
│  └─ LOW PRIORITY (🟢 Verify):
│     ├─ Any weapon ≥ 0.55
│     ├─ Likely false positive (pipe on bike, mirror reflection)
│     ├─ Operator action: Quick verify, usually dismiss
│     └─ Review time: 4-5 seconds
│
├─ Alert Package Contents:
│  ├─ Video evidence:
│  │  ├─ Current frame (640×640, annotated with bboxes)
│  │  ├─ ±5 frame context (10 frames total, ~330ms)
│  │  ├─ Zoomed frame if available
│  │  └─ Thermal overlay side-by-side
│  │
│  ├─ Detection metadata:
│  │  ├─ Weapon class: "Machete" / "Crowbar" / "Metal Pipe"
│  │  ├─ Confidence: 0-100% (visual bar + number)
│  │  ├─ Thermal confidence: 0-100% (separate indicator)
│  │  ├─ Combined confidence: Weighted fusion result
│  │  └─ Boost reason: "Thermal verified" / "Zoom confirmed"
│  │
│  ├─ Person association:
│  │  ├─ Owner track ID: "Person #42"
│  │  ├─ Person description: "Rider on motorcycle"
│  │  ├─ Clothing color (if detectable): "Red shirt"
│  │  └─ Motorcycle type (if detected): "Honda Wave"
│  │
│  ├─ Location data:
│  │  ├─ GPS coordinates: 10.7769° N, 106.6970° E
│  │  ├─ Accuracy: ±2.5m
│  │  ├─ Altitude: 52m
│  │  ├─ Street name (geocoded): "Nguyen Hue St, District 1"
│  │  └─ Map thumbnail: Mini-map with location marker
│  │
│  ├─ Tracking data:
│  │  ├─ Track ID: weapon_20260109_143052_001
│  │  ├─ Track age: 47 frames (1.6 seconds)
│  │  ├─ Trajectory: GPS polyline on map
│  │  ├─ Direction: Heading 45° NE
│  │  └─ Speed estimate: ~30 km/h (motorcycle typical)
│  │
│  ├─ Timestamp:
│  │  ├─ Frame number: 142,857
│  │  ├─ Unix timestamp: 1736416252.347
│  │  └─ Human readable: "2026-01-09 14:30:52"
│  │
│  └─ Operator actions:
│     ├─ [CONFIRM] - Dispatch police, save evidence
│     ├─ [DISMISS] - False positive, discard
│     ├─ [INVESTIGATE] - Request gimbal zoom
│     ├─ [FLAG] - Suspicious but uncertain
│     └─ [NOTE] - Add free-text observation
│
└─ Dashboard User Interface:
   ├─ Main video feed: 640×640 or 1920×1080 (selectable)
   │  ├─ Bounding box overlays (color-coded by class)
   │  ├─ Track ID labels (e.g., "K#42" for knife track 42)
   │  ├─ Confidence percentages
   │  └─ Person-weapon links (dashed lines)
   │
   ├─ Detection history panel (right side):
   │  ├─ Last 50 detections (scrollable)
   │  ├─ Filter by: class, confidence, alert level
   │  └─ Click to jump to frame
   │
   ├─ Alert queue (top):
   │  ├─ New alerts (highest priority first)
   │  ├─ Audio: Beep for HIGH, Alarm for VERY_HIGH
   │  └─ Auto-dismiss after 30s if LOW priority
   │
   ├─ Map view (bottom-left):
   │  ├─ City map (OpenStreetMap / Google Maps)
   │  ├─ Drone position marker
   │  ├─ Weapon GPS locations (color-coded icons)
   │  └─ Track trajectories (polylines)
   │
   ├─ Gimbal control panel (bottom-right):
   │  ├─ Current pitch/yaw angles
   │  ├─ Zoom level indicator
   │  ├─ Manual override buttons
   │  └─ "Center on Track #X" quick button
   │
   └─ Statistics panel:
      ├─ Alerts today: 12 (3 HIGH, 5 MEDIUM, 4 LOW)
      ├─ Confirmed weapons: 2
      ├─ False positive rate: 18%
      └─ System status: ✅ All sensors OK
    ↓
Alert generation latency: <1ms
    ↓
================================================================================
                EVIDENCE LOGGING & CLOUD SYNC
================================================================================
    ↓
├─ Local Storage (Drone SD Card):
│  │
│  ├─ Continuous recording:
│  │  ├─ Full H.264 video: All 30 FPS, quality preset "High"
│  │  ├─ Bitrate: ~15 Mbps (1.8 GB/hour)
│  │  ├─ Filename: flight_20260109_143000.mp4
│  │  └─ Duration: Continuous during flight
│  │
│  ├─ Metadata stream (JSONL format, 1KB per frame):
│  │  {
│  │    "frame_id": 142857,
│  │    "timestamp_ms": 1736416252347,
│  │    "detections": [
│  │      {"bbox": [0.42, 0.38, 0.08, 0.15], "class": 1, "conf": 0.78,
│  │       "thermal_conf": 0.65, "owner_id": 42}
│  │    ],
│  │    "tracks": [
│  │      {"track_id": 87, "class": 1, "state": "ACTIVE", "age": 47}
│  │    ],
│  │    "gps": {"lat": 10.7769, "lon": 106.6970, "alt": 52.0},
│  │    "gimbal": {"pitch": -90.0, "yaw": 15.2, "zoom": 4.0},
│  │    "laser_distance": 51.3,
│  │    "thermal_features": {"F1": 0.72, "F2": 0.45, "F3": 0.88, "F4": 0.67, "F5": 0.91}
│  │  }
│  │
│  ├─ Evidence package (created on CONFIRM):
│  │  ├─ Folder: /evidence/20260109/weapon_001/
│  │  ├─ Contents:
│  │  │  ├─ clip.mp4: 10-second video (±5 sec from trigger)
│  │  │  ├─ frame_001.jpg ... frame_005.jpg: 5 key frames (1080p)
│  │  │  ├─ thermal_001.png ... thermal_005.png: Thermal frames
│  │  │  ├─ trajectory.geojson: GPS track as GeoJSON polyline
│  │  │  ├─ metadata.json: Complete sensor readings
│  │  │  ├─ operator_notes.txt: Free-text observations
│  │  │  └─ signature.sha256: Digital checksum
│  │  │
│  │  └─ Chain of custody:
│  │     ├─ created_by: "operator_badge_12345"
│  │     ├─ created_at: "2026-01-09T14:31:05Z"
│  │     └─ hash: SHA-256 of all files
│  │
│  └─ Retention policy:
│     ├─ Evidence packages: Keep indefinitely (until case closed)
│     ├─ Full video: 30 days rolling
│     └─ Metadata: 90 days rolling
│
└─ Cloud Sync (LTE Modem, Asynchronous):
   │
   ├─ Trigger conditions:
   │  ├─ Immediate: Evidence confirmed by operator
   │  ├─ Scheduled: Daily at 11 PM (bulk upload)
   │  └─ Manual: Operator requests sync
   │
   ├─ Upload pipeline:
   │  ├─ Compress: H.264 already compressed, skip
   │  ├─ Encrypt: AES-256-GCM with per-file key
   │  ├─ Chunk: Split into 10MB parts (LTE reliability)
   │  ├─ Upload: HTTPS PUT to AWS S3
   │  ├─ Retry: 3× with exponential backoff (1s, 5s, 30s)
   │  └─ Verify: SHA-256 checksum after upload
   │
   ├─ Cloud storage structure:
   │  └─ s3://weapon-evidence-bucket/
   │     └─ evidence/
   │        └─ 2026/
   │           └─ 01/
   │              └─ 09/
   │                 └─ weapon_001_1736416252/
   │                    ├─ clip.mp4.enc (encrypted)
   │                    ├─ metadata.json.enc
   │                    └─ manifest.json
   │
   ├─ Backend processing:
   │  ├─ RDS database: Index metadata for search
   │  ├─ Elasticsearch: Full-text search on notes
   │  ├─ Lambda: Generate thumbnails, transcode video
   │  └─ API Gateway: REST interface for law enforcement
   │
   └─ Legal compliance:
      ├─ Encryption: AES-256 at rest and in transit
      ├─ Access control: Role-based (admin, operator, law enforcement)
      ├─ Audit log: Every access logged with IP, timestamp, user
      ├─ Retention: Auto-delete after 90 days unless flagged
      └─ Export: Generate court-admissible evidence package


================================================================================
                        LATENCY SUMMARY
================================================================================

┌─────────────────────────────┬───────────┬────────────┐
│ STAGE                       │ LATENCY   │ CUMULATIVE │
├─────────────────────────────┼───────────┼────────────┤
│ Hub → Jetson transfer       │ 2ms       │ 2ms        │
│ Preprocessing               │ 3ms       │ 5ms        │
│ RGB backbone                │ 7ms       │ 12ms       │
│ Thermal branch              │ 2ms       │ 14ms       │
│ CBAM fusion                 │ 1ms       │ 15ms       │
│ Detection head              │ 2ms       │ 17ms       │
│ NMS + filtering             │ 1ms       │ 18ms       │
│ Person-weapon linking       │ 1ms       │ 19ms       │
│ Thermal features (5)        │ 1ms       │ 20ms       │
│ LSTM verification           │ 1ms       │ 21ms       │
│ ByteTrack                   │ 2ms       │ 23ms       │
│ Gimbal command              │ 2ms       │ 25ms       │
│ GPS transformation          │ 2ms       │ 27ms       │
│ Alert generation            │ 1ms       │ 28ms       │
├─────────────────────────────┼───────────┼────────────┤
│ TOTAL PIPELINE              │ 28ms      │ ✅ < 33ms  │
└─────────────────────────────┴───────────┴────────────┘

Budget: 33ms for 30 FPS → Headroom: 5ms (15% margin)


================================================================================
                        MODULE FILE MAPPING
================================================================================

┌─────────────────────────┬────────────────────────────────┬──────────────────────────┐
│ COMPONENT               │ SOURCE FILE                    │ KEY CLASS                │
├─────────────────────────┼────────────────────────────────┼──────────────────────────┤
│ Gimbal Hub              │ src/gimbal/hub.py              │ GimbalHub, FramePacket   │
│ Detection + CBAM        │ src/detection/detector.py      │ WeaponDetector, CBAMFusion│
│ Person-Weapon Linking   │ src/detection/linker.py        │ PersonWeaponLinker       │
│ Thermal Features        │ src/thermal/verifier.py        │ ThermalFeatureExtractor  │
│ Thermal LSTM            │ src/thermal/verifier.py        │ ThermalLSTM, ThermalVerifier│
│ ByteTrack               │ src/tracking/tracker.py        │ ByteTracker, Track       │
│ Gimbal Controller       │ src/gimbal/controller.py       │ GimbalController         │
│ GPS Transformer         │ src/gps/transformer.py         │ GPSTransformer           │
│ Alert System            │ src/evidence/alerts.py         │ AlertGenerator           │
│ Evidence Collector      │ src/evidence/collector.py      │ EvidenceCollector        │
│ Main Pipeline           │ src/pipeline.py                │ WeaponDetectionPipeline  │
└─────────────────────────┴────────────────────────────────┴──────────────────────────┘
```

---

**Document Version**: v7.0  
**Last Updated**: January 9, 2026  
**Status**: ✅ Complete with Person Detection + Gimbal Hub Integration
