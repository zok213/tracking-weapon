from ultralytics.nn.modules import GatedSpatialFusion, GatedSpatialFusion_V3
print(f"V2: {GatedSpatialFusion}")
print(f"V3: {GatedSpatialFusion_V3}")
print(f"Same? {GatedSpatialFusion is GatedSpatialFusion_V3}")
