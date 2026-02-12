import torch
import torch.nn as nn
import timm
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# --- 1. Define Model Architecture (Same as before) ---
class LightweightStudent(nn.Module):
    def __init__(self, num_classes=1530):
        super().__init__()
        self.rgb_bb = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=0)
        self.ir_bb = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=0)
        with torch.no_grad():
            self.feat_dim = self.rgb_bb(torch.randn(1,3,288,144)).shape[-1]
        self.rgb_bn = nn.BatchNorm1d(self.feat_dim)
        self.ir_bn = nn.BatchNorm1d(self.feat_dim)
        self.projector = nn.Sequential(nn.Linear(self.feat_dim, 2048), nn.ReLU(), nn.Linear(2048, 2048))
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

    def forward(self, rgb=None, ir=None):
        out = {}
        if rgb is not None:
            f = self.rgb_bb(rgb)
            bn = self.rgb_bn(f)
            out['rgb'] = bn 
        if ir is not None:
            f = self.ir_bb(ir)
            bn = self.ir_bn(f)
            out['ir'] = bn
        return out

# --- 2. Setup Transform (Must match training) ---
transform_val = transforms.Compose([
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def validate_real():
    print("="*50)
    print("REAL AI ENGINEER VERIFICATION: VTMOT DATASET")
    print("="*50)

    # Paths
    base_dir = '/home/student/Toan/data/VT-MOT_ReID_Person_Only'
    
    # 1. Anchor: ID 1320001 (RGB)
    anchor_path = os.path.join(base_dir, 'bounding_box_train/1320001_c1s132_000001_00.jpg')
    # 2. Positive: ID 1320001 (IR) -> SAME PERSON, DIFFERENT MODALITY (Hardest Test!)
    pos_path = os.path.join(base_dir, 'ir_bounding_box_train/1320001_c1s132_000001_00.jpg')
    # 3. Negative: ID 1320002 (RGB) -> DIFFERENT PERSON
    neg_path = os.path.join(base_dir, 'bounding_box_train/1320002_c1s132_000001_00.jpg')

    print(f"[ANCHOR] {os.path.basename(anchor_path)} (RGB, ID 001)")
    print(f"[POS]    {os.path.basename(pos_path)}    (IR,  ID 001)")
    print(f"[NEG]    {os.path.basename(neg_path)}    (RGB, ID 002)")

    # Load Images
    def load_img(path):
        img = Image.open(path).convert('RGB')
        return transform_val(img).unsqueeze(0) # Batch size 1

    t_anchor = load_img(anchor_path)
    t_pos = load_img(pos_path)
    t_neg = load_img(neg_path)

    # Load Model
    model = LightweightStudent(num_classes=1530)
    state_dict = torch.load('/home/student/Toan/analysis_c2kd/student_vtmot_best.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Inference
    with torch.no_grad():
        # Anchor (RGB)
        feat_anchor = model(rgb=t_anchor)['rgb']
        # Positive (IR) -> Pass to IR branch!
        feat_pos = model(ir=t_pos)['ir']
        # Negative (RGB)
        feat_neg = model(rgb=t_neg)['rgb']

    # Normalize Embeddings
    feat_anchor = F.normalize(feat_anchor, dim=1)
    feat_pos = F.normalize(feat_pos, dim=1)
    feat_neg = F.normalize(feat_neg, dim=1)

    # Compute Distances (1 - Cosine Similarity) is standard for ReID
    # Or Euclidean. Let's use Euclidean distance for clarity.
    dist_pos = torch.norm(feat_anchor - feat_pos).item()
    dist_neg = torch.norm(feat_anchor - feat_neg).item()
    
    # Cosine Similarity (Higher is better)
    sim_pos = torch.mm(feat_anchor, feat_pos.t()).item()
    sim_neg = torch.mm(feat_anchor, feat_neg.t()).item()

    print("-" * 50)
    print(f"RESULTS (Cross-Modal Matching):")
    print(f"Similarity (Same Person, RGB vs IR):  {sim_pos:.4f}  (Should meet threshold > 0.4)")
    print(f"Similarity (Diff Person, RGB vs RGB): {sim_neg:.4f}  (Should be low)")
    print("-" * 50)
    
    if sim_pos > sim_neg:
        print("✅ SUCCESS: Model correctly matches Cross-Modal ID!")
        margin = sim_pos - sim_neg
        print(f"   Margin: +{margin:.4f}")
        if margin > 0.2:
            print("   Verdict: STRONG DISCRIMINATION (Excellent)")
        else:
            print("   Verdict: WEAK DISCRIMINATION (Needs Improvement)")
    else:
        print("❌ FAILURE: Model confused IDs.")

if __name__ == '__main__':
    validate_real()
