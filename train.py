import os
import ssl
import torch
from pathlib import Path

# -------------------------------------------------------
# 1 SSL FIX and Imports
# -------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pytorch_lightning import seed_everything

# -------------------------------------------------------
# 2 Configuration
# -------------------------------------------------------
seed_everything(42)

#  to your FINAL PROCESSED dataset
root_dir = Path(r"D:\ECU_New_Patchcore") 

# -------------------------------------------------------
# 3 Dataset Configuration
# -------------------------------------------------------
dataset = Folder(
    name="ECU_PatchCore_Processed_Final",
    root=root_dir,
    normal_dir="train/good",      # D:\ECU_New_Patchcore\train\good
    abnormal_dir="test/pin_tilt", # D:\ECU_New_Patchcore\test\pin_tilt
    normal_split_ratio=0.8,     
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=1,
)

# -------------------------------------------------------
# 4 Model Definition 
# -------------------------------------------------------
try:
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        
        # Using all 3 layers
        layers=["layer1", "layer2", "layer3"], 
        
        # Hyper-sensitive 'k' in KNN 
        num_neighbors=5, 
        
        # Using 100% of features for max detail
        coreset_sampling_ratio=1.0, 
    )
    print(" Loaded pretrained backbone successfully.")
except Exception as e:
    print(f"Could not load pretrained backbone due to: {e}")
    print("Falling back to non-pretrained mode.")
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=False,
        layers=["layer1", "layer2", "layer3"],
        num_neighbors=5,
        coreset_sampling_ratio=1.0,
    )

# -------------------------------------------------------
# 5 Training Engine Setup
# -------------------------------------------------------
engine = Engine(  #it tells the model, the data, and the hardware when to start and stop working together.
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else "auto",
    max_epochs=5, 
    default_root_dir=Path("results_final_dataset"), # New results folder
)

# -------------------------------------------------------
# 6️⃣ Training & Testing Execution
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n Starting FINAL training on PROCESSED fixture dataset...\n")
    engine.fit(model=model, datamodule=dataset)

    print("\n Training Complete! Starting testing...\n")
    engine.test(model=model, datamodule=dataset)

    print("\n Training & Testing Complete!")
    print(" Results saved in the 'results_final_dataset' folder.")