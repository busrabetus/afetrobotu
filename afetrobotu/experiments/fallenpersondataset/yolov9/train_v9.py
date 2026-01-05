from ultralytics import YOLO
import os

# ===============================
# DATASET YOLU
# ===============================
DATA_YAML = "dataset2/data.yaml"

if __name__ == '__main__':
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"{DATA_YAML} bulunamadı")

    # ===============================
    # MODEL 
    # ===============================
    model = YOLO("yolov9t.pt")  
    # ===============================
    # TRAIN
    # ===============================
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=-1,        
        device=0,        
        workers=0,       

        # ===== AUGMENTATION KAPALI =====
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,

        seed=42,
        deterministic=True,    
    )

    print("✅ Eğitim tamamlandı")
