import cv2
import time
import numpy as np
import os
import psutil
import tensorflow as tf

# ----------------------
# TFLite Interpreter
# ----------------------
Interpreter = tf.lite.Interpreter

# ======================
# CONFIG
# ======================
MODEL_PATH = "/home/muon/Desktop/rescue_project/best(v5n)-fp16.tflite"
PT_MODEL_PATH = "/home/pi/Desktop/rescue_project/models/best(v5s).pt"  # opsiyonel: sadece boyut kontrolü

IMGSZ = 640
CONF_THRES = 0.35
IOU_THRES = 0.45
DRAW = True

CLASS_NAMES = ["fallen", "lying", "sitting", "standing"]

# CPU/RAM her kaç frame'de bir ölçülsün
SYS_EVERY_N_FRAME = 10

# Kamera ayarı
CAM_INDEX = 4
USE_V4L2 = True  # Raspberry Pi/Linux için genelde daha stabil

# ======================
# GLOBAL STATS
# ======================
SCRIPT_START_TIME = time.time()

TOTAL_INFER_TIME = 0.0
INFER_COUNT = 0

TOTAL_CPU = 0.0
TOTAL_RAM = 0.0
SYS_SAMPLE_COUNT = 0

frame_count = 0
prev = time.time()
fps = 0.0

# ======================
# MODEL FILE SIZES
# ======================
pt_size_mb = os.path.getsize(PT_MODEL_PATH) / (1024 * 1024) if os.path.exists(PT_MODEL_PATH) else -1
tflite_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024) if os.path.exists(MODEL_PATH) else -1

# ======================
# MODEL LOAD
# ======================
t_model_load_start = time.time()
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
interpreter.allocate_tensors()
MODEL_LOAD_TIME = time.time() - t_model_load_start

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inp_idx = input_details[0]["index"]
out_idx = output_details[0]["index"]

inp_dtype = input_details[0]["dtype"]
inp = np.empty((1, IMGSZ, IMGSZ, 3), dtype=inp_dtype)

# ======================
# CAMERA
# ======================
if USE_V4L2:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(CAM_INDEX, 200)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMGSZ)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGSZ)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(
        f"Kamera açılamadı. CAM_INDEX={CAM_INDEX}. "
        "ls -l /dev/video* ile kontrol et ve index'i 0/1/2/3 diye dene."
    )

# ======================
# STATE (ekran için)
# ======================
last_boxes, last_scores, last_classes = [], [], []
last_best_cls, last_best_score = None, None
last_cpu_usage = 0.0
last_ram_usage = 0.0

print("✅ YOLO TFLite | Her karede inference (q/ESC ile çık)")

# ======================
# NMS
# ======================
def nms_xywh(boxes, scores, iou_thres):
    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRES,
        iou_thres
    )
    return idxs.flatten() if len(idxs) > 0 else []

# ======================
# MAIN LOOP
# ======================
try:
    while True:
        ret, frame = cap.read()
        if (not ret) or (frame is None):
            print("⚠️ cap.read() frame döndürmedi (ret=False). Çıkılıyor...")
            break

        h0, w0 = frame.shape[:2]
        frame_count += 1

        # ----------------------
        # PREPROCESS
        # ----------------------
        resized = cv2.resize(frame, (IMGSZ, IMGSZ))

        if inp_dtype == np.float32:
            inp[0] = resized.astype(np.float32) / 255.0
        elif inp_dtype == np.float16:
            inp[0] = resized.astype(np.float16) / np.float16(255.0)
        else:
            inp[0] = resized.astype(inp_dtype)

        interpreter.set_tensor(inp_idx, inp)

        # ----------------------
        # INFERENCE
        # ----------------------
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()

        TOTAL_INFER_TIME += (t1 - t0)
        INFER_COUNT += 1

        preds = interpreter.get_tensor(out_idx)[0]

        # ----------------------
        # POSTPROCESS (YOLOv5 TFLite: cx,cy,w,h,obj,cls...)
        # ----------------------
        boxes, scores, classes = [], [], []
        best_score = -1.0
        best_cls = None

        coords_normalized = np.max(preds[:50, :4]) <= 1.5

        for p in preds:
            cx, cy, bw, bh = p[:4]
            obj = p[4]
            cls_scores = p[5:5 + len(CLASS_NAMES)]
            cls_id = int(np.argmax(cls_scores))
            conf = float(obj * cls_scores[cls_id])

            if conf < CONF_THRES:
                continue

            if coords_normalized:
                cx *= IMGSZ
                cy *= IMGSZ
                bw *= IMGSZ
                bh *= IMGSZ

            x1 = int((cx - bw / 2) * (w0 / IMGSZ))
            y1 = int((cy - bh / 2) * (h0 / IMGSZ))
            x2 = int((cx + bw / 2) * (w0 / IMGSZ))
            y2 = int((cy + bh / 2) * (h0 / IMGSZ))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)

            ww, hh = x2 - x1, y2 - y1
            if ww <= 0 or hh <= 0:
                continue

            boxes.append([x1, y1, ww, hh])
            scores.append(conf)
            classes.append(cls_id)

        if boxes:
            idxs = nms_xywh(np.array(boxes), np.array(scores), IOU_THRES)
            last_boxes, last_scores, last_classes = [], [], []

            for i in idxs:
                last_boxes.append(boxes[i])
                last_scores.append(scores[i])
                last_classes.append(classes[i])

                if scores[i] > best_score:
                    best_score = scores[i]
                    best_cls = classes[i]

            last_best_cls = best_cls
            last_best_score = best_score
        else:
            last_boxes, last_scores, last_classes = [], [], []
            last_best_cls, last_best_score = None, None

        # ----------------------
        # SYSTEM USAGE (N frame'de bir)
        # ----------------------
        if frame_count % SYS_EVERY_N_FRAME == 0:
            last_cpu_usage = psutil.cpu_percent(interval=None)
            last_ram_usage = psutil.virtual_memory().percent

            TOTAL_CPU += last_cpu_usage
            TOTAL_RAM += last_ram_usage
            SYS_SAMPLE_COUNT += 1

        # ----------------------
        # DRAW
        # ----------------------
        if DRAW:
            for i in range(len(last_boxes)):
                x, y, w, h = last_boxes[i]
                cls_id = last_classes[i]
                score = last_scores[i]
                name = CLASS_NAMES[cls_id]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}:{score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ----------------------
        # FPS
        # ----------------------
        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # ----------------------
        # OVERLAY
        # ----------------------
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"CPU: {last_cpu_usage:.1f}%  RAM: {last_ram_usage:.1f}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if last_best_cls is not None and last_best_score is not None:
            best_name = CLASS_NAMES[last_best_cls]
            cv2.putText(frame, f"Pred: {best_name} ({last_best_score:.2f})",
                        (10, h0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO TFLite Profiling", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except Exception as e:
    print(f"\n🛑 Hata yakalandı: {type(e).__name__}: {e}")

finally:
    # Kaynakları kapat
    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # ======================
    # FINAL REPORT
    # ======================
    SCRIPT_TOTAL_TIME = time.time() - SCRIPT_START_TIME
    AVG_INFER_TIME = TOTAL_INFER_TIME / max(1, INFER_COUNT)

    AVG_CPU = TOTAL_CPU / max(1, SYS_SAMPLE_COUNT)
    AVG_RAM = TOTAL_RAM / max(1, SYS_SAMPLE_COUNT)

    AVG_FPS_OVERALL = frame_count / max(1e-9, SCRIPT_TOTAL_TIME)
    YOLO_FPS = INFER_COUNT / max(1e-9, SCRIPT_TOTAL_TIME)

    print("\n====== 📊 PROFILING REPORT ======")
    print(f"Model Load Time        : {MODEL_LOAD_TIME:.3f} s")
    print(f"Total Script Run Time  : {SCRIPT_TOTAL_TIME:.2f} s")
    print(f"Total Frames           : {frame_count}")
    print(f"Avg FPS (overall)      : {AVG_FPS_OVERALL:.2f}")
    print(f"Infer/sec (YOLO FPS)   : {YOLO_FPS:.2f}")
    print(f"Infer Count            : {INFER_COUNT}")
    print(f"Avg Inference Time     : {AVG_INFER_TIME*1000:.2f} ms")
    print(f"Avg CPU Usage          : {AVG_CPU:.2f} %")
    print(f"Avg RAM Usage          : {AVG_RAM:.2f} %")
    print(f"PT Model Size          : {pt_size_mb:.2f} MB")
    print(f"TFLite Model Size      : {tflite_size_mb:.2f} MB")
    print("=================================")