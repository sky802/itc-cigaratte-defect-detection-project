# YOLOv8 Defect Detection - Machine Learning Backend Code
# This script handles training and inference using the Ultralytics YOLOv8 model
# It also includes label mapping for cigarette defect types and background person labeling.
# /opt/anaconda3/bin/python3 ~/Desktop/Cig-defect/trials.py

from ultralytics import YOLO
import cv2
import os
import zipfile
import shutil
import matplotlib.pyplot as plt



# ------------------------
# 0. Unzip Dataset (if running in notebook)
# ------------------------
def unzip_dataset():
    zip_paths = {
        "train_images": "train image.zip",
        "train_labels": "train labels.zip",
        "valid_images": "valid image.zip",
        "valid_labels": "valid labels .zip",
        "test_images": "test image.zip",
        "test_labels": "test labels.zip"
    }

    for name, path in zip_paths.items():
        extract_to = name.replace(" ", "_")
        if os.path.exists(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f"Extracted: {path} -> {extract_to}")

# ------------------------
# 1. Load Trained Model
# ------------------------
model = YOLO("/Users/shrayknat/Desktop/Cig-defect/best.pt")  # Replace with your custom-trained model path

# ------------------------
# 2. Class Label Mapping
# ------------------------
label_map = {
    0: "D shape",
    1: "Dented",
    2: "Person",
    3: "Tobacco coming out",
    4: "Wrinkled",
    5: "Crushed tips",
    
}

# ------------------------
# 3. Predict on Webcam or Image
# ------------------------
def run_live_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    accuracy_scores = []
    frame_counts = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.3)
        frame_accuracy = 0
        detections = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = label_map.get(cls_id, "Unknown")
                conf = float(box.conf[0])
                frame_accuracy += conf
                detections += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detections > 0:
            avg_accuracy = frame_accuracy / detections
            accuracy_scores.append(avg_accuracy)
            frame_counts.append(frame_id)

            # Plot real-time accuracy graph
            plt.clf()
            plt.plot(frame_counts, accuracy_scores, label="Avg Confidence")
            plt.xlabel("Frame")
            plt.ylabel("Confidence")
            plt.title("Live Detection Confidence Graph")
            plt.ylim(0, 1)
            plt.legend()
            plt.pause(0.001)

        frame_id += 1
        cv2.imshow("Live YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

def run_image_detection(image_path):
    results = model.predict(source=image_path, conf=0.3)
    results[0].show()  # Show image with bounding boxes
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = label_map.get(cls_id, "Unknown")
        print(f"Detected: {label}")
    return results

# ------------------------
# 4. Train Model (Optional)
# ------------------------
def train_model():

    #unzip_dataset()  # Make sure data is extracted
    model = YOLO("yolov8n.pt")  # Start from a base model (nano/small/medium)
    data_yaml = "/Users/shrayknat/Desktop/Cig-defect/data3.yaml"

    results = model.train(
        data=data_yaml,  # <- Pass the path as a string
        epochs=50,
        imgsz=640,
        batch=16
    )
    # Copy the best model to project root for easier access
    best_model_path = results.save_dir / "weights" / "best.pt"
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, "best.pt")
        print(f"Saved best model to: best.pt")

# ------------------------
# 5. Evaluate Model (Optional)
# ------------------------
def evaluate_model():
    metrics = model.val()
    print("\n--- Model Evaluation Metrics ---")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print("--------------------------------")

# ------------------------
# 6. Export Model (Optional)
# ------------------------
def export_model():
    model.export(format="onnx")  # Formats: onnx, torchscript, coreml, etc.

# ------------------------
# 7. Main - Choose Your Mode
# ------------------------
if __name__ == "__main__":
    print("Running YOLO Defect Detection Script...")
    mode = input("Enter mode (live, image, train, eval, export): ").strip().lower()

    if mode == "live":
        print("Running live webcam detection...")
        run_live_detection()
    elif mode == "image":
        print("Running image detection...")
        run_image_detection("/Users/shrayknat/Desktop/Cig-defect/test.jpg")  # Replace with your test image path
    elif mode == "train":
        print("Starting training...")
        train_model()
    elif mode == "eval":
        print("Evaluating model...")
        evaluate_model()
    elif mode == "export":
        print("Exporting model...")
        export_model()
    else:
        print("Invalid mode selected.")
