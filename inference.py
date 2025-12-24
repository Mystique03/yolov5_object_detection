import cv2
import torch
import argparse
import os
from pathlib import Path


def load_model():
    """
    Loads pretrained YOLOv5 model using PyTorch Hub
    """
    model = torch.hub.load(
        'ultralytics/yolov5',
        'yolov5s',
        pretrained=True
    )
    model.conf = 0.15
    model.iou = 0.45
    model.eval()
    return model


def run_inference(source, save=True, show=True, output_dir="outputs"):
    """
    Runs YOLOv5 inference on image or video
    """
    model = load_model()
    os.makedirs(output_dir, exist_ok=True)

    source_path = Path(source)

    # ---------- IMAGE ----------
    if source_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(str(source_path))
        if img is None:
            raise ValueError("Could not read image")

        results = model(img)
        rendered_img = results.render()[0]

        if show:
            cv2.imshow("YOLOv5 Detection", rendered_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save:
            out_path = os.path.join(output_dir, source_path.name)
            cv2.imwrite(out_path, rendered_img)
            print(f"[INFO] Saved output to {out_path}")

    # ---------- VIDEO ----------
    else:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError("Could not open video")

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            rendered_frame = results.render()[0]

            if show:
                cv2.imshow("YOLOv5 Detection", rendered_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save:
                out_path = os.path.join(
                    output_dir, f"frame_{frame_id:05d}.jpg"
                )
                cv2.imwrite(out_path, rendered_frame)
                frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Saved frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Inference Script")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image or video file"
    )
    parser.add_argument(
        "--noshow",
        action="store_true",
        help="Disable OpenCV window display"
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Disable saving outputs"
    )

    args = parser.parse_args()

    run_inference(
        source=args.source,
        show=not args.noshow,
        save=not args.nosave
    )
