import base64
import os
import cv2
from model.full_frame_emmbedings import FrameEmbeddingGenerator
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from qdrant_client import QdrantClient
from model.pred_func import extract_frames, preprocess_frame, pred_vid, real_or_fake
from PIL import Image
from torchvision import transforms
from model.pred_func import *
from skimage.exposure import match_histograms
from facenet_pytorch import MTCNN
import argparse
from time import perf_counter
from datetime import datetime


COLLECTION_NAME = "original_video_frames_2048"
VECTOR_SIZE = 2048
qdrant_client = QdrantClient("http://localhost:6333")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False, device=device)

def search_original_frame(embedding):
    """Search for the closest matching embedding in Qdrant."""
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=1
    )
    if search_result:
        payload = search_result[0].payload
        distance = search_result[0].score
        return payload, distance
    return None, None


def process_frame(frame, frame_idx, model, threshold=0.8):
    """Process a single frame: detect, classify, and replace if deepfaked."""
    # print(f"[INFO] Processing frame {frame_idx + 1}...")
    embeddings = FrameEmbeddingGenerator()
    df = df_face_by_frame(frame, num_frames=1)
    
    if len(df) >= 1:
        prediction, confidence = pred_vid(df, model)
        label = real_or_fake(prediction)
    else:
        # print("[WARN] No faces detected in the frame. Defaulting to REAL.")
        prediction, confidence = 0, 0.5 
        label = "REAL"

    # print("label",label)

    if label == "REAL":
        # print(f"[INFO] Frame {frame_idx + 1}: REAL.")
        return frame

    print(f"[INFO] Frame {frame_idx + 1}: FAKE with confidence {confidence}. Searching for original frame...")
    embedding = embeddings.generate_embedding(frame)
    original_frame_info, distance  = search_original_frame(embedding)
    # print(f"\n\n\n[DEBUG] Retrieved Distance: {distance}")

    if original_frame_info:
        original_frame_data = original_frame_info.get("frame_data")
        if original_frame_data:
            original_frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(original_frame_data), np.uint8),
                cv2.IMREAD_COLOR
            )
            original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            updated_frame = replace_face_in_frame(frame, original_frame_rgb)
            return updated_frame
        else:
            print(f"[WARN] Original frame data missing for frame {frame_idx + 1}.")
    else:
        print(f"[WARN] No matching original frame found for frame {frame_idx + 1}.")

    return frame


def replace_face_in_frame(deepfake_frame, original_frame):
    """Replace the deepfaked face with the original face."""
    df_boxes, _ = mtcnn.detect(deepfake_frame)
    orig_boxes, _ = mtcnn.detect(original_frame)

    if df_boxes is not None and orig_boxes is not None:
        df_box = df_boxes[0]
        orig_box = orig_boxes[0]

        df_face = deepfake_frame[int(df_box[1]):int(df_box[3]), int(df_box[0]):int(df_box[2])]
        orig_face = original_frame[int(orig_box[1]):int(orig_box[3]), int(orig_box[0]):int(orig_box[2])]

        orig_face_resized = cv2.resize(orig_face, (df_face.shape[1], df_face.shape[0]))

        deepfake_frame[int(df_box[1]):int(df_box[3]), int(df_box[0]):int(df_box[2])] = orig_face_resized

    return deepfake_frame


def save_video(frames, output_path, fps=30):
    """
    Save frames as a video.
    
    Args:
        frames (list of np.ndarray): List of processed frames.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    if not frames:
        print("[WARN] No frames to save!")
        return
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    # print(f"[INFO] Video saved to: {output_path}")


def process_video(video_path, output_path, model):
    """Process a video to detect and replace deepfaked frames."""
    frames = extract_frames(video_path, frames_nums=200)
    updated_frames = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for frame_idx, frame in enumerate(frames):
            futures.append(executor.submit(process_frame, frame, frame_idx, model))

        for future in tqdm(futures, desc="Processing Frames"):
            updated_frames.append(future.result())

    save_video(updated_frames, output_path, fps=30)
    # print(f"[INFO] Updated video saved to {output_path}")


def gen_parser():
    """Generate a command-line parser for user input."""
    parser = argparse.ArgumentParser(description="Deepfake detection and face replacement.")
    parser.add_argument(
        "--i",
        type=str,
        required=True,
        help="Path to the input video file to be processed."
    )
    parser.add_argument(
        "--o",
        type=str,
        required=True,
        help="Path to save the processed video with replaced faces."
    )
    parser.add_argument(
        "--w",
        type=str,
        default="cvit_deepfake_detection_ep_50.pth",
        help="Path to the pre-trained CViT model weights."
    )
    return parser.parse_args()

def main():
    start_time = perf_counter()

    # Parse arguments
    args = gen_parser()

    # Load the model
    print("[INFO] Loading model...")
    model = load_cvit(args.w, fp16=False)

    # Process the video
    print(f"[INFO] Processing video: {args.i}")
    process_video(args.i, args.o, model)

    # Save the output video
    print(f"[INFO] Processed video saved to: {args.o}")

    end_time = perf_counter()
    print(f"\n\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    # video_path = "data/fake/id1_id6_0009.mp4"
    # output_path = "data/updated_video_1.mp4"

    # model = load_cvit("cvit_deepfake_detection_ep_50.pth", fp16=False)

    # process_video(video_path, output_path, model)
     main()
