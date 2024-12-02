from model.pred_func import *
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
import os
from tqdm import tqdm

# Qdrant setup
COLLECTION_NAME = "original_video_frames_1024"
VECTOR_SIZE = 1024  # Assuming face embedding size from Dlib
qdrant_client = QdrantClient("http://localhost:6333")

if COLLECTION_NAME not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance="Cosine"),
    )
else:
    print(f"[INFO] Collection '{COLLECTION_NAME}' already exists with size 1024.")

def generate_embedding_from_frame(frame):
    """
    Generates embeddings from the detected face(s) in a frame.

    Args:
        frame (np.ndarray): Input frame in RGB format.

    Returns:
        list: List of embedding vectors of the detected faces.
    """
    face_embeddings = []
    print("[DEBUG] Generating embeddings for a frame...")

    # Detect faces in the frame
    faces, count = face_rec([frame])
    print(f"[DEBUG] Detected {count} faces in the frame.")

    if count > 0:
        preprocessed_faces = preprocess_frame(faces)  # Preprocess detected faces
        for face_tensor in preprocessed_faces:
            try:
                # Extract embeddings using the model's extract_features method
                face_tensor = face_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
                embedding = model.extract_features(face_tensor)  # Extract intermediate features

                # Ensure `embedding` is converted to a NumPy array
                if isinstance(embedding, torch.Tensor):
                    embedding_np = embedding.cpu().detach().numpy().flatten()
                    face_embeddings.append(embedding_np)
                else:
                    print(f"[ERROR] Embedding extraction returned a non-tensor object: {type(embedding)}")
            except Exception as e:
                print(f"[ERROR] Failed to generate embedding for a face: {e}")
    else:
        print(f"[WARN] No embeddings generated for this frame due to face detection failure.")

    return face_embeddings  # Return a list of NumPy arrays
 # Always return a list of embeddings

def store_embeddings_in_qdrant(video_folder):
    """
    Extracts frames from videos, generates embeddings, and stores them in Qdrant.

    Args:
        video_folder (str): Path to the folder containing original videos.
    """
    video_files = [f for f in os.listdir(video_folder) if is_video(os.path.join(video_folder, f))]
    point_id = 0

    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(video_folder, video_file)
        print(f"[INFO] Processing video: {video_file}")

        # Extract frames
        frames = extract_frames(video_path, frames_nums=15)

        for frame_idx, frame in enumerate(frames):
            print(f"[INFO] Processing frame {frame_idx + 1}/{len(frames)}")
            embeddings = generate_embedding_from_frame(frame)

            if not embeddings:
                print(f"[WARN] No embeddings generated for frame {frame_idx + 1}. Skipping...")
                continue

            for embedding in embeddings:
                try:
                    if isinstance(embedding, np.ndarray):
                        qdrant_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[
                                {
                                    "id": point_id,
                                    "vector": embedding.tolist(), 
                                    "payload": {
                                        "video_name": video_file,
                                        "frame_index": frame_idx,
                                    },
                                }
                            ],
                        )
                        point_id += 1
                        print(f"[INFO] Successfully stored embedding for frame {frame_idx + 1}.")
                    else:
                        print(f"[ERROR] Embedding is not a numpy array: {type(embedding)}")
                except Exception as e:
                    print(f"[ERROR] Failed to store embedding in Qdrant: {e}")

        print(f"[INFO] Finished processing video: {video_file}")


if __name__ == "__main__":
    video_folder_path = "data/fake"  # Path to folder containing original videos
    try:
        print("[INFO] Starting embedding storage process...")
        # Load the model
        model = load_cvit("cvit_deepfake_detection_ep_50.pth", fp16=False)
        store_embeddings_in_qdrant(video_folder_path)
        print("[INFO] All embeddings stored successfully.")
    except Exception as e:
        print(f"[ERROR] An error occurred during the embedding process: {e}")