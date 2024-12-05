import base64
from model.full_frame_emmbedings import FrameEmbeddingGenerator
from model.pred_func import *
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Qdrant setup
COLLECTION_NAME = "original_video_frames_2048"
VECTOR_SIZE = 2048 
qdrant_client = QdrantClient("http://localhost:6333")

if COLLECTION_NAME not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=2048, distance="Cosine"),
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
                face_tensor = face_tensor.unsqueeze(0).to(device)  
                embedding = model.extract_features(face_tensor) 

                # `embedding` is converted to a NumPy array
                if isinstance(embedding, torch.Tensor):
                    embedding_np = embedding.cpu().detach().numpy().flatten()
                    face_embeddings.append(embedding_np)
                else:
                    print(f"[ERROR] Embedding extraction returned a non-tensor object: {type(embedding)}")
            except Exception as e:
                print(f"[ERROR] Failed to generate embedding for a face: {e}")
    else:
        print(f"[WARN] No embeddings generated for this frame due to face detection failure.")

    return face_embeddings 


def store_embeddings_in_qdrant(video_folder, save_frames_folder=None, visualize=False, batch_size=10):
    """
    Extracts frames from videos, generates embeddings, and stores them in Qdrant in batches.

    Args:
        video_folder (str): Path to the folder containing original videos.
        save_frames_folder (str): Path to save processed frames for visualization.
        visualize (bool): Whether to visualize frames during processing.
        batch_size (int): Number of embeddings to process in a single batch.
    """

    video_files = [f for f in os.listdir(video_folder) if is_video(os.path.join(video_folder, f))]
    point_id = 201

    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(video_folder, video_file)
        print(f"[INFO] Processing video: {video_file}")

        # Extract frames
        frames = extract_frames(video_path, frames_nums=100)
        embedding_generator = FrameEmbeddingGenerator(model_name="resnet50")

        batch_embeddings = [] 
        batch_payloads = []   

        for frame_idx, frame in enumerate(frames):
            print(f"[INFO] Processing frame {frame_idx + 1}/{len(frames)}")

            # Generate embedding
            embedding = embedding_generator.generate_embedding(frame)
            if embedding is None or len(embedding) == 0:
                print(f"[WARN] No embedding generated for frame {frame_idx + 1}. Skipping...")
                continue

            encoded_frame = encode_frame_to_base64(frame)
            # Append embedding and payload to batch
            batch_embeddings.append(embedding.tolist())
            batch_payloads.append({
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": {
                    "video_name": video_file,
                    "frame_index": frame_idx,
                    "frame_data": encoded_frame
                },
            })
            point_id += 1

            # Process batch when size limit is reached
            if len(batch_payloads) >= batch_size:
                _process_batch(batch_payloads)
                batch_embeddings.clear()
                batch_payloads.clear()

        # Process any remaining embeddings in the batch
        if batch_payloads:
            _process_batch(batch_payloads)

        print(f"[INFO] Finished processing video: {video_file}")

def encode_frame_to_base64(frame):
    """
    Encodes a frame as a base64 string.

    Args:
        frame (np.ndarray): Frame in RGB format.

    Returns:
        str: Base64 encoded string of the frame.
    """
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

def _process_batch(batch_payloads):
    """
    Helper function to process and store a batch of embeddings in Qdrant.

    Args:
        batch_payloads (list): List of payload dictionaries to store in Qdrant.
    """
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_payloads
        )
        print(f"[INFO] Successfully stored batch of {len(batch_payloads)} embeddings in Qdrant.")
    except Exception as e:
        print(f"[ERROR] Failed to store batch in Qdrant: {e}")


if __name__ == "__main__":
    video_folder_path = "data/real"
    try:
        print("[INFO] Starting embedding storage process...")
        # Load the model
        model = load_cvit("cvit_deepfake_detection_ep_50.pth", fp16=False)
        save_frames_folder = "processed_frames"
        store_embeddings_in_qdrant(
            video_folder_path, 
            save_frames_folder=save_frames_folder, 
            visualize=True
        )
        print("[INFO] All embeddings stored successfully.")
    except Exception as e:
        print(f"[ERROR] An error occurred during the embedding process: {e}")