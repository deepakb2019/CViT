# import cv2
# import os
# import torch
# from decord import VideoReader, cpu
# from decord import VideoWriter

# # Load your trained model (assuming it's already set up)
# from model.pred_func import pred_vid, preprocess_frame

# def detect_and_filter_frames(video_path, model, threshold=0.5):
#     # Extract frames from video
#     vr = VideoReader(video_path, ctx=cpu(0))
#     frame_indices = range(len(vr))
#     real_frames = []

#     # Process each frame
#     for idx in frame_indices:
#         frame = vr[idx].asnumpy()
#         preprocessed_frame = preprocess_frame([frame])  # Preprocess single frame

#         with torch.no_grad():
#             pred, score = pred_vid(preprocessed_frame, model)

#         # Add frame to real_frames if it’s classified as "Real"
#         if pred == 0 and score > threshold:  # Assuming 0 = REAL
#             real_frames.append(frame)

#     return real_frames

# def save_filtered_video(frames, output_path, fps=30):
#     height, width, _ = frames[0].shape

#     # Initialize the VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Write each frame to the video
#     for frame in frames:
#         out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     out.release()

#     print(f"Filtered video saved at: {output_path}")

# def process_video(input_video, output_video, model, fps=30):
#     print("Detecting fake frames...")
#     real_frames = detect_and_filter_frames(input_video, model)

#     if not real_frames:
#         print("No real frames detected. Exiting.")
#         return

#     print(f"Detected {len(real_frames)} real frames. Saving video...")
#     save_filtered_video(real_frames, output_video, fps)

# def play_video(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cv2.imshow('Filtered Video', frame)

#         # Press 'q' to exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import os
from decord import VideoReader, cpu
import torch
from model.pred_func import pred_vid, preprocess_frame, real_or_fake,load_data
from model.cvit import CViT
import numpy as np
from model.pred_func import *



# def detect_and_filter_frames(video_path, model, threshold=0.5):
#     # Extract frames from video
#     vr = VideoReader(video_path, ctx=cpu(0))
#     frame_indices = range(len(vr))
#     real_frames = []

#     for idx in frame_indices:
#         frame = vr[idx].asnumpy()

#         # Assume `preprocess_frame` and `pred_vid` are already defined
#         preprocessed_frame = preprocess_frame([frame])
#         with torch.no_grad():
#             pred, score = pred_vid(preprocessed_frame, model)

#         if pred == 0 and score > threshold:  # 0 = REAL
#             real_frames.append(frame)

#     return real_frames

# def detect_and_filter_frames(video_path, model):
#     """
#     Detects and filters fake frames from the video using the provided model.

#     Args:
#         video_path (str): Path to the video file.
#         model (torch.nn.Module): Trained model for fake frame detection.

#     Returns:
#         list: List of real frames from the video.
#     """
#     print("[DEBUG] Starting fake frame detection...")
#     real_frames = []
#     # vr = VideoReader(video_path, ctx=cpu(0))
#     df = df_face(video_path, num_frames=15) 
#     print(f"[DEBUG] Total frames in video: {len(df)}")
#     # print(f"[DEBUG] Total frames in video: {df}")
#     # # frames = vr[:2].asnumpy() 
#     # for i, frame in enumerate(df):
#     #     print(f"[DEBUG] Processing frame {i+1}/{len(vr)} with shape: {frame.shape}")
        
#     #     try:
#     #         # Preprocess the frame
#     #         preprocessed_frame = preprocess_frame(frame)
#     #         print(f"[DEBUG] Preprocessed frame shape: {preprocessed_frame.shape}")

#     #         # Make a prediction
#     #         pred, score = pred_vid(preprocessed_frame, model)
#     #         print(f"[DEBUG] Prediction: {pred}, Score: {score}")

#     #         # Append to real_frames if it's real
#     #         if real_or_fake(pred) == "REAL":
#     #             real_frames.append(frame.asnumpy())  # Save the original frame in NumPy format

#     #     except Exception as e:
#     #         print(f"[ERROR] Failed to process frame {i+1}: {e}")

#     # print("[DEBUG] Completed fake frame detection.")
#     return []
def detect_and_filter_frames(video_path, model):
    """
    Detects and filters fake frames from the video using the provided model.

    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Trained model for fake frame detection.

    Returns:
        list: List of real frames from the video.
    """
    print("[DEBUG] Starting fake frame detection...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"[DEBUG] Total frames in video: {total_frames}")
        # Extract and preprocess frames
        df = df_face(video_path, num_frames=total_frames)  # Existing frame extraction function
        print(f"[DEBUG] Number of frames preprocessed: {df.shape[0]}")

        real_frames = []

        for i in range(len(df)):
            print(f"[DEBUG] Processing frame {i+1}/{len(df)}")
            frame_tensor = df[i].unsqueeze(0)  # Add batch dimension
            print(f"[DEBUG] Frame tensor shape for prediction: {frame_tensor.shape}")

            # Make a prediction
            pred, score = pred_vid(frame_tensor, model)
            print(f"[DEBUG] Prediction: {real_or_fake(pred)}, Score: {score}")

            # Save only real frames
            if real_or_fake(pred) == "REAL":
                real_frames.append(df[i].permute(1, 2, 0).cpu().numpy())  # Convert back to NumPy for saving/display

    except Exception as e:
        print(f"[ERROR] Failed to process video: {e}")
        return []

    print("[DEBUG] Completed fake frame detection.")
    return real_frames


# def save_filtered_video(frames, output_path, fps=30):
#     if not frames:
#         print("No frames to save!")
#         return

#     # Get dimensions of the first frame
#     height, width, _ = frames[0].shape

#     # Initialize OpenCV VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     for frame in frames:
#         out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

#     out.release()
#     print(f"Filtered video saved to: {output_path}")
# def save_filtered_video(frames, output_path, fps=30):
#     """
#     Saves the filtered frames as a video.
    
#     Args:
#         frames (list): List of frames to save.
#         output_path (str): Path to save the output video.
#         fps (int): Frames per second for the output video.
#     """
#     if not frames:
#         print("No frames to save!")
#         return

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Get dimensions of the first frame
#     height, width, _ = frames[0].shape

#     # Initialize OpenCV VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#     print(f"[DEBUG] Creating VideoWriter with path: {output_path}, FPS: {fps}, Size: ({width}, {height})")
    
#     try:
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#         if not out.isOpened():
#             raise IOError("Failed to open VideoWriter. Check codec support or file path.")

#         for frame in frames:
#             out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

#         out.release()
#         print(f"Filtered video saved to: {output_path}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save video: {e}")
def save_filtered_video(frames, output_path, fps=30):
    """
    Saves the filtered frames as a video.
    
    Args:
        frames (list): List of frames to save.
        output_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    if not frames:
        print("[ERROR] No frames to save!")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get dimensions of the first frame
    height, width, channels = frames[0].shape
    assert channels == 3, "[ERROR] Frames must have 3 color channels (RGB)."

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        try:
            # Convert frame to BGR format for OpenCV
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # Rescale if normalized
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        except Exception as e:
            print(f"[ERROR] Failed to write frame {i + 1}: {e}")

    out.release()
    print(f"[INFO] Filtered video saved to: {output_path}")

def process_video(input_video, output_video, model, fps=30):
    print("Detecting fake frames...")
    real_frames = detect_and_filter_frames(input_video, model)

    print(f"Detected {len(real_frames)} real frames. Saving video...")
    save_filtered_video(real_frames, output_video, fps)


input_video = "data/combined_video.mp4"
output_video = "data/filtered_video.mp4"


model = load_cvit('cvit_deepfake_detection_ep_50.pth', 30)


# # Load your trained model
# model = torch.load("weight/cvit_deepfake_detection_ep_50.pth", map_location=torch.device("cpu"))
# model.eval()

# Process the video
process_video(input_video, output_video, model, fps=30)

def combine_videos(real_video_path, fake_video_path, output_path):
    # Open the real video
    real_cap = cv2.VideoCapture(real_video_path)
    fake_cap = cv2.VideoCapture(fake_video_path)

    # Get properties of the real video
    width = int(real_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(real_cap.get(cv2.CAP_PROP_FPS))

    # Initialize the video writer with the properties of the first video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Function to process and write frames from a video
    def process_and_write_frames(cap, source_name):
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break

            # Resize frame if dimensions don't match
            frame_height, frame_width, _ = frame.shape
            if frame_width != width or frame_height != height:
                print(f"[INFO] Resizing frames from {source_name} to match the first video.")
                frame = cv2.resize(frame, (width, height))

            out.write(frame)

    # Add frames from the real video
    print("Adding frames from the real video...")
    process_and_write_frames(real_cap, "real video")

    # Add frames from the fake video
    print("Adding frames from the fake video...")
    process_and_write_frames(fake_cap, "fake video")

    # Release resources
    real_cap.release()
    fake_cap.release()
    out.release()

    print(f"Combined video saved to: {output_path}")


# Example Usage
# combine_videos("sample__prediction_data/sample_1.mp4", "sample__prediction_data/aajsqyyjni.mp4", "data/combined_video.mp4")