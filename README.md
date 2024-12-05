# DeepfakeRestorer

DeepfakeRestorer is an advanced AI-driven project aimed at identifying and reversing deepfake modifications in video frames. It uses embeddings and a vector database to restore the original face in deepfaked frames, effectively performing a reverse deepfake mechanism.

---

## Key Features
- **Deepfake Detection**: Identifies deepfaked frames in input videos using AI-based models.
- **Frame Restoration**: Replaces detected deepfaked faces with the original faces using embeddings stored in a vector database.
- **Vector Database Integration**: Uses Qdrant for efficient embedding storage and retrieval.
- **Parallel Processing**: Processes video frames in parallel to improve performance and speed.
- **Dynamic Video Handling**: Supports random sampling of frames for quick fake detection and detailed processing.

---

## Technologies Used
- **Deep Learning Framework**: PyTorch
- **Face Detection**: MTCNN
- **Vector Database**: Qdrant
- **Pre-trained Models**: ResNet, CViT
- **Video Processing**: OpenCV and Decord

---

## How It Works
1. **Frame Sampling**: Extracts a few frames from the video to determine if it is deepfaked.
2. **Deepfake Classification**: Uses AI models to classify frames as real or fake.
3. **Original Frame Retrieval**: If fake frames are detected, retrieves the original corresponding frames from the vector database.
4. **Face Replacement**: Replaces the deepfaked faces with the original faces.
5. **Video Reconstruction**: Saves the processed frames as an updated video.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/deepakb2019/DeepfakeReversal.git
   cd DeepfakeRestorer
2. Install dependencies:
    using requirements.txt
    ```bash
    pip install -r requirements.txt
3. Start Qdrant locally or connect to a remote instance:
    ```bash
    docker run -p 6333:6333 -d qdrant/qdrant
## Usage

### 1. Store Original Frames in Vector Database
   To store embeddings of original video frames:
```bash
python store_embeddings.py
```
### 2. Process a Video
To detect and restore deepfaked frames:
```bash
python process_video.py --i data/fake/id1_id6_0009.mp4 --o data/updated_video.mp4
```
### `store_embeddings.py`

To store embeddings of original video frames, use the following command:

```bash
python store_embeddings.py --video-folder <path_to_videos> --save-frames-folder <path_to_save_frames> --batch-size <batch_size>
```
#### Options:
- `--i` (str): Path to the folder containing original videos. Default: `data/original_videos`.
- `--0` (str): Path to save extracted frames. Default: `data/processed_frames`.


### New Updates

#### Added Features
- **Dynamic Frame Sampling**: Randomly sample frames to determine if a video is fake, saving time.
- **Face Restoration**: Replace deepfaked faces with original faces across all video frames.
- **Parallel Frame Processing**: Speed up video processing with multithreading.

#### Optimizations
- **Distance-Based Matching**: Improved efficiency for embedding retrieval.
- **Enhanced Face Replacement Blending**: Smoother integration of original faces into deepfaked frames.
- **Streamlined Command-Line Options**: Simplified usage and better user experience.
### Contributing
Feel free to fork this repository, make improvements, and submit pull requests. Contributions are always welcome!

### License
This project is licensed under the MIT License.



