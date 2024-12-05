import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FrameEmbeddingGenerator:
    def __init__(self, model_name="resnet50", device=None):
        """
        Initialize the embedding generator with a pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model to use for embedding generation.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name)
        self.transform = self._define_transform()

    def _load_model(self, model_name):
        """
        Load a pre-trained model and modify it to output embeddings.

        Args:
            model_name (str): Name of the pre-trained model.

        Returns:
            torch.nn.Module: Modified model.
        """
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Identity()  # Remove classification layer
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.eval()
        model.to(self.device)
        return model

    def _define_transform(self):
        """
        Define preprocessing transformations for input frames.

        Returns:
            torchvision.transforms.Compose: Transformation pipeline.
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the frame
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

    def generate_embedding(self, frame):
        """
        Generate an embedding for a given frame.

        Args:
            frame (PIL.Image or np.ndarray): Input frame in RGB format.

        Returns:
            np.ndarray: Generated embedding vector.
        """
        try:
            # Ensure the frame is a valid type
            if isinstance(frame, np.ndarray):
                # print(f"[DEBUG] Converting numpy.ndarray to PIL.Image.")
                frame = Image.fromarray(frame)  # Convert NumPy array to PIL Image
            elif not isinstance(frame, Image.Image):
                raise ValueError(f"Unexpected frame type: {type(frame)}. Expected PIL.Image or np.ndarray.")

            # Preprocess the frame
            frame_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            # print(f"[DEBUG] Frame tensor shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}")

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(frame_tensor)

            # Ensure the embedding is a tensor
            if not isinstance(embedding, torch.Tensor):
                raise ValueError(f"Model output is not a tensor: {type(embedding)}")

            # Convert to NumPy and flatten
            embedding_np = embedding.cpu().numpy().flatten()
            # print(f"[DEBUG] Generated embedding shape: {embedding_np.shape}")
            return embedding_np

        except Exception as e:
            print(f"[ERROR] Failed to generate embedding: {e}")
            return None
