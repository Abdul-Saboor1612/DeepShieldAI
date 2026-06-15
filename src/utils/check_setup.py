import torch
import cv2
import numpy as np
import pandas as pd

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)
print("Pandas Version:", pd.__version__)

print("\nSetup Successful!")