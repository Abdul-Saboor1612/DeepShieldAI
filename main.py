# main.py

# import sys

# from src.inference.model_loader import (
#     load_model
# )

# from src.inference.inference_router import (
#     route_inference
# )

# MODEL_PATH = (
#     "models/exp002_best_model.pth"
# )


# def main():

#     if len(sys.argv) != 2:

#         print(
#             "Usage: python main.py <file>"
#         )

#         return

#     file_path = sys.argv[1]

#     print("Loading model...")

#     model, device = load_model(
#         MODEL_PATH
#     )

#     result = route_inference(
#         file_path,
#         model,
#         device
#     )

#     print("\nRESULT")
#     print("=" * 40)

#     for key, value in result.items():

#         print(
#             f"{key}: {value}"
#         )


# if __name__ == "__main__":
#     main()

#temporary
from fastapi import FastAPI

from src.api.routes import (
    router
)

app = FastAPI(
    title="DeepFakeGuard"
)

app.include_router(
    router
)