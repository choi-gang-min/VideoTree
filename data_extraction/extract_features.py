# from PIL import Image
# import requests
# import torch
# import transformers
# from pathlib import Path
# import json
# import pickle
# from tqdm import tqdm
# import os
# from pprint import pprint
# import pdb

# import numpy as np
# from kmeans_pytorch import kmeans
# from sklearn.cluster import KMeans
# from transformers import AutoModel, AutoConfig
# from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode



# def save_image_features(img_feats, name_ids, save_folder):
#     """
#     Save image features to a .pt file in a specified folder.

#     Args:
#     - img_feats (torch.Tensor): Tensor containing image features
#     - name_ids (str): Identifier to include in the filename
#     - save_folder (str): Path to the folder where the file should be saved

#     Returns:
#     - None
#     """
#     filename = f"{name_ids}.pt"  # Construct filename with name_ids
#     filepath = os.path.join(save_folder, filename)
#     torch.save(img_feats, filepath)


# def load_json(fn):
#     with open(fn, 'r') as f:
#         data = json.load(f)
#     return data

# def save_json(data, fn, indent=4):
#     with open(fn, 'w') as f:
#         json.dump(data, f, indent=indent)



# # image_path = "CLIP.png"
# model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
# image_size = 224

# processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


# def clip_es():
#     device = "cuda" if torch.cuda.is_available() else "cpu" 

#     model = AutoModel.from_pretrained(
#     model_name_or_path, 
#     torch_dtype=torch.float16,
#     trust_remote_code=True).to('cuda').eval()
#     resume = True

#     base_path = Path('/data/dataset/intentqa_videotree/frames/')
#     save_folder = '/data/dataset/intentqa_videotree/features'
#     all_data = []

#     with open('/data/dataset/egoschema/subset_answers.json', 'r') as file:
#         json_data = json.load(file)    
#     subset_names_list = list(json_data.keys())

#     example_path_list = list(base_path.iterdir())

#     pbar = tqdm(total=len(example_path_list))

#     i = 0 
#     max = 50

#     for example_path in example_path_list:

#         # for subset videos, comment out for fullset
#         if example_path.name not in subset_names_list:
#             continue
#         # else:
#         #     print("example_path in subset")

#         image_paths = list(example_path.iterdir())
#         image_paths.sort(key=lambda x: int(x.stem))
#         img_feature_list = []
#         for image_path in image_paths:
#             image = Image.open(str(image_path))

#             input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

#             with torch.no_grad(), torch.cuda.amp.autocast():
#                 image_features = model.encode_image(input_pixels)
#                 img_feature_list.append(image_features)
#         img_feature_tensor = torch.stack(img_feature_list)
#         img_feats = img_feature_tensor.squeeze(1)

#         name_ids = example_path.name


#         save_image_features(img_feats, name_ids, save_folder)
#         pbar.update(1)


#     pbar.close()

# if __name__ == '__main__':
#     clip_es()

# ---------해당폴더에있는것들 그냥 다 feature시켜
# from PIL import Image
# import requests
# import torch
# import transformers
# from pathlib import Path
# import json
# import pickle
# from tqdm import tqdm
# import os
# from pprint import pprint
# import pdb

# import numpy as np
# from kmeans_pytorch import kmeans
# from sklearn.cluster import KMeans
# from transformers import AutoModel, AutoConfig
# from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode


# def save_image_features(img_feats, name_ids, save_folder):
#     """
#     Save image features to a .pt file in a specified folder.

#     Args:
#     - img_feats (torch.Tensor): Tensor containing image features
#     - name_ids (str): Identifier to include in the filename
#     - save_folder (str): Path to the folder where the file should be saved

#     Returns:
#     - None
#     """
#     filename = f"{name_ids}.pt"  # Construct filename with name_ids
#     filepath = os.path.join(save_folder, filename)
#     torch.save(img_feats, filepath)


# def load_json(fn):
#     with open(fn, 'r') as f:
#         data = json.load(f)
#     return data

# def save_json(data, fn, indent=4):
#     with open(fn, 'w') as f:
#         json.dump(data, f, indent=indent)


# # image_path = "CLIP.png"
# model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
# image_size = 224

# processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


# def clip_es():
#     device = "cuda" if torch.cuda.is_available() else "cpu" 

#     model = AutoModel.from_pretrained(
#     model_name_or_path, 
#     cache_dir = '/data/gangmin3552/huggingface/',
#     torch_dtype=torch.float16,
#     trust_remote_code=True).to('cuda').eval()
#     resume = True

#     base_path = Path('/data/dataset/intentqa_videotree/frames/')
#     save_folder = '/data/dataset/intentqa_videotree/features'
    
#     # save_folder가 없으면 생성
#     os.makedirs(save_folder, exist_ok=True)
    
#     all_data = []

#     example_path_list = list(base_path.iterdir())

#     pbar = tqdm(total=len(example_path_list))
    
#     # 에러 추적을 위한 리스트
#     error_folders = []
#     empty_folders = []

#     i = 0 
#     max = 50

#     for example_path in example_path_list:
#         try:
#             # 이미지 파일만 필터링 (jpg, png 등)
#             image_paths = [p for p in example_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
#             # 폴더가 비어있는 경우 체크
#             if not image_paths:
#                 empty_folders.append(example_path.name)
#                 pbar.update(1)
#                 continue
                
#             image_paths.sort(key=lambda x: int(x.stem))
#             img_feature_list = []
            
#             for image_path in image_paths:
#                 try:
#                     image = Image.open(str(image_path))

#                     input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

#                     with torch.no_grad(), torch.cuda.amp.autocast():
#                         image_features = model.encode_image(input_pixels)
#                         img_feature_list.append(image_features)
#                 except Exception as e:
#                     print(f"Error processing image {image_path}: {str(e)}")
#                     continue
            
#             # feature list가 비어있는 경우 체크
#             if not img_feature_list:
#                 error_folders.append((example_path.name, "No valid images processed"))
#                 pbar.update(1)
#                 continue
                
#             img_feature_tensor = torch.stack(img_feature_list)
#             img_feats = img_feature_tensor.squeeze(1)

#             name_ids = example_path.name

#             save_image_features(img_feats, name_ids, save_folder)
            
#         except Exception as e:
#             error_folders.append((example_path.name, str(e)))
#             print(f"Error processing folder {example_path.name}: {str(e)}")
            
#         pbar.update(1)

#     pbar.close()
    
#     # 결과 출력
#     print(f"\n처리 완료!")
#     print(f"총 폴더 수: {len(example_path_list)}")
#     print(f"빈 폴더 수: {len(empty_folders)}")
#     print(f"에러 폴더 수: {len(error_folders)}")
    
#     if empty_folders:
#         print(f"\n빈 폴더들: {empty_folders[:5]}...")  # 처음 5개만 출력
        
#     if error_folders:
#         print(f"\n에러 발생 폴더들:")
#         for folder, error in error_folders[:5]:  # 처음 5개만 출력
#             print(f"  - {folder}: {error}")
    
#     # 에러 로그 저장
#     if empty_folders or error_folders:
#         error_log = {
#             'empty_folders': empty_folders,
#             'error_folders': error_folders
#         }
#         save_json(error_log, os.path.join(save_folder, 'processing_errors.json'))


# if __name__ == '__main__':
#     clip_es()







#------------
from PIL import Image
import requests
import torch
import transformers
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import os
from pprint import pprint
import pdb

import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = AutoModel.from_pretrained(
    model_name_or_path, 
    cache_dir = '/data/gangmin3552/huggingface/',
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()
    resume = True

    base_path = Path('/data/dataset/intentqa_videotree/frames/')
    save_folder = '/data/dataset/intentqa_videotree/features_test'
    
    # save_folder가 없으면 생성
    os.makedirs(save_folder, exist_ok=True)
    
    all_data = []

    # JSON 읽는 부분 삭제
    # with open('/data/dataset/egoschema/subset_answers.json', 'r') as file:
    #     json_data = json.load(file)    
    # subset_names_list = list(json_data.keys())

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0 
    max = 50

    for example_path in example_path_list:

        # subset 필터링 부분 삭제
        # if example_path.name not in subset_names_list:
        #     continue

        image_paths = list(example_path.iterdir())
        image_paths.sort(key=lambda x: int(x.stem))
        img_feature_list = []
        for image_path in image_paths:
            image = Image.open(str(image_path))

            input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(input_pixels)
                img_feature_list.append(image_features)
        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1)

        name_ids = example_path.name

        save_image_features(img_feats, name_ids, save_folder)
        pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    clip_es()