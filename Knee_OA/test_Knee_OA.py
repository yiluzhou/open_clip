# Batch test for Direct Classification benchmark 4 (cropped square & data augmentation images, normalizing with mean and std)
import torch, os, open_clip
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import ToTensor
import albumentations as A


class AlbumentationsTransform2:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)  # Convert to numpy array
        img = self.transform(image=img)['image']
        return Image.fromarray(img)  # Convert back to PIL image

# Define the CLAHE transformation
clahe = A.CLAHE(p=1.0, clip_limit=6.0, tile_grid_size=(12, 12))
CLAHE = AlbumentationsTransform2(clahe) # normalizing using Adaptive Histogram Equalization (CLAHE)

caption_map = {
    "healthy": "healthy normal",
    "doubtful": "doubtful osteoarthritis",
    "minimal": "minimal osteoarthritis",
    "moderate": "moderate osteoarthritis",
    "severe": "severe osteoarthritis"
}

classification = list(caption_map.values())

def get_ground_truth(filename, caption_map):
    for keyword, ground_truth in caption_map.items():
        if keyword in filename:
            return ground_truth
    return None

def process_single_epoch(pretrained_model, sentences):
    img_dir = '/mnt/g/Datasets/Knee_OA/Original/test/images/'
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=pretrained_model,
    )
    pt_dir = os.path.dirname(pretrained_model)
    epoch_number = os.path.basename(pretrained_model).split('epoch_')[1].split('.pt')[0]
    df_all = pd.DataFrame()

    # Prepare the tokenizer for sentences
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
    text = tokenizer(sentences)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    def read_single_image(img_dir, img_filename, sentences, text_features):
        #get the ground truth value
        image_path = os.path.join(img_dir, img_filename)
        ground_truth = get_ground_truth(img_filename, caption_map)
        
        # load a sample image
        image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            text_probs = text_probs.cpu().tolist()[0]

        # Construct the dictionary
        class_predict_dict = dict(zip(sentences, text_probs))
        # Extract the key with the largest value
        class_predict = max(class_predict_dict, key=class_predict_dict.get)
        
        if ground_truth == class_predict:
            predicted_correct = 1
        else:
            predicted_correct = 0
        
        pred_dict = {'img_filename': [img_filename], 'class_predict_dict': [class_predict_dict], 'ground_truth': [ground_truth], 'class_predict': [class_predict], 'predicted_correct': [predicted_correct]}
        df_single = pd.DataFrame(pred_dict)
        
        return df_single

    all_images = os.listdir(img_dir)
    for img_filename in tqdm(all_images, desc=f"epoch_{epoch_number}: Processing images", unit="image"):
        df = read_single_image(img_dir, img_filename, sentences, text_features)
        df_all = pd.concat([df_all, df]).reset_index(drop=True)

    #save df_all to csv
    csv_file = f"/home/yilu/Development/open_clip/Knee_OA/temp/epoch_{epoch_number}.csv"
    df_all.to_csv(csv_file, index=False)

    total_rows = len(df_all)
    # Count the occurrences of '1' in the 'predicted_correct' column
    count_ones = df_all['predicted_correct'].sum()
    epoch_filename = os.path.splitext(os.path.basename(pretrained_model))[0]
    predicted_correct_pct = count_ones /total_rows *100
    print(f"Epoch_file: {epoch_filename}. Total images: {total_rows}, predicted_correct': {count_ones}, predicted_correct (%)': {predicted_correct_pct}%")
    
    epoch_dict = {'pt_dir': [pt_dir],
                'epoch_filename': [epoch_filename], 
                'sentences': [sentences], 
                'total_rows': [total_rows], 
                'predicted_correct': [count_ones], 
                'predicted_correct (%)': [predicted_correct_pct]}
    df_single_epoch = pd.DataFrame(epoch_dict)
    return df_single_epoch


def process_single_pretrained_model(pretrained_model, csv_file):
    df_single_pretrained_model = pd.DataFrame()
    df = process_single_epoch(pretrained_model, classification)
    df_single_pretrained_model = pd.concat([df_single_pretrained_model, df]).reset_index(drop=True)
    save_to_csv(df_single_pretrained_model, csv_file)
    
def extract_epoch_num(filepath):
    # Split the filename from the path
    filename = os.path.basename(filepath) #filepath.split('/')[-1]
    # Extract the number between "epoch_" and ".pt"
    epoch_num = int(filename.split('epoch_')[1].split('.pt')[0])
    return epoch_num

def save_to_csv(df, csv_file):
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        # If the CSV file already exists, append without header
        df.to_csv(csv_file, mode='a', header=False, index=False)

def main():
    #obtain the latest pretrained model folder
    pt_folder_root = '/mnt/g/Logtemp/open_clip/Knee_OA/'
    dir_1 = [a for a in os.listdir(pt_folder_root) if a.startswith('2023_08_24-23')][0]
    pt_folder_path = os.path.join(pt_folder_root, dir_1, 'checkpoints')

    pt_files = [os.path.join(pt_folder_path, filename) for filename in os.listdir(pt_folder_path) if filename.endswith('.pt')]
    #filter pt_files with low epoch number
    pt_files = [pt_file for pt_file in pt_files if extract_epoch_num(pt_file) >= 0]
    # Sort the list
    sorted_pt_files = sorted(pt_files, key=extract_epoch_num)
    csv_file = "/home/yilu/Development/open_clip/Knee_OA/temp/benchmark_all_epochs.csv"
    for pretrained_model in sorted_pt_files:
        process_single_pretrained_model(pretrained_model, csv_file)


if __name__ == "__main__":
    main()