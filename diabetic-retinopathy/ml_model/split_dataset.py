import os
import shutil
from sklearn.model_selection import train_test_split

def count_images(class_path):
    images = [name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))]
    return len(images)

def split_dataset(source_dir, train_dir, test_dir, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    classes = ['No_DR', 'Proliferate_DR', 'Mild', 'Moderate', 'Severe']

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Class '{class_name}' directory not found at '{class_path}'. Skipping this class.")
            continue
        
        num_images = count_images(class_path)
        print(f"Class '{class_name}' contains {num_images} images.")
        
        if num_images == 0:
            print(f"Warning: No images found in class directory '{class_path}'. Skipping this class.")
            continue
        
        images = os.listdir(class_path)
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)
        
        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_class_dir, image))
        
        for image in test_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(test_class_dir, image))

# Define source directory and train/test directories
source_dir = r'C:\Users\ishani anushka\Desktop\Effective_Solutions\diabetic-retinopathy\ml_model\dataset'
train_dir = os.path.join(source_dir, 'train')
test_dir = os.path.join(source_dir, 'test')

# Split the dataset
split_dataset(source_dir, train_dir, test_dir)
