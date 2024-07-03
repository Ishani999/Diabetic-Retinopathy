import os

# Define the path to the train directory
train_dir = r'c:\Users\ishani anushka\Desktop\Effective_Solutions\diabetic-retinopathy\ml_model\dataset\train'

# Check if the directory exists
if os.path.exists(train_dir):
    # List all files and directories in the train directory
    contents = os.listdir(train_dir)
    
    # Print out the contents
    print(f"Contents of '{train_dir}':")
    for item in contents:
        print(item)
else:
    print(f"Directory '{train_dir}' not found.")
