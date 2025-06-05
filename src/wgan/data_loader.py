import cv2
import random
import os
from tqdm import tqdm
import multiprocessing

def process_one_image(args):
    input_path, output_path = args
    # Create a new DataLoader instance for each process (avoid sharing self)
    loader = DataLoader()
    processed = loader.preprocess_oct_image(input_path)
    cv2.imwrite(output_path, processed)

class DataLoader:
    def __init__(self):
        pass

    def preprocess_oct_image(self, image_path):
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert white pixels to black
        img[img == 255] = 0


        '''
        ## Random spin of the image
        angle = random.randint(0, 360)  # Generate random angle between 0 and 360 degrees
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)  # Get rotation matrix
        img = cv2.warpAffine(img, M, (cols, rows))  # Apply rotation
        '''

        # Optional: Normalize the image
        cleaned_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Convert white pixels to black again (in case normalization created new white pixels)
        cleaned_image[cleaned_image == 255] = 0

        return cleaned_image

    def process_dataset(self, input_dir, output_dir):
        # Get the class name from the input directory
        class_name = os.path.basename(input_dir)
        
        # Create class-specific output directory
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist")
            return

        # Get all image files from input directory
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Get existing processed files
        existing_files = set(f.replace('processed_', '') for f in os.listdir(class_output_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        
        # Filter out already processed images
        to_process = [f for f in image_files if f not in existing_files]
        
        if not to_process:
            print(f"All images for class {class_name} already processed, skipping...")
            return
        
        print(f"Processing {len(to_process)} new images for class {class_name}...")

        # Prepare arguments for multiprocessing
        args = [
            (os.path.join(input_dir, image_file),
             os.path.join(class_output_dir, f"processed_{image_file}"))
            for image_file in to_process
        ]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Use tqdm to wrap the imap iterator for progress bar
            list(tqdm(pool.imap(process_one_image, args), total=len(args), desc=f"Processing {class_name}"))


if __name__ == "__main__":
    data_loader = DataLoader()