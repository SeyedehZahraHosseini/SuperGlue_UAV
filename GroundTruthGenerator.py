import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("F:\IDM\SuperPoint\dataset\CGTD_Tool\SuperGluePretrainedNetwork-master"))

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from PIL import Image, ImageTk
import tkinter as tk

import torch
print(torch.version.cuda)        # e.g. "11.8" or None
print(torch.backends.cudnn.enabled)  # should be True if CUDA build

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: " + device)

#########################################
# Utility Function: Point-Line Distance
#########################################
def point_line_distance(point, line_start, line_end):
    """
    Computes the perpendicular distance from a point to a line segment.
    
    Args:
        point (tuple): (x, y) coordinates of the point.
        line_start (tuple): (x, y) coordinates of the start of the line segment.
        line_end (tuple): (x, y) coordinates of the end of the line segment.
        
    Returns:
        float: The perpendicular distance from the point to the line segment.
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)
    t = np.clip(np.dot(point_vec, line_vec) / line_len, 0, 1)
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

#########################################
# GroundTruthGenerator Class
#########################################
class GroundTruthGenerator:
    """
    Orchestrates the workflow for processing RGB and Thermal image pairs.
    
    Workflow:
      1. Loads an RGB and Thermal image pair from the data directory.
      2. Extracts keypoints and descriptors from each image using SuperPoint.
      3. Matches keypoints between images using SuperGlue.
      4. Visualizes the matches and provides an interactive UI for user quality control.
         - The user can select/deselect matches (selected matches highlighted in green).
         - Zoom and pan functionality are added.
         - The match currently hovered over is highlighted in dark blue.
      5. Upon user confirmation, writes the approved matches to a ground truth file in the SuperGlue format.
      6. Waits for a 'Next' command from the user to process the next image pair.
    """
    def __init__(self, data_dir: str, backend_dir: str, pair_ids: list,
                 debug: bool = True, sp_model_path: str = None,
                 sp_config: dict = None, sg_config: dict = None):
        self.debug = debug
        self.data_dir = data_dir
        self.backend_dir = backend_dir
        self.pair_ids = pair_ids

        # پیکربندی مدل‌ها
        default_sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        }
        default_sg_config = {'weights': 'outdoor'}

        # اگر کاربر چیزی نداد، مقدار پیش‌فرض رو استفاده کن
        self.sp_config = sp_config or default_sp_config
        self.sg_config = sg_config or default_sg_config

        # SuperPoint
        self.superpoint = SuperPoint(self.sp_config).eval().to(device)
        if sp_model_path is not None:
            if os.path.exists(sp_model_path):
                state_dict = torch.load(sp_model_path, map_location=device)
                self.superpoint.load_state_dict(state_dict)
                if self.debug:
                    print(f"[DEBUG] SuperPoint weights loaded from {sp_model_path}")
            else:
                print(f"[WARNING] SuperPoint model path not found: {sp_model_path}")

        # SuperGlue
        self.superglue = SuperGlue(self.sg_config).eval().to(device)
        if self.debug:
            print(f"[GroundTruthGenerator] Models loaded on {device}")

        # وابستگی‌ها
        self.image_loader = ImageLoader(data_dir)
        self.sp_extractor = SuperPointExtractor(self.superpoint)
        self.sg_matcher = SuperGlueMatcher(self.superglue)
        self.file_writer = GroundTruthFileWriter(backend_dir, debug=debug)
        self.current_pair_id = None
        self.rgb = None
        self.thermal = None

    
    def on_matches_selected(self, selected_matches):
        if self.debug:
            print(f"[DEBUG] User approved {len(selected_matches)} matches.")
        if not hasattr(self, "current_pair_id"):
            print("[ERROR] current_pair_id is not set before saving matches.")
            return
        self.file_writer.write_ground_truth_file(self.current_pair_id, selected_matches)
        if self.debug:
            print(f"[DEBUG] Ground truth file written for pair: {self.current_pair_id}")

    def visualize_keypoints(self, keypoints_list, scores_list , images, cmap='gray'):
        """
        اجرای مدل تشخیص نقاط کلیدی و رسم نتایج با رنگ و اندازه متناسب با confidence.

        پارامترها:
            sp_model : مدل (مانند SuperPoint)
            images : آرایه numpy با ابعاد (N, H, W)
            cmap : نقشه رنگ برای نمایش تصاویر
        """

        # رسم تصاویر
        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
        if len(images) == 1:
            axes = [axes]

        for img, keypoints, scores, ax in zip(images, keypoints_list, scores_list, axes):
            ax.set_title(f"number of keypoints:{len(keypoints)}")
            ax.imshow(img, cmap=cmap)
            ax.axis('off')

            # نرمال‌سازی score بین 0 و 1
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            # تعیین سایز بر اساس score
            sizes = 2 + 8 * norm_scores  # حداقل 2، حداکثر 10

            # تعیین رنگ بر اساس score (کمتر → آبی، بیشتر → قرمز)
            colors = np.stack([norm_scores, np.zeros_like(norm_scores), 1 - norm_scores], axis=1)

            ax.scatter(keypoints[:, 0], keypoints[:, 1], s=sizes, c=colors, lw=0)

        plt.tight_layout()
        plt.show()

    def visualize_matches(self, images, matches_list, threshold=5.0, save_path=None):
        """
        نمایش تصاویر و تطابق‌ها با رنگ‌های سبز/قرمز بر اساس شباهت نقاط.
        
        پارامترها:
            images: لیست دو تصویر
            matches_list: لیست tuples [(pt0, pt1), ...]
            threshold: حداکثر فاصله پیکسل برای در نظر گرفتن match صحیح
            save_path: مسیر ذخیره تصویر
        """
        processed = []
        for img in images:
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            processed.append(img)

        img1, img2 = processed
        w = img1.shape[1]
        out = cv2.hconcat([img1, img2])

        correct_count = 0
        incorrect_count = 0

        correct_matches = []

        for matches in matches_list:
            for pt0, pt1 in matches:
                distance = np.linalg.norm(np.array(pt0) - np.array(pt1))
                if distance <= threshold:
                    color = (0, 255, 0)  # سبز برای match خوب
                    correct_count += 1
                    correct_matches.append((pt0, pt1))
                else:
                    color = (0, 0, 255)  # قرمز برای match نادرست
                    incorrect_count += 1

                pt1_shifted = (int(pt1[0] + w), int(pt1[1]))
                cv2.line(out, (int(pt0[0]), int(pt0[1])), pt1_shifted, color, 1)

        # نمایش با matplotlib
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Correct matches: {correct_count}, Incorrect matches: {incorrect_count}", fontsize=16)
        plt.show()

        if save_path:
            cv2.imwrite(save_path, out)
        return correct_matches, correct_count


    def process_image_pair(self, pair_id: str = None, rgb_image=None, thermal_image=None, debug: bool = False):
        """
        Process a pair of images (RGB + Thermal).
        The user can either provide the images directly, or specify a pair_id to load them from disk.

        Args:
            pair_id (str): Optional ID for loading the image pair from disk.
            rgb_image (np.ndarray or torch.Tensor): Optional directly provided RGB image.
            thermal_image (np.ndarray or torch.Tensor): Optional directly provided thermal image.
            debug (bool): If True, prints debug information.
        """
        if debug:
            print(f"[DEBUG] Starting processing for image pair: {pair_id}")

        self.current_pair_id = pair_id 

        # Step 1: Load or use provided images
        if rgb_image is not None and thermal_image is not None:
            if debug:
                print("[DEBUG] Using directly provided image pair.")
            rgb_image = cv2.resize(rgb_image, (640, 480))
            thermal_image = cv2.resize(thermal_image, (640, 480))

            if debug or self.debug:
                print(f"[DEBUG] Loaded images: RGB {rgb_image.shape}, Thermal {thermal_image.shape}")
        else:
            if pair_id is None:
                raise ValueError("Either pair_id or both rgb_image and thermal_image must be provided.")
            if debug:
                print("[DEBUG] Loading image pair from dataset.")
            rgb_image, thermal_image = self.image_loader.load_image_pairs(pair_id)

        # Step 1: Load the image pair.
        # rgb_image, thermal_image = self.image_loader.load_image_pairs(pair_id)

        # Convert images to RGB for display.
        org_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        org_thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)

        # Create grayscale versions for feature extraction.
        grayscale_rgb_image1 = cv2.cvtColor(org_rgb_image, cv2.COLOR_BGR2GRAY)
        grayscale_thermal_image1 = cv2.cvtColor(org_thermal_image, cv2.COLOR_BGR2GRAY)

        grayscale_rgb_image2 = grayscale_rgb_image1.astype(np.float32) / 255.0
        grayscale_thermal_image2 = grayscale_thermal_image1.astype(np.float32) / 255.0

        grayscale_rgb_image = torch.from_numpy(grayscale_rgb_image2).unsqueeze(0).unsqueeze(0).to(device)
        grayscale_thermal_image = torch.from_numpy(grayscale_thermal_image2).unsqueeze(0).unsqueeze(0).to(device)

        if debug:
            print(f"[DEBUG] Loaded images and grayscale images for pair {pair_id}.")

        # Step 2: Extract keypoints and descriptors.
        kpts1, desc1, scores1 = self.sp_extractor.extract_keypoints(grayscale_rgb_image)
        kpts2, desc2, scores2 = self.sp_extractor.extract_keypoints(grayscale_thermal_image)
        if debug:
            print("[DEBUG] Extracted keypoints for RGB and Thermal.")

        kpts = [kpts1.cpu().numpy(), kpts2.cpu().numpy()]
        scores = [scores1.cpu().numpy(), scores2.cpu().numpy()]
        images = [grayscale_rgb_image2, grayscale_thermal_image2 ]
        
        self.visualize_keypoints(kpts, scores, images)


        kpts1 = kpts1.unsqueeze(0)
        kpts2 = kpts2.unsqueeze(0)
        scores1 = scores1.unsqueeze(0)
        scores2 = scores2.unsqueeze(0)

        # Step 3: Prepare data dictionary for SuperGlue.
        data = {
            'image0': grayscale_rgb_image, 
            'image1': grayscale_thermal_image,
            'keypoints0': kpts1, 
            'keypoints1': kpts2,
            'descriptors0': desc1, 
            'descriptors1': desc2,
            'scores0': scores1, 
            'scores1': scores2
        }
        if data['scores0'].dim() == 3:
            data['scores0'] = data['scores0'].squeeze(-1)
        if data['scores1'].dim() == 3:
            data['scores1'] = data['scores1'].squeeze(-1)

        print(data['scores0'].mean(), data['scores1'].mean())

        
        # Step 4: Compute matches using SuperGlue.
        matches = self.sg_matcher.match_keypoints(data)
        if debug:
            print(f"[DEBUG] Computed {len(matches)} matches.")

        correct_matches, correct_count = self.visualize_matches(images, [matches], threshold=10.0, save_path=None)

        # self.file_writer.write_ground_truth_file(pair_id, correct_matches, debug=debug)
       

        return correct_count

    def run(self, debug: bool = False):
        if debug:
            print("[DEBUG] Starting main processing loop...")
        pair_ids = self.pair_ids
        correct_matches = []
        for pair_id in pair_ids:
            if debug:
                print(f"[DEBUG] Processing next image pair: {pair_id}")
            correct_count = self.process_image_pair(pair_id, debug=debug)
            correct_matches.append(correct_count)
            if debug:
                print(f"[DEBUG] Finished {pair_id}: {len(correct_matches)} GT matches saved.\n")
        if debug:
            print(f"[DEBUG] All {len(self.pair_ids)} pairs processed.")
            print(f"[DEBUG] Ground truth files saved in: {self.backend_dir}")
        
        return correct_matches

    def get_all_pair_ids(self, debug: bool = False):
        # Placeholder implementation.
        pair_ids = ["pair_1", "pair_2", "pair_3"]
        if debug:
            print(f"[DEBUG] Retrieved pair identifiers: {pair_ids}")
        return pair_ids

#########################################
# Dependency 1: ImageLoader
#########################################
class ImageLoader:
    def __init__(self, data_dir: str, debug: bool = False):
        self.data_dir = data_dir
        self.debug = debug
        if self.debug:
            print(f"[DEBUG] ImageLoader initialized with directory: {data_dir}")
    
    def load_image_pairs(self, pid, debug=False):
        

        # for pid in pair_ids:
        rgb_filename = f"satellite_{pid}.png"
        thermal_filename = f"thermal_{pid}.png"

        rgb_path = os.path.join(self.data_dir, "rgb", rgb_filename)
        thermal_path = os.path.join(self.data_dir, "thermal", thermal_filename)

        if debug or self.debug:
            print(f"[DEBUG] Loading RGB image from: {rgb_path}")
            print(f"[DEBUG] Loading Thermal image from: {thermal_path}")

        rgb_image = cv2.imread(rgb_path)
        thermal_image = cv2.imread(thermal_path)

        if rgb_image is None:
            raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")
        if thermal_image is None:
            raise FileNotFoundError(f"Could not load Thermal image: {thermal_path}")

        # rgb_image = cv2.resize(rgb_image, (640, 480))
        # thermal_image = cv2.resize(thermal_image, (640, 480))

        if debug or self.debug:
            print(f"[DEBUG] Loaded images: RGB {rgb_image.shape}, Thermal {thermal_image.shape}")


        return rgb_image, thermal_image


#########################################
# Dependency 2: SuperPointExtractor
#########################################

class SuperPointExtractor:
    def __init__(self,superpoint_model, debug: bool = False):
        self.orb = cv2.ORB_create()
        self.debug = debug
        self.sp_config = {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        }
        self.superpoint = superpoint_model
        if self.debug:
            print("[DEBUG] SuperPointExtractor initialized.")


    
    def extract_keypoints(self, image, debug: bool = False):
        if image is None:
            raise ValueError("Failed to load image.")
        with torch.no_grad():
            output = self.superpoint({'image': image})
        keypoints = output['keypoints'][0].to(device)

        descriptors = output['descriptors'][0].to(device)
        scores = output['scores'][0].to(device)

        if self.debug:
            print(f"[DEBUG] Extracted {len(keypoints)} keypoints.")
        
        return keypoints, descriptors, scores

#########################################
# Dependency 3: SuperGlueMatcher
#########################################
class SuperGlueMatcher:
    def __init__(self, superglue_model=None, sg_config=None, debug: bool = False):
        self.debug = debug
        self.sg_config = sg_config or {'weights': 'outdoor', 'match_threshold': 0.1}
        self.superglue = superglue_model or SuperGlue(self.sg_config).eval().to(device)
        if self.debug:
            print(f"[DEBUG] SuperGlueMatcher initialized with config: {self.sg_config}")

    def match_keypoints(self, data, debug: bool = False):
        with torch.no_grad():
            raw_matches = self.superglue(data)['matches0'][0].cpu().numpy()            
        if self.debug or debug:
            num_valid = np.sum(raw_matches > -1)
            print(f"[DEBUG] Found {num_valid} valid matches out of {raw_matches.shape[0]} keypoints.")
        keypoints0 = data['keypoints0'][0].cpu().numpy()  
        keypoints1 = data['keypoints1'][0].cpu().numpy()  
        match_pairs = []
        for i, match_idx in enumerate(raw_matches):
            if match_idx > -1:
                pt0 = keypoints0[i]
                pt1 = keypoints1[int(match_idx)]
                match_pairs.append((pt0, pt1))
        if self.debug or debug:
            print(f"[DEBUG] Converted matches to {len(match_pairs)} coordinate pairs.")
        return match_pairs

#########################################
# Dependency 5: GroundTruthFileWriter
#########################################

class GroundTruthFileWriter:
    def __init__(self, backend_dir: str, debug: bool = False):
        self.backend_dir = backend_dir
        self.debug = debug
        os.makedirs(self.backend_dir, exist_ok=True)  # اطمینان از وجود پوشه
        if self.debug:
            print(f"[DEBUG] GroundTruthFileWriter initialized with backend directory: {backend_dir}")
    
    def write_ground_truth_file(self, pair_id: str, correct_matches, debug: bool = False):
        filename = os.path.join(self.backend_dir, f"{pair_id}_ground_truth.txt")
        local_debug = debug or self.debug
        
        if local_debug:
            print(f"[DEBUG] Writing ground truth file to: {filename}")
            print(f"[DEBUG] Number of correct matches: {len(correct_matches)}")
        
        with open(filename, 'w') as f:
            for pt_rgb, pt_thermal in correct_matches:
                line = f"{pt_rgb[0]:.6f} {pt_rgb[1]:.6f} {pt_thermal[0]:.6f} {pt_thermal[1]:.6f}\n"
                f.write(line)
                if local_debug:
                    print(f"[DEBUG] Wrote GT match: {line.strip()}")
        
        if local_debug:
            print(f"[DEBUG] Ground truth file saved: {filename}")
#########################################
# End of File
#########################################