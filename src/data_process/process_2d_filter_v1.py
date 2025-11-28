import json
import os
import pickle
import uuid
import ast
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm
import numpy as np
from PIL import Image
import nibabel as nib
from util import sort_files, group_and_merge_3d_bboxes_v2, group_files
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.desc_map = self._load_desc_map(config['desc_path'])
        self.uid = str(uuid.uuid4())
        
    @staticmethod
    def _load_desc_map(desc_path):
        """Load description mapping from JSON file."""
        with open(desc_path, "r") as f:
            return json.load(f)

    def process_data(self):
        """Main method to process data from source to target directories."""
        target_root = self._create_target_directory()
        
        for split in self.config['splits']:
            self._process_split(split, target_root)

    def _create_target_directory(self):
        """Create target directory structure."""
        target_root = os.path.join(
            self.config['target_root'],
            f"{self.config['num_concat']}_{self.config['tag']}_slices",
            self.uid
        )
        os.makedirs(target_root, exist_ok=True)
        logger.info(f"Target root: {target_root}")
        return target_root

    def _process_split(self, split, target_root):
        """Process data for a single split (train/test)."""
        source_dir = os.path.join(self.config['source_root'], split)
        save_split_dir = os.path.join(target_root, split)
        save_data_dir = os.path.join(save_split_dir, "data")
        os.makedirs(save_data_dir, exist_ok=True)

        patient_json_files = os.listdir(os.path.join(source_dir, "data"))
        p_ids = [path.split(".")[0] for path in patient_json_files]
        
        logger.info(f"Processing {split} split with {len(p_ids)} patients")
        total_data = 0

        for p_id in tqdm(p_ids, desc=f"Processing {split} patients"):
            patient_data = self._process_patient(p_id, source_dir, save_data_dir)
            total_data += len(patient_data)

        logger.info(f"Total data processed for {split}: {total_data}")
        logger.info(f"Processed data saved to: {save_data_dir}")
    def _process_patient(self, p_id, source_dir, save_data_dir):
        """Process data for a single patient."""
        with open(os.path.join(source_dir, "data", f"{p_id}.json"), "rb") as f:
            data = json.load(f)

        slide_data_map = self._clean_patient_data(data)
        
        patient_chunks = self._process_slices(p_id, source_dir, slide_data_map)
        
        with open(os.path.join(save_data_dir, f"{p_id}.json"), "w") as f:
            json.dump(patient_chunks, f)
            
        return data

    @staticmethod
    def _clean_patient_data(data):
        """Remove unwanted keys from patient data."""
        drop_keys = [
            "image", "image_with_bboxes", "Original", "__index__level_0__", "No.",
            "Column 9", "Deliverable", "Doctor", "Start date", "Google Drive Link",
            "rotated_link", "vn", "fr", "de", "mandarin", "korean", "japanese", "vi"
        ]
        
        slide_data_map = {}
        for d in data:
            for k in drop_keys:
                d.pop(k, None)
            slide_data_map[d["Slide"]] = d
        return slide_data_map

    def _process_slices(self, p_id, source_dir, slide_data_map):
        """Process and merge slices for a patient."""
        img_slide_dir = os.path.join(source_dir, "image", p_id)
        slide_base = [f.split(".")[0] for f in os.listdir(img_slide_dir)]
        slide_subgroups = group_files(slide_base)
        
        patient_chunks = []
        for subgroup in slide_subgroups:
            slide_shape_map, annot_shape_map = self._load_slice_data(p_id, source_dir, subgroup)
            valid_slides = self._filter_valid_slides(subgroup, slide_shape_map)
            
            if self.config['num_concat'] == -1:
                patient_chunks.append(self.merge_slices(valid_slides, slide_data_map))
            else:
                for i in range(0, len(valid_slides) - self.config['num_concat'] + 1):
                    chunk_slides = valid_slides[i:i + self.config['num_concat']]
                    patient_chunks.append(self.merge_slices(chunk_slides, slide_data_map))
                    
        return patient_chunks

    def _load_slice_data(self, p_id, source_dir, subgroup):
        """Load slice and annotation data for a subgroup of slides."""
        img_slide_dir = os.path.join(source_dir, "image", p_id)
        annot_slide_dir = os.path.join(source_dir, "image_with_bboxes", p_id)
        
        slide_shape_map = {}
        annot_shape_map = {}
        
        for slide in subgroup:
            with open(os.path.join(img_slide_dir, f"{slide}.pkl"), "rb") as f:
                slice_data = pickle.load(f)
            with open(os.path.join(annot_slide_dir, f"{slide}.pkl"), "rb") as f:
                annot_data = pickle.load(f)
                
            slide_shape_map[slide] = self.rgb_to_grayscale(slice_data)
            annot_shape_map[slide] = self.rgb_to_grayscale(annot_data)
            
        return slide_shape_map, annot_shape_map

    @staticmethod
    def _filter_valid_slides(subgroup, slide_shape_map):
        """Filter slides to keep only those with the most common shape."""
        shape_counter = Counter(slice_.shape for slice_ in slide_shape_map.values())
        reference_shape = max(shape_counter.items(), key=lambda x: x[1])[0]
        
        valid_slides = [s for s in subgroup if slide_shape_map[s].shape == reference_shape]
        logger.info(f"Invalid slides removed: {len(subgroup) - len(valid_slides)}")
        return valid_slides

    @staticmethod
    def rgb_to_grayscale(img_rgb):
        """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
        return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def merge_slices(self, list_slices, slice_data_map):
        """Merge data from multiple slices into a single output."""
        slice_data = [slice_data_map[s] for s in list_slices]
        
        return {
            "Patient ID": slice_data[0]["Patient ID"],
            "Q1": list(set(slice["Q1"] for slice in slice_data)),
            "Q2": list(set(slice["Q2"] for slice in slice_data)),
            "Q3": list(set(slice["Q3"] for slice in slice_data)),
            "Q4": list(set(slice["Q4"] for slice in slice_data)),
            "A1": self.merge_A1_data([slice["A1"] for slice in slice_data]),
            "A3": self.merge_A3_data([slice["A3"] for slice in slice_data]),
            "A2": self.merge_A2_data(self.merge_A3_data([slice["A3"] for slice in slice_data])),
            "A4": self.merge_A4_data([slice["A4"] for slice in slice_data]),
            "bbox_3d": group_and_merge_3d_bboxes_v2(
                [ast.literal_eval(slice["A1"]) for slice in slice_data]),
            "slice order": list_slices
        }

    def merge_A1_data(self,list_A1):
        """Merge A1 (bounding box) data."""
        bbox_input = []
        for bboxes_ls in list_A1:
            if bboxes_ls:
                try:
                    bbox_input.append(ast.literal_eval(bboxes_ls))
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error converting to bbox_A1: {e}")
        return bbox_input

    
    def merge_A2_data(self,pairs):
        """Merge A2 data using description mapping."""
        
        try:
            return "\n".join(
                self.desc_map[key][str(value)]
                for key, value in pairs.items()
            )
        except Exception as e:
            logger.error(f"Error in merge_A2: {e}")
            return ""

    def merge_A3_data(self,list_A3):
        """Merge A3 scoring data."""
        result = {"GCA": 0, "Koedam": 0, "MTA": 0}
        if not list_A3:
            return result
            
        for s in list_A3:
            if s:
                try:
                    for pair in [p.strip() for p in s.split(",")]:
                        key, value = pair.split("=")
                        result[key.strip()] = max(result.get(key.strip(), 0), int(value.strip()))
                except Exception as e:
                    logger.error(f"Error merging A3 data {s}: {e}")
        return result

    def merge_A4_data(self, list_A4):
        """Merge A4 (dementia status) data."""
       
        degree_levels={
            "non": "Non-Dementia",
            "mild": "Mild-Dementia",
            "moderate": "Moderate-Dementia"
        }

        if not list_A4:
            logger.warning("Empty A4 list, defaulting to Non-Dementia")
            return "Non-Dementia"

        cleaned_list = [str(s) for s in list_A4 if s is not None]
        for level,description in reversed(degree_levels.items()):
        
            for s in cleaned_list:
                if level.lower() in s.lower():
                    print("match s", s, " for", level)
                    return description

        logger.warning(f"No dementia level found in A4, defaulting to Non-Dementia: {cleaned_list}")
        return "Non-Dementia"

    @staticmethod
    def save_dsc_data(data_dir, save_path="desc.json"):
        """Generate and save description mapping from training data."""
        if os.path.exists(save_path):
            os.remove(save_path)
            
        result = defaultdict(dict)
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

        for f in tqdm(files, desc="Generating description map"):
            with open(f, "r") as f:
                data = json.load(f)
                
            for d in data:
                for line in [l.strip() for l in d["Notes"].split("\n") if l.strip()]:
                    try:
                        scale_part, desc_part = line.split("=", 1)
                        score_part, description = desc_part.split(":", 1)
                        
                        scale = scale_part.strip()
                        score = int(score_part.strip())
                        description = description.strip()

                        result[scale][score] = description
                    except Exception as e:
                        logger.error(f"Error processing line: {line}\nError: {e}")

        with open(save_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    config = {
        'desc_path': "/root/TracGPT-R3D/src/data_process/desc.json",
        'source_root': "clean_data_s_chain",
        'target_root': "pseudo_3d",
        'num_concat': 32,
        'tag': "all",
        'splits': ["train", "test"]
    }
    
    processor = DataProcessor(config)
    processor.process_data()