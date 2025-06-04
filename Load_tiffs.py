#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:15:19 2025

@author: Jessica Angulo Capel
"""
import tifffile
import numpy as np
from cellpose import models
from scipy.ndimage import zoom
from skimage.measure import regionprops, label
from scipy.spatial.distance import cdist
from skimage.segmentation import relabel_sequential
from google.colab import files
import io


class Segmentator():
    def __init__(
            self,
            segmentation_type: int = 2,
            cell_diameter: int = 100,
            model_type: str = 'cyto3',
            flow_threshold = 0.3,
            stitch_threshold = 0.1,
            skip: bool= False,
            use_gpu: bool = True,):

        self.segmentation_type = segmentation_type
        self.cell_diameter = cell_diameter
        self.flow_threshold = flow_threshold
        self.stitch_threshold = stitch_threshold
        self.skip_half = skip
        self.model_type = model_type
        self.use_gpu = use_gpu

        # Load the model; use 'cyto' or train a custom model if needed
        self.model = models.CellposeModel(
            gpu=self.use_gpu,
            model_type=self.model_type)
        
        
    def load_images(self):
        # Ask user to select two TIFF files
        uploaded = files.upload()
        # Load them into numpy arrays (preserves Z dimension)
        filenames = list(uploaded.keys())
        # Load the two TIFF stacks
        cyto = tifffile.imread(io.BytesIO(uploaded[filenames[0]]))
        auto = tifffile.imread(io.BytesIO(uploaded[filenames[1]])) 
        
        if self.skip_half:
            cyto_half = cyto[::2]  # skip every second slice
            auto_half = auto[::2]
            image = np.stack([cyto_half, auto_half], axis=-1)
        else:
            # Stack them into a multi-channel image (channels last format expected by Cellpose)
            image = np.stack([cyto, auto], axis=-1)  # Shape: [Z, Y, X, 2]

        return image            
        
    def segmentate_image(self, image: np.ndarray):
        masks_3d = []
        if self.segmentation_type == 2:
            masks_3d = self.segmentate_2D(image)
        elif self.segmentation_type == 3:
            masks_3d = self.segmentate_3D(image)
        else:
            raise Exception("Wrong type of segmentation. Choose either 2 for 2D or 3 for 3D") 
        return masks_3d

    def segmentate_2D(self, image: np.ndarray):
        # Segment each slice independently
        masks_list = []
        
        for z in range(image.shape[0]):
            img_slice = image[z]  # [Y, X, 2]
            masks, _, _ = self.model.eval(
                img_slice,
                channels=[0, 1],
                do_3D=False,
                diameter=self.cell_diameter,
                flow_threshold=self.flow_threshold)
            masks_list.append(masks)
        
        # Re-stack into 3D mask
        masks_stacked = np.stack(masks_list, axis=0)  # shape: [Z, Y, X]
        
        # Connect labels between stacks with nearest neighbours
        max_centroid_dist = 30 #number of pixel, distance between centroids
        min_overlap = 0.2 #fraction of pixels that overlap
        masks_3d = np.zeros_like(masks_stacked, dtype=np.int32)
        next_label = 1
        previous_regions = {}
        for z in range(masks_stacked.shape[0]):
            labeled = label(masks_stacked[z])  # ensure contiguous labels per slice
            props = regionprops(labeled)
            current_regions = {}
            
            for region in props:
                centroid = region.centroid  # (row, col)
                coords = region.coords
                label_id = None

                # Try to find a match from previous slice
                for prev_id, prev in previous_regions.items():
                    dist = np.linalg.norm(np.array(prev['centroid']) - np.array(centroid))
                    if dist > max_centroid_dist:
                        continue

                    # Check pixel overlap
                    overlap = np.intersect1d(
                        np.ravel_multi_index(coords.T, labeled.shape),
                        np.ravel_multi_index(prev['coords'].T, labeled.shape)
                    )
                    if len(overlap) / len(coords) > min_overlap:
                        label_id = prev_id
                        break

                if label_id is None:
                    label_id = next_label
                    next_label += 1

                masks_3d[z][coords[:, 0], coords[:, 1]] = label_id
                current_regions[label_id] = {
                    'centroid': centroid,
                    'coords': coords
                }

            previous_regions = current_regions

        # Optional: relabel sequentially
        masks_3d, _, _ = relabel_sequential(masks_3d)
        
        return masks_3d
        
    def segmentate_3D(self, image: np.ndarray):
        # Run segmentation
        masks_3d, _, _ = self.model.eval(
            image,
            channels=[0, 1],  # channel 0 = to segment, 1 = additional
            do_3D=True,
            diameter=self.cell_diameter,
            flow_threshold=self.flow_threshold,
            stitch_threshold=self.stitch_threshold)
        return masks_3d
    

def main():
    segmentator_object = Segmentator()
    image_to_segment = segmentator_object.load_images()
    masks_3d = segmentator_object.segmentate_image(
        image_to_segment)
    
    # Save result
    tifffile.imwrite('cell_masks_adria_2d.tif', masks_3d.astype(np.uint16))
    tifffile.imwrite('cell_image_adria_2d.tif', image_to_segment.astype(np.uint16))
    
    # Upload mask again
    #masks_3d = tifffile.imread('cell_masks_adria_2d.tif')
    #image = tifffile.imread('cell_image_adria_2d.tif')

if __name__ == '__main__':
    main()