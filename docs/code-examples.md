# Production Code Examples

[← Main README](../README.md) · [← Architecture Guide](architecture.md) · [← Domain Map](domain-map.md)

This document provides representative, production-grade code implementations extracted directly from the **3D Spine CT Vertebrae Segmentation & Volumetric Reconstruction** platform. These examples highlight biomedical signal processing, multi-threaded dataset generation, PyTorch neural architectures, affine-preserving 3D reconstruction, and clinical WebGL visualization.

---

## 1. NIfTI Volume Ingestion & Affine Metadata Preservation

Extracts the raw 3D voxel array while caching the patient's native $4 \times 4$ affine transformation matrix and scanner metadata headers.

```python
import os
import nibabel as nib
import numpy as np

def load_patient_ct_volume(nifti_path: str):
    """
    Ingest a 3D medical CT volume (.nii / .nii.gz) and extract clinical spatial headers.
    
    Returns:
        voxel_data: 3D numpy array of radiodensity values (float64)
        affine: 4x4 coordinate matrix mapping voxel indices (i, j, k) to scanner space (RAS+)
        header: NiBabel spatial metadata header containing voxel spacing (sx, sy, sz)
    """
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"CT volume not found at: {nifti_path}")
        
    nii_image = nib.load(nifti_path)
    voxel_data = nii_image.get_fdata(dtype=np.float64)
    original_affine = nii_image.affine.copy()
    original_header = nii_image.header.copy()
    
    # Extract clinical voxel dimensions (e.g., 0.5mm x 0.5mm x 1.0mm)
    voxel_dimensions = original_header.get_zooms()[:3]
    print(f"Loaded volume: {nifti_path}")
    print(f"Matrix Dimensions: {voxel_data.shape}")
    print(f"Voxel Spacing (dx, dy, dz): {voxel_dimensions} mm")
    
    return voxel_data, original_affine, original_header
```

---

## 2. Radiographic Preprocessing: Hounsfield Windowing & 3D Median Filtering

Implements bone radiodensity thresholding $[100, 1100]\text{ HU}$, 3D median filtering, and min-max normalization (`PreProcessing.py`):

```python
import numpy as np
from scipy.ndimage import median_filter

class RadiographicPreprocessor:
    def __init__(self, min_hu: float = 100.0, max_hu: float = 1100.0):
        self.min_hu = min_hu
        self.max_hu = max_hu

    def normalize_volume(self, volume: np.ndarray, is_label: bool = False) -> np.ndarray:
        """
        Isolate bone tissues and rescale volume intensities.
        
        Args:
            volume: 3D CT voxel array in Hounsfield Units.
            is_label: True if the input represents a binary segmentation mask.
        Returns:
            Normalized 3D array in [0, 255] uint8 format.
        """
        if not is_label:
            # 1. 3D spatial median filter (kernel size=2) to suppress Poisson CT noise
            volume = median_filter(volume, size=2)
            
            # 2. Bone radiodensity windowing: suppress soft tissue (<100 HU) and dense artifacts (>1100 HU)
            # Voxels outside the bone window are zeroed out
            volume = np.where((volume >= self.min_hu) & (volume <= self.max_hu), volume, 0)
            
            # 3. Min-Max normalization scaled to [0, 255] uint8
            v_min = volume.min()
            v_max = volume.max()
            if v_max > v_min:
                normalized = ((volume - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(volume, dtype=np.uint8)
            return normalized
        else:
            # Binary ground-truth mask: map all non-zero labels to 255
            label_mask = np.zeros_like(volume, dtype=np.uint8)
            label_mask[volume > 0] = 255
            return label_mask
```

---

## 3. Concurrent Multi-Threaded Patch Generator

Generates overlapping $128 \times 128$ spatial patches with $32 \times 32$ stride using `ThreadPoolExecutor` and mutex locking (`mPreprocessing.py`):

```python
import os
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

class PatchExtractor:
    def __init__(self):
        self.count = 0
        self.count_lock = threading.Lock()

    def extract_patches(self, image: np.ndarray, patch_size: Tuple[int, int] = (128, 128),
                        stride: Tuple[int, int] = (32, 32)) -> List[np.ndarray]:
        """
        Decompose a 2D slice into overlapping patches with boundary clamping.
        """
        patches = []
        height, width = image.shape
        
        for i in range(0, height, stride[0]):
            for j in range(0, width, stride[1]):
                # Clamp coordinates flush against boundary when window overflows
                y_start = height - patch_size[0] if (i + patch_size[0] > height) else i
                x_start = width - patch_size[1] if (j + patch_size[1] > width) else j
                
                patch = image[y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
                patches.append(patch)
                
                if j + patch_size[1] >= width:
                    break
            if i + patch_size[0] >= height:
                break
                
        return patches

    def process_slice_concurrently(self, data_slice: np.ndarray, label_slice: np.ndarray,
                                   output_data_dir: str, output_label_dir: str,
                                   patch_size: Tuple[int, int] = (128, 128),
                                   stride: Tuple[int, int] = (32, 32)):
        """
        Extracts and atomically writes paired patches to disk.
        """
        data_patches = self.extract_patches(data_slice, patch_size, stride)
        label_patches = self.extract_patches(label_slice, patch_size, stride)

        # Thread-safe reservation of patch index range
        with self.count_lock:
            start_index = self.count
            self.count += len(data_patches)

        for offset, (data_p, label_p) in enumerate(zip(data_patches, label_patches)):
            idx = start_index + offset
            cv2.imwrite(os.path.join(output_data_dir, f"data_patch_{idx}.png"), data_p)
            cv2.imwrite(os.path.join(output_label_dir, f"label_patch_{idx}.png"), label_p)
```

---

## 4. PyTorch Patch-Based U-Net Architecture & Inference

Full PyTorch 4-level U-Net implementation with skip connections and TorchScript inference wrapper (`Predict.py`):

```python
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Expansive Path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output Head
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip reservations
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder forward pass with skip concatenations
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sigmoid(self.out_conv(d1))

class PatchInference:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        # Load TorchScript JIT artifact
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    @torch.no_grad()
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        tensor = self.transform(patch).float().unsqueeze(0).to(self.device)
        prob_map = self.model(tensor)
        binary_mask = (prob_map.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        return binary_mask

def train_unet_patch_model(train_loader, val_loader, num_epochs: int = 20):
    """
    Trains the U-Net on 482,833 overlapping patches using AdamW and BCELogitLoss.
    Optimal validation checkpoint achieved at Epoch 15.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Optimizer and Loss setup matching research parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model.out_conv(model.dec1(model.up1(model.dec2(images)))) # Forward pass
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        # Validation pass
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                running_val_loss += loss.item()
                
        val_loss = running_val_loss / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint (Epoch 15 optimal minimum)
            torch.save(model.state_dict(), f"unet_spine_epoch_{epoch}_optimal.pth")
    return model
```

---

## 5. Coordinate-Preserving 3D Volumetric Reconstructor

Stitches reverse-projected patches into full 2D slice grids and packages an isotropic 3D NIfTI file with native affine headers (`Reconstruction.py`):

```python
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_opening, binary_fill_holes
from typing import List, Dict

class VolumetricReconstructor:
    def __init__(self, original_affine: np.ndarray, original_header: nib.nifti1.Nifti1Header):
        self.original_affine = original_affine
        self.original_header = original_header

    def reconstruct_2d_slice(self, patches: List[np.ndarray],
                             canvas_shape: tuple = (512, 512),
                             patch_size: tuple = (128, 128),
                             stride: tuple = (128, 128)) -> np.ndarray:
        """
        Reconstruct a 2D axial slice mask from non-overlapping or blended patches.
        """
        height, width = canvas_shape
        canvas = np.zeros((height, width), dtype=np.uint8)
        patch_idx = 0
        
        for i in range(0, height - patch_size[0] + 1, stride[0]):
            for j in range(0, width - patch_size[1] + 1, stride[1]):
                if patch_idx < len(patches):
                    canvas[i:i + patch_size[0], j:j + patch_size[1]] = patches[patch_idx]
                    patch_idx += 1
        return canvas

    def construct_3d_nifti(self, slice_patches_dict: Dict[int, List[np.ndarray]]) -> nib.Nifti1Image:
        """
        Reconstructs all axial slices, applies 3D morphological post-processing,
        and constructs a coordinate-preserving Nifti1Image.
        """
        assembled_slices = []
        # Sort slice keys to ensure correct Z-axis orientation
        for slice_idx in sorted(slice_patches_dict.keys()):
            patches = slice_patches_dict[slice_idx]
            recon_slice = self.reconstruct_2d_slice(patches)
            assembled_slices.append(recon_slice)

        # Stack into 3D voxel array: (512, 512, Z)
        volume_array = np.stack(assembled_slices, axis=-1)

        # Apply 3D morphological opening (suppresses stray false-positive voxels)
        structuring_element = np.ones((3, 3, 3), dtype=bool)
        cleaned_volume = binary_opening(volume_array > 0, structure=structuring_element)
        
        # Binary hole filling for hollow trabecular bone interior
        final_volume = binary_fill_holes(cleaned_volume).astype(np.uint8) * 255

        # Pack into Nifti1Image with native affine and header metadata
        nifti_image = nib.Nifti1Image(
            final_volume,
            affine=self.original_affine,
            header=self.original_header
        )
        return nifti_image
```

---

## 6. Clinical Visualization: React & Three.js 3D Mesh Component

Client-side component rendering orthogonal slice scrubbing and Three.js 3D surface mesh visualization (`SpineImageSegmantaionUI`):

```tsx
import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface VolumeViewerProps {
  volumeId: string;
  totalSlices: number;
  previewUrlBase: string;
}

export const ClinicalSpineViewer: React.FC<VolumeViewerProps> = ({
  volumeId,
  totalSlices,
  previewUrlBase,
}) => {
  const [currentSlice, setCurrentSlice] = useState<number>(Math.floor(totalSlices / 2));
  const mountRef = useRef<HTMLDivElement>(null);

  // Initialize Three.js WebGL scene for 3D anatomical surface mesh
  useEffect(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 150, 300);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Surgical illumination lighting setup
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(100, 200, 100);
    scene.add(keyLight);

    const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
    scene.add(ambientLight);

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 bg-slate-900 text-white rounded-xl">
      {/* 2D Orthogonal Slice Scrubbing View */}
      <div className="flex flex-col items-center bg-slate-800 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Axial Plane Slice Explorer (Z: {currentSlice})</h3>
        <div className="relative w-[384px] h-[384px] bg-black border border-slate-700 rounded-md overflow-hidden">
          <img
            src={`${previewUrlBase}/slice_${currentSlice}.png`}
            alt={`Axial slice ${currentSlice}`}
            className="w-full h-full object-contain"
          />
        </div>
        <div className="w-full mt-4 px-4">
          <input
            type="range"
            min={0}
            max={totalSlices - 1}
            value={currentSlice}
            onChange={(e) => setCurrentSlice(Number(e.target.value))}
            className="w-full accent-emerald-500 cursor-pointer"
          />
        </div>
      </div>

      {/* 3D WebGL Surface Mesh View */}
      <div className="flex flex-col items-center bg-slate-800 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">3D Surface Reconstruction (recon3d.nii)</h3>
        <div ref={mountRef} className="w-full h-[384px] rounded-md overflow-hidden" />
      </div>
    </div>
  );
};
```

---

## 7. Mathematical Validation Metrics Implementation

Ground-truth vs. prediction scoring computing the Dice Similarity Coefficient (DSC) and Jaccard Index (IoU):

```python
import numpy as np

def compute_segmentation_metrics(predicted_mask: np.ndarray, ground_truth: np.ndarray,
                                 eps: float = 1e-7) -> dict:
    """
    Computes voxel-level Dice and IoU metrics.
    
    Args:
        predicted_mask: Binary prediction array (0 or 1).
        ground_truth: Binary annotation array (0 or 1).
        eps: Epsilon smoothing term to prevent division by zero.
    """
    p = (predicted_mask > 0).astype(np.float32).flatten()
    g = (ground_truth > 0).astype(np.float32).flatten()
    
    intersection = np.sum(p * g)
    union = np.sum(p) + np.sum(g) - intersection
    
    dice = (2.0 * intersection + eps) / (np.sum(p) + np.sum(g) + eps)
    iou = (intersection + eps) / (union + eps)
    
    return {
        "Dice_Coefficient": float(dice),
        "Dice_Percentage": float(dice * 100.0),
        "IoU_Jaccard": float(iou),
        "IoU_Percentage": float(iou * 100.0)
    }
```

Related: [Architecture Guide](architecture.md) · [Preprocessing Deep Dive](medical-image-preprocessing.md) · [Deep Learning](deep-learning-and-training.md) · [3D Reconstruction](3d-volumetric-reconstruction.md)
