#!/usr/bin/env python3
"""
Annotate images using REAL bounding boxes from gt.csv
"""

import os
import csv
from PIL import Image, ImageDraw

BASE_DIR = os.path.expanduser("~/bee_monitoring/data")
GT_CSV = os.path.expanduser("~/Downloads/bee_demo_samples/gt.csv")
UNHEALTHY_IN = os.path.join(BASE_DIR, "demo_images", "unhealthy")
UNHEALTHY_OUT = os.path.join(BASE_DIR, "demo_images", "unhealthy_annotated")

os.makedirs(UNHEALTHY_OUT, exist_ok=True)

# Parse gt.csv
print("📖 Reading gt.csv annotations...")
annotations = {}

with open(GT_CSV, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        filename = parts[0]
        # Extract just the image filename (not the full path)
        img_name = os.path.basename(filename)
        
        num_varroa = int(parts[1])
        boxes = []
        
        # Parse bounding boxes (groups of 4 numbers: x1 y1 x2 y2)
        i = 2
        while i + 3 < len(parts):
            try:
                x1 = int(parts[i])
                y1 = int(parts[i+1])
                x2 = int(parts[i+2])
                y2 = int(parts[i+3])
                boxes.append((x1, y1, x2, y2))
                i += 4
            except:
                break
        
        annotations[img_name] = {
            'num_varroa': num_varroa,
            'boxes': boxes
        }

print(f"✓ Loaded {len(annotations)} annotations")

# Annotate images
print("\n🐝 Annotating unhealthy images with REAL varroa boxes...")

annotated_count = 0
for img_file in os.listdir(UNHEALTHY_IN):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(UNHEALTHY_IN, img_file)
    out_path = os.path.join(UNHEALTHY_OUT, img_file)
    
    # Check if we have annotations for this image
    if img_file not in annotations:
        print(f"⚠️ No annotations for {img_file}, skipping")
        continue
    
    anno = annotations[img_file]
    
    try:
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw REAL varroa bounding boxes (RED)
        for (x1, y1, x2, y2) in anno['boxes']:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
            try:
                draw.text((x1 + 2, y1 + 2), "Varroa", fill=(255, 0, 0), stroke_width=2, stroke_fill=(0, 0, 0))
            except:
                draw.text((x1 + 2, y1 + 2), "Varroa", fill=(255, 0, 0))
        
        img.save(out_path, quality=95)
        annotated_count += 1
        print(f"✓ {img_file}: {len(anno['boxes'])} varroa boxes")
        
    except Exception as e:
        print(f"✗ Error with {img_file}: {e}")

print(f"\n✅ Done! Annotated {annotated_count} images with REAL varroa locations")
print(f"   Output: {UNHEALTHY_OUT}")
