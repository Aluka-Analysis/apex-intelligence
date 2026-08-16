"""
Diagnostic script for MiniFASNet anti-spoofing model.
Tests different normalization methods to find which one works.
"""

import cv2
import numpy as np
import onnxruntime as ort
import sys
sys.path.append('.')
from ai.recognition.detector import FaceDetector

# Load model
session = ort.InferenceSession(
    'models/antispoof/minifasnet_v2.onnx',
    providers=['CPUExecutionProvider']
)
detector = FaceDetector()
cap = cv2.VideoCapture(0)

print('Opening camera — press SPACE to capture — Q to quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow('Diagnose', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):
        face = detector.detect_largest_face(frame)
        if face is not None:
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(frame.shape[1], x2 + pad_x)
            y2 = min(frame.shape[0], y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (80, 80))

            print('\n--- Testing different normalizations ---')

            # Method 1: Simple /255
            img1 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img1 = np.expand_dims(img1.transpose(2, 0, 1), 0)
            out1 = session.run(None, {session.get_inputs()[0].name: img1})[0][0]
            exp1 = np.exp(out1 - out1.max())
            p1 = exp1 / exp1.sum()
            print(f'Method 1 /255:            raw={out1}  probs={p1.round(4)}')

            # Method 2: ImageNet normalization
            img2 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img2 = (img2 - mean) / std
            img2 = np.expand_dims(img2.transpose(2, 0, 1).astype(np.float32), 0)
            out2 = session.run(None, {session.get_inputs()[0].name: img2})[0][0]
            exp2 = np.exp(out2 - out2.max())
            p2 = exp2 / exp2.sum()
            print(f'Method 2 ImageNet norm:   raw={out2}  probs={p2.round(4)}')

            # Method 3: Mean 0.5 std 0.5
            img3 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img3 = (img3 - 0.5) / 0.5
            img3 = np.expand_dims(img3.transpose(2, 0, 1).astype(np.float32), 0)
            out3 = session.run(None, {session.get_inputs()[0].name: img3})[0][0]
            exp3 = np.exp(out3 - out3.max())
            p3 = exp3 / exp3.sum()
            print(f'Method 3 mean0.5 std0.5:  raw={out3}  probs={p3.round(4)}')

            # Method 4: BGR no conversion
            img4 = crop.astype(np.float32) / 255.0
            img4 = np.expand_dims(img4.transpose(2, 0, 1), 0)
            out4 = session.run(None, {session.get_inputs()[0].name: img4})[0][0]
            exp4 = np.exp(out4 - out4.max())
            p4 = exp4 / exp4.sum()
            print(f'Method 4 BGR /255:        raw={out4}  probs={p4.round(4)}')

            print('\nClass indices: [0]=?, [1]=?, [2]=?')
            print('Look for which method gives highest score for YOUR REAL FACE')
        else:
            print('No face detected')

cap.release()
cv2.destroyAllWindows()