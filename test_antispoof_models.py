"""
Test both MiniFASNet models (V2 and V1SE) side by side.
Compares their performance on real faces vs photos.
"""

import cv2
import numpy as np
import onnxruntime as ort
import sys
sys.path.append('.')
from ai.recognition.detector import FaceDetector

# Load both models
print("Loading models...")
model1 = ort.InferenceSession('models/antispoof/MiniFASNetV2.onnx', providers=['CPUExecutionProvider'])
model2 = ort.InferenceSession('models/antispoof/MiniFASNetV1SE.onnx', providers=['CPUExecutionProvider'])
detector = FaceDetector()

print('Model 1 (V2) input shape:', model1.get_inputs()[0].shape)
print('Model 2 (V1SE) input shape:', model2.get_inputs()[0].shape)
print('Model 1 (V2) output shape:', model1.get_outputs()[0].shape)
print('Model 2 (V1SE) output shape:', model2.get_outputs()[0].shape)

def predict(session, crop):
    """Run inference on a face crop."""
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img.transpose(2, 0, 1), 0)
    out = session.run(None, {session.get_inputs()[0].name: img})[0][0]
    exp = np.exp(out - out.max())
    return exp / exp.sum()

cap = cv2.VideoCapture(0)
print('\n📸 Press SPACE to test — Q to quit')
print('Test 1: Your real face')
print('Test 2: Hold a photo of a face\n')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    cv2.imshow('Diagnose Both Models', frame)
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
            crop = cv2.resize(frame[y1:y2, x1:x2], (80, 80))

            # Get predictions
            p1 = predict(model1, crop)
            p2 = predict(model2, crop)
            combined = (p1 + p2) / 2.0

            print('\n' + '='*50)
            print('📊 RESULTS')
            print('='*50)
            print(f'Model 1 (MiniFASNetV2):   {p1.round(4)}')
            print(f'Model 2 (MiniFASNetV1SE): {p2.round(4)}')
            print(f'Combined average:          {combined.round(4)}')
            print('-'*50)
            print(f'Index [0] (Spoof): {combined[0]:.4f}')
            print(f'Index [1] (Unknown): {combined[1]:.4f}')
            print(f'Index [2] (Live): {combined[2]:.4f}')
            print('-'*50)
            print(f'🏆 Highest index: {combined.argmax()} = {combined.max():.4f}')
            
            if combined.argmax() == 2:
                print('✅ VERDICT: REAL FACE')
            elif combined.argmax() == 0:
                print('❌ VERDICT: SPOOF (Photo/Screen)')
            else:
                print('⚠️ VERDICT: UNCERTAIN')
            print('='*50)
        else:
            print('❌ No face detected')

cap.release()
cv2.destroyAllWindows()
print('\n✅ Test complete.')