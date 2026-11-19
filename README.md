# Face Recognition using InsightFace

Recognises people from a **live webcam** or a **recorded video** using the
InsightFace library (`buffalo_l` model pack).

## How it works
1. **Detect** each face in the frame.
2. **Align** it and convert it into a 512-number embedding (a face "fingerprint").
3. **Compare** that embedding with the embeddings of known people using cosine similarity.
4. If the best score is above the threshold, show the name; otherwise show `Unknown`.

## Setup
```bash
pip install -r requirements.txt
```
Model files download automatically the first time you run it.

Add photos of people you know, one folder per person:
```
known_faces/
  Vijay/
    photo1.jpg
    photo2.jpg
  Friend/
    photo1.jpg
```
Only use your own photos, or photos of people who have agreed to it. Photos are
git-ignored so they are never uploaded.

## Run
```bash
# webcam
python recognize_faces.py --source 0

# recorded video
python recognize_faces.py --source my_video.mp4

# tune the match threshold and save the result
python recognize_faces.py --source my_video.mp4 --threshold 0.45 --save output.mp4
```
Press `q` to quit.

## Limitations
- The default threshold (0.4) is a starting point. Tune it on your own photos.
- Accuracy drops with poor lighting, extreme angles, masks and very small faces.
- Face recognition raises privacy concerns: only use it with consent.
