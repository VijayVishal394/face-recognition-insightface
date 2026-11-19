"""Face recognition with InsightFace.

1. Put photos of people you know in known_faces/<person_name>/ (use your own
   photos, or photos of people who have given consent).
2. Run on a webcam or a recorded video. Known faces are labelled; others are "Unknown".
"""
import argparse
import glob
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def parse_args():
    p = argparse.ArgumentParser(description="InsightFace face recognition")
    p.add_argument("--source", default="0",
                   help="0 = webcam (or another camera index), or a path to a video file")
    p.add_argument("--known", default="known_faces",
                   help="folder with one sub-folder of photos per person")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="minimum similarity to accept a match (tune on your own data)")
    p.add_argument("--gpu", action="store_true", help="use GPU 0 instead of CPU")
    p.add_argument("--save", default=None, help="optional path to save the annotated video")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    return p.parse_args()


def face_area(face):
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def load_known_faces(app, folder):
    """Return {name: embedding}. One averaged embedding per person."""
    known = {}
    if not os.path.isdir(folder):
        return known
    for person in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for path in glob.glob(os.path.join(person_dir, "*")):
            img = cv2.imread(path)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                print(f"No face found in {path}, skipping")
                continue
            embeddings.append(max(faces, key=face_area).normed_embedding)
        if embeddings:
            mean = np.mean(embeddings, axis=0)
            known[person] = mean / np.linalg.norm(mean)
            print(f"Loaded {person}: {len(embeddings)} photo(s)")
    return known


def match(embedding, known, threshold):
    """Compare a face embedding with every known one. Higher score = more similar."""
    best_name, best_score = "Unknown", 0.0
    for name, ref in known.items():
        score = float(np.dot(embedding, ref))  # cosine similarity (embeddings are normalised)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < threshold:
        best_name = "Unknown"
    return best_name, best_score


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if args.gpu else -1, det_size=(640, 640))

    known = load_known_faces(app, args.known)
    if not known:
        print("No known faces loaded. Everyone will be labelled Unknown.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for face in app.get(frame):
            name, score = match(face.normed_embedding, known, args.threshold)
            x1, y1, x2, y2 = face.bbox.astype(int)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("InsightFace recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
