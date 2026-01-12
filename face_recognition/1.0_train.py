"""Train an SVM classifier for face recognition."""

import argparse
import os
import pickle
import random

import face_recognition
from PIL import Image
from sklearn import svm
from tqdm import tqdm

DEFAULT_CACHE_PATH = "encoding_caches/train_encodings_cache_real.pkl"
DEFAULT_MODEL_PATH = "../models/trained_svm_model_real25.pkl"
DEFAULT_TRAIN_DIR = "../datasets/train_dir_real"
DEFAULT_MAX_IMAGES_PER_PERSON = 25


def load_cached_encodings(cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_handle:
            cache = pickle.load(cache_handle)
        encodings = cache.get("encodings", [])
        names = cache.get("names", [])
        if encodings and names:
            print("Loaded face encodings from cache")
        return encodings, names
    return [], []


def gather_training_data(train_dir: str, max_images_per_person: int, encodings, names):
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    people = [person for person in os.listdir(train_dir) if not person.startswith('.')]

    for person in tqdm(people, desc="People"):
        person_dir = os.path.join(train_dir, person)
        if not os.path.isdir(person_dir):
            continue

        pix = os.listdir(person_dir)
        visible_pix = [
            img
            for img in pix
            if not img.startswith('.') and os.path.isfile(os.path.join(person_dir, img))
        ]
        if max_images_per_person > 0 and len(visible_pix) > max_images_per_person:
            visible_pix = random.sample(visible_pix, max_images_per_person)

        for person_img in tqdm(visible_pix, desc=f"Images for {person}", leave=False):
            face_path = os.path.join(person_dir, person_img)
            face = face_recognition.load_image_file(face_path)
            face_bounding_boxes = face_recognition.face_locations(face)

            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                names.append(person)
            else:
                if not face_bounding_boxes:
                    print(person + "/" + person_img + " was skipped because no faces were detected")
                    continue

                print(person + "/" + person_img + " contains multiple faces; splitting")
                for index, face_location in enumerate(face_bounding_boxes, start=1):
                    top, right, bottom, left = face_location
                    face_image_array = face[top:bottom, left:right]
                    split_name = os.path.splitext(person_img)[0] + f"_face_{index}.jpg"
                    split_path = os.path.join(person_dir, split_name)

                    if not os.path.exists(split_path):
                        Image.fromarray(face_image_array).save(split_path)

                    face_enc = face_recognition.face_encodings(
                        face, known_face_locations=[face_location]
                    )
                    if face_enc:
                        encodings.append(face_enc[0])
                        names.append(person)
                    else:
                        print(split_name + " could not be encoded and was skipped")

    return encodings, names


def persist_encodings(cache_path: str, encodings, names) -> None:
    if encodings and names:
        with open(cache_path, "wb") as cache_handle:
            pickle.dump({"encodings": encodings, "names": names}, cache_handle)


def train_model(train_dir: str, cache_path: str, model_path: str, max_images_per_person: int):
    encodings, names = load_cached_encodings(cache_path)
    encodings, names = gather_training_data(
        train_dir, max_images_per_person, encodings, names
    )

    if not encodings or not names:
        raise RuntimeError("No face encodings available for training.")

    persist_encodings(cache_path, encodings, names)

    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(encodings, names)

    with open(model_path, "wb") as model_handle:
        pickle.dump(clf, model_handle)

    print(f"Saved trained model to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SVM face recognition model.")
    parser.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing training images organized by person.",
    )
    parser.add_argument(
        "--cache-path",
        default=DEFAULT_CACHE_PATH,
        help="Path to cache file for storing face encodings.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Destination path for the trained SVM model.",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_PERSON,
        help="Maximum number of images to use per person (<=0 to use all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    max_images = args.max_images_per_person if args.max_images_per_person >= 0 else -1
    train_model(
        train_dir=args.train_dir,
        cache_path=args.cache_path,
        model_path=args.model_path,
        max_images_per_person=max_images,
    )


if __name__ == "__main__":
    main()