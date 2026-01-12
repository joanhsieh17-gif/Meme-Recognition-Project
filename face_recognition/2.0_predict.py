"""Run predictions with a trained SVM face recognition model."""

import argparse
import os
import pickle

import face_recognition

DEFAULT_MODEL_PATH = "../models/trained_svm_model_real25.pkl"
DEFAULT_IMAGE_PATH = "test_image.jpg"
DEFAULT_PREDICTION_THRESHOLD = 0.40


def load_model(model_path: str):
    with open(model_path, "rb") as model_handle:
        return pickle.load(model_handle)


def predict_names_multilabel(model, image_path: str, prediction_threshold: float):
    test_image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(test_image)
    num_faces = len(face_locations)
    print("Number of faces detected:", num_faces)

    predict_res = {
        "face_recg_pred": [],
        "face_recg_cand": [],
    }

    if not face_locations:
        print("No faces detected; nothing to predict.")
        return predict_res

    test_image_encodings = face_recognition.face_encodings(
        test_image, known_face_locations=face_locations
    )

    print("Found:")
    for index, test_image_enc in enumerate(test_image_encodings, start=1):
        probabilities = model.predict_proba([test_image_enc])[0]
        entries = [
            {
                "bbox_index": index,
                "pred_name": name,
                "pred_prob": prob,
            }
            for name, prob in zip(model.classes_, probabilities)
        ]

        sorted_entries = sorted(entries, key=lambda item: item["pred_prob"], reverse=True)

        for entry in sorted_entries:
            if entry["pred_prob"] >= prediction_threshold:
                predict_res["face_recg_pred"].append(entry)
            else:
                predict_res["face_recg_cand"].append(entry)

        formatted = ", ".join(
            f"{item['pred_name']} ({item['pred_prob']:.2%})" for item in sorted_entries
        )
        print(f"Face {index}: {formatted}")

    return predict_res


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict faces in an image using a trained SVM model."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained SVM model file.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image containing unknown faces.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help="Minimum probability required to report a prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Train the model before predicting."
        )

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}.")

    model = load_model(args.model_path)
    predict_res = predict_names_multilabel(model, args.image, args.threshold)
    return predict_res


if __name__ == "__main__":
    main()

