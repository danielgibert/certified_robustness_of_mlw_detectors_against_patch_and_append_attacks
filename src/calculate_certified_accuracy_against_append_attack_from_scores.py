import argparse
from tqdm import tqdm
import sys
import math
import numpy as np
import math
sys.path.append("../")
from src.ml.datasets.binary_dataset import BinaryDataset
from src.utils import count_benign_and_malicious_predictions, get_predicted_label


def is_certified(num_top_class, num_second_class, chunk_size, adversarial_payload_size):
    delta = math.ceil(adversarial_payload_size / chunk_size) + 1
    if num_top_class > num_second_class + delta:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate certified accuracy based on the smoothed classifiers\' predictions')
    parser.add_argument("goodware_filepath",
                        type=str,
                        help="Where the goodware is located")
    parser.add_argument("malware_filepath",
                        type=str,
                        help="Where the malware is located")
    parser.add_argument("goodware_subset_filepath",
                        type=str,
                        help="Goodware subset filepath")
    parser.add_argument("malware_subset_filepath",
                        type=str,
                        help="Malware subset filepath")
    parser.add_argument("predictions_filepath",
                        type=str,
                        help="Filepath where the predictions for each sample will be stored")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Where to store the certified accuracy results")
    parser.add_argument("--max_len",
                        type=int,
                        help="Maximum length of the executables",
                        default=2000000)
    parser.add_argument('--sort-by-size', dest="sort_by_size", action="store_true")
    parser.add_argument('--no-sort-by-size', dest='sort_by_size', action='store_false')
    parser.set_defaults(sort_by_size=False)
    parser.add_argument("--padding_value",
                        type=float,
                        default=256.0,
                        help="Padding value. Either 0.0 or 256.0")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=200,
                        help="Chunk size")
    parser.add_argument("--adversarial_payload_size",
                        type=int,
                        help="Adversarial payload size",
                        default=None)
    parser.add_argument("--adversarial_payload_percentage",
                        type=float,
                        help="Adversarial payload percentage",
                        default=None)
    args = parser.parse_args()

    print("Building binary dataloaders")
    dataset = BinaryDataset(
        args.goodware_filepath,
        args.malware_filepath,
        goodware_subset_filepath=args.goodware_subset_filepath,
        malware_subset_filepath=args.malware_subset_filepath,
        max_len=args.max_len,
        sort_by_size=args.sort_by_size,
        padding_value=args.padding_value
    )


    with open(args.predictions_filepath, "r") as predictions_file:
        predictions = []
        y_preds = []
        for line in predictions_file.readlines():
            line = [float(val) for val in line.strip().split(",")]

            predicted = [1 if val >= 0.5 else 0 for val in line[1:]]
            y_preds.append(get_predicted_label(predicted))
            predictions.append(line[1:])
    print(y_preds)

    with open(args.output_filepath, "w") as certified_file:
        certified_file.write("y_true,y_pred,is_certified\n")

        total_accuracy = 0
        total_certified_accuracy = 0
        for index in range(0, dataset.__len__()):
            x, y_true = dataset.__getitem__(index)
            print(index, x.shape[0], y_true[0])
            if args.adversarial_payload_percentage is not None:
                adversarial_payload_size = math.ceil(args.adversarial_payload_percentage * x.shape[0])
            else:
                adversarial_payload_size = args.adversarial_payload_size
            print(dataset.__len__(), len(y_preds))

            y_pred = y_preds[index]
            num_benign_chunks, num_malicious_chunks = count_benign_and_malicious_predictions(predictions[index])
            if y_true == y_pred:
                total_accuracy += 1
                if y_true == 0:
                    certified = is_certified(num_benign_chunks, num_malicious_chunks, args.chunk_size, adversarial_payload_size)
                else:
                    certified = is_certified(num_malicious_chunks, num_benign_chunks, args.chunk_size, adversarial_payload_size)
            else:
                certified = False
            total_certified_accuracy += certified
            certified_file.write("{},{},{}\n".format(y_true,y_pred,certified))

        certified_file.write("Accuracy: {}/{}={}\n".format(total_accuracy, dataset.__len__(), total_accuracy/dataset.__len__()))
        certified_file.write("Certified accuracy: {}/{}={}\n".format(total_certified_accuracy, dataset.__len__(), total_certified_accuracy/dataset.__len__()))

