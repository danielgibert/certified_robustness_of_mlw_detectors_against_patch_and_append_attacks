import argparse
import torch
import multiprocessing
from tqdm import tqdm
import sys
sys.path.append("../")
from src.ml.end2end_builders.malconv_builder import MalConvBuilder
from src.ml.datasets.binary_builder import BinaryBuilder, SequentialFixedChunkAblationsBinaryBuilder
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from src.ml.classifiers.smoothed_classifier import SmoothedClassifier

torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument("model_name",
                        type=str,
                        help="Model's name. Choose one of the following: [MalConv]")
    parser.add_argument("model_checkpoint",
                        type=str,
                        help="Checkpoint of the model")
    parser.add_argument("hyperparameters_filepath",
                        type=str,
                        help="Hyperparameters of the model")
    parser.add_argument("goodware_filepath",
                        type=str,
                        help="Where the goodware is located")
    parser.add_argument("malware_filepath",
                        type=str,
                        help="Where the malware is located")
    parser.add_argument("output_filepath",
                        type=str,
                        help="Filepath to where want to store the results")
    parser.add_argument('--max_len',
                        type=int,
                        default=16000000,
                        help='Maximum length of input file in bytes, at which point files will be truncated')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size')
    parser.add_argument("--goodware_subset_filepath",
                        type=str,
                        default=None,
                        help="Goodware subset filepath")
    parser.add_argument("--malware_subset_filepath",
                        type=str,
                        default=None,
                        help="Malware subset filepath")
    parser.add_argument("--goodware_metadata_filepath",
                        type=str,
                        default=None,
                        help="Goodware metadata filepath")
    parser.add_argument("--malware_metadata_filepath",
                        type=str,
                        default=None,
                        help="Malware metadata filepath")
    parser.add_argument("--padding_value",
                        type=float,
                        default=256.0,
                        help="Padding value. Either 0.0 or 256.0")
    parser.add_argument("--dataset_type",
                        type=str,
                        help="Type of binary dataset. Options:"
                             "1/ Vanilla"
                             "2/ Smoothing",
                        default="Vanilla")
    parser.add_argument("--chunk_size",
                        type=int,
                        default=500,
                        help="Minimum size of the chunk size")
    parser.add_argument("--ablations_generator_type",
                        type=str,
                        help="Must be one of the following: SequentialFixedChunkGenerator",
                        default="SequentialChunkGeneratror")
    parser.add_argument("--overlapping_percentage",
                        type=float,
                        default=0.0,
                        help="Overlapping percentage between chunks. Value between 0.0 and 1.0")
    parser.add_argument("--predictions_filepath",
                        type=str,
                        help="Filepath where the predictions for each sample will be stored",
                        default=None)
    parser.add_argument("--min_threshold",
                        type=float,
                        default=0.5,
                        help="Minimum percentage of chunks to be considered benign. If more than (min_threshold*100)% are malicious, then file is malicious")
    parser.add_argument('--sort-by-size', dest="sort_by_size", action="store_true")
    parser.add_argument('--no-sort-by-size', dest='sort_by_size', action='store_false')
    parser.set_defaults(sort_by_size=False)
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.model_name == "MalConv":
        print("Building MalConv")
        builder = MalConvBuilder()
    else:
        raise NotImplementedError

    model, hyperparameters = builder.build(
        args.hyperparameters_filepath,
        model_checkpoint=args.model_checkpoint,
        pretrained_emb=None,
        device=device,
        padding_idx=int(args.padding_value)
    )

    loader_threads = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
    if args.dataset_type == "Vanilla":
        print("Building binary dataloaders")
        dataset_builder = BinaryBuilder(
            args.goodware_filepath,
            args.malware_filepath,
            goodware_subset_filepath=args.goodware_subset_filepath,
            malware_subset_filepath=args.malware_subset_filepath,
            max_len=args.max_len,
            sort_by_size=args.sort_by_size,
            padding_value=args.padding_value,
            num_workers=1,
            batch_size=args.batch_size
        )
        test_dataset, test_dataloader = dataset_builder.build()
    elif args.dataset_type == "Smoothing":
        if args.ablations_generator_type == "SequentialFixedChunkGenerator":
            dataset_builder = SequentialFixedChunkAblationsBinaryBuilder(
                args.goodware_filepath,
                args.malware_filepath,
                goodware_subset_filepath=args.goodware_subset_filepath,
                malware_subset_filepath=args.malware_subset_filepath,
                max_len=args.max_len,
                sort_by_size=args.sort_by_size,
                padding_value=args.padding_value,
                chunk_size=args.chunk_size,
                num_workers=1,
                batch_size=1,
                train_mode=False
            )
        else:
            raise NotImplementedError
        test_dataset, test_dataloader = dataset_builder.build()

        model = SmoothedClassifier(
            model,
            dataset_builder.ablated_generator
        )

    else:
        raise Exception("Choose one of the implemented binary dataloaders")

    model.eval()
    test_correct = 0
    test_total = 0

    preds = []
    truths = []

    if args.predictions_filepath is not None:
        f = open(args.predictions_filepath, "w")

    i = 0
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            print("Inputs: ", inputs, inputs.shape)
            outputs, _, _ = model(inputs)
            #print("Input: {}; labels: {}; Output: {}".format(inputs.shape, labels.shape, outputs.shape))

            # Different from the SmoothedClassifier than from the Non-SmoothedClassifier
            if type(model) == SmoothedClassifier:
                y_probs, y_preds = model.predict_from_outputs(outputs)
                y_prob, y_pred = model.get_predicted_label(y_preds, min_threshold=args.min_threshold)
                #y_prob, y_pred = model.majority_vote(y_preds)
                if args.predictions_filepath is not None:
                    f.write("{},".format(inputs.shape[0]*inputs.shape[1]))
                    f.write("{},".format(labels[0]))
                    f.write(",".join([str(prob.numpy()) for prob in y_probs]))
                    f.write("\n")
            else: # Non-smoothed classifier
                y_prob, y_pred = model.predict_from_outputs(outputs)

            preds.append(y_pred)
            label = labels.detach().cpu().numpy().ravel()[0]
            print(label, y_pred)
            truths.append(label)
            test_total += labels.size(0)
            i += 1

    try:
        f.close()
    except Exception as e:
        print(e)

    end_time = time.time()
    test_acc = test_correct * 1.0 / test_total
    try:
        test_acc = accuracy_score(truths, preds)
        test_auc = roc_auc_score(truths, preds)
        test_precision = precision_score(truths, preds)
        test_recall = recall_score(truths, preds)
    except ValueError as e:
        print(e)
        test_auc = None
        test_precision = None
        test_recall = None

    print(np.array(truths).astype(int), np.array(preds).round().astype(int))
    cm = confusion_matrix(np.array(truths).astype(int), np.array(preds).round().astype(int), normalize=None)
    cm_normalized = confusion_matrix(np.array(truths).astype(int), np.array(preds).round().astype(int), normalize="true")

    print("Test accuracy: {}".format(test_acc))
    print("Test AUC score: {}".format(test_auc))
    print("Confusion matrix: {}".format(cm))
    print("Normalized confusion matrix: {}".format(cm_normalized))
    print("Prediction time: {}".format(end_time-start_time))

    with open(args.output_filepath, "w") as output_file:
        output_file.write("Test accuracy:{}\n".format(test_acc))
        output_file.write("Test precision:{}\n".format(test_precision))
        output_file.write("Test recall:{}\n".format(test_recall))
        output_file.write("Test AUC score:{}\n".format(test_auc))
        output_file.write("Confusion matrix:\n {}\n".format(cm))
        output_file.write("Normalized confusion matrix:\n {}\n".format(cm_normalized))
        output_file.write("Prediction time: {}".format(end_time-start_time))

