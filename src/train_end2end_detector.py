import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
sys.path.append("../")
from src.ml.end2end_builders.malconv_builder import MalConvBuilder
from src.ml.datasets.binary_builder import BinaryBuilder, SequentialFixedChunkAblationsBinaryBuilder
import os
import wandb
import time

#os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("model_name",
                        type=str,
                        help="Model's name. Choose one of the following: "
                             "1/ MalConv"
                        )
    parser.add_argument("hyperparameters_filepath",
                        type=str,
                        help="Hyperparameters of the model")
    parser.add_argument("goodware_filepath",
                        type=str,
                        help="Where the goodware is located")
    parser.add_argument("malware_filepath",
                        type=str,
                        help="Where the malware is located")
    parser.add_argument("training_goodware_subset_filepath",
                        type=str,
                        help="Training goodware subset filepath")
    parser.add_argument("training_malware_subset_filepath",
                        type=str,
                        help="Training malware subset filepath")
    parser.add_argument("validation_goodware_subset_filepath",
                        type=str,
                        help="Validation goodware subset filepath")
    parser.add_argument("validation_malware_subset_filepath",
                        type=str,
                        help="Validation malware subset filepath")
    parser.add_argument("--dataset_type",
                        type=str,
                        help="Type of binary dataset. Options:"
                             "1/ Vanilla"
                             "2/ Smoothing",
                        default="Vanilla")
    parser.add_argument("--output_filepath",
                        type=str,
                        help="Filepath to where want to store the results",
                        default=None)
    parser.add_argument('--max_len',
                        type=int,
                        default=16000000,
                        help='Maximum length of input file in bytes, at which point files will be truncated')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--model_checkpoint",
                        type=str,
                        help="Model weights",
                        default=None)
    parser.add_argument("--patience",
                        type=int,
                        help="Number of epochs with no improvement after which training will be stopped.",
                        default=5)
    parser.add_argument("--start_epoch",
                        type=int,
                        help="Start epoch when loading a pretrained model",
                        default=0)
    parser.add_argument("--padding_value",
                        type=float,
                        default=0.0,
                        help="Padding value. Either 0.0 or 256.0")
    parser.add_argument("--pretrained_emb",
                        type=str,
                        default=None,
                        help="Pretrained embedding weights")
    parser.add_argument("--goodware_metadata_filepath",
                        type=str,
                        help="Goodware metadata filepath",
                        default=None)
    parser.add_argument("--malware_metadata_filepath",
                        type=str,
                        help="Malware metadata filepath",
                        default=None)
    parser.add_argument("--chunk_size",
                        type=int,
                        default=100000,
                        help="Minimum size of the chunk size")
    parser.add_argument("--ablations_generator_type",
                        type=str,
                        help="Must be one of the following: SequentialFixedChunkGenerator",
                        default="SequentialChunkGeneratror")
    parser.add_argument("--min_threshold",
                        type=float,
                        default=0.5,
                        help="Minimum percentage of chunks to be considered benign. If more than (min_threshold*100)% are malicious, then file is malicious")
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--no-sort', dest='sort_by_size', action='store_false')
    parser.set_defaults(sort_by_size=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name == "MalConv":
        print("Building MalConv")
        builder = MalConvBuilder()
    else:
        raise NotImplementedError

    model, hyperparameters = builder.build(
        args.hyperparameters_filepath,
        model_checkpoint=args.model_checkpoint,
        pretrained_emb=args.pretrained_emb,
        device=device,
        padding_idx=int(args.padding_value)
    )
    if args.ablations_generator_type is None:
        wandb.init(
            project="CertifiedRobustness_"+args.dataset_type + "_" + type(model).__name__,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "channels": hyperparameters["channels"],
                "window_size": hyperparameters["window_size"],
                "stride": hyperparameters["stride"],
                "layers": hyperparameters["layers"],
                "embed_size": hyperparameters["embed_size"],
                "padding_value": args.padding_value,
            },
        )
    else:
        wandb.init(
            project="CertifiedRobustness_" + args.dataset_type + "_" + args.ablations_generator_type + "_" + type(model).__name__,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "channels": hyperparameters["channels"],
                "window_size": hyperparameters["window_size"],
                "stride": hyperparameters["stride"],
                "layers": hyperparameters["layers"],
                "embed_size": hyperparameters["embed_size"],
                "padding_value": args.padding_value,
            },
        )

    loader_threads = max(multiprocessing.cpu_count() - 4, multiprocessing.cpu_count() // 2 + 1)
    if args.dataset_type == "Vanilla":
        print("Building binary dataloaders")
        training_dataset_builder = BinaryBuilder(
            args.goodware_filepath,
            args.malware_filepath,
            goodware_subset_filepath=args.training_goodware_subset_filepath,
            malware_subset_filepath=args.training_malware_subset_filepath,
            max_len=args.max_len,
            sort_by_size=args.sort_by_size,
            padding_value=args.padding_value,
            num_workers=loader_threads,
            batch_size=args.batch_size
        )
        validation_dataset_builder = BinaryBuilder(
            args.goodware_filepath,
            args.malware_filepath,
            goodware_subset_filepath=args.validation_goodware_subset_filepath,
            malware_subset_filepath=args.validation_malware_subset_filepath,
            max_len=args.max_len,
            sort_by_size=args.sort_by_size,
            padding_value=args.padding_value,
            num_workers=loader_threads,
            batch_size=args.batch_size
        )
    elif args.dataset_type == "Smoothing":
        print("Building chunk-based binary dataloaders")
        if args.ablations_generator_type == "SequentialFixedChunkGenerator":
            training_dataset_builder = SequentialFixedChunkAblationsBinaryBuilder(
                args.goodware_filepath,
                args.malware_filepath,
                goodware_subset_filepath=args.training_goodware_subset_filepath,
                malware_subset_filepath=args.training_malware_subset_filepath,
                max_len=args.max_len,
                sort_by_size=args.sort_by_size,
                padding_value=args.padding_value,
                chunk_size=args.chunk_size,
                num_workers=loader_threads,
                batch_size=args.batch_size,
                train_mode=True,
            )
            validation_dataset_builder = SequentialFixedChunkAblationsBinaryBuilder(
                args.goodware_filepath,
                args.malware_filepath,
                goodware_subset_filepath=args.validation_goodware_subset_filepath,
                malware_subset_filepath=args.validation_malware_subset_filepath,
                max_len=args.max_len,
                sort_by_size=args.sort_by_size,
                padding_value=args.padding_value,
                chunk_size=args.chunk_size,
                num_workers=1,
                batch_size=1,
                train_mode=True,
            )
        else:
            raise NotImplementedError
    else:
        raise Exception("Choose one of the implemented binary dataloaders")

    training_dataset, training_dataloader = training_dataset_builder.build()
    validation_dataset, validation_dataloader = validation_dataset_builder.build()

    base_name = "{}_channels_{}_filterSize_{}_stride_{}_embdSize_{}_padding_value_{}".format(
        type(model).__name__,
        hyperparameters["channels"],
        hyperparameters["window_size"],
        hyperparameters["stride"],
        hyperparameters["embed_size"],
        args.padding_value
    )
    if args.output_filepath is None:
        if not os.path.exists(base_name):
            os.makedirs(base_name)
        file_name = os.path.join(base_name, base_name)
        output_filepath = base_name

    else:
        if not os.path.exists(args.output_filepath):
            os.makedirs(args.output_filepath)
        file_name = os.path.join(args.output_filepath, base_name)
        output_filepath = args.output_filepath

    headers = ['epoch', 'train_acc', 'train_auc', 'train_loss', 'val_acc', 'val_auc', 'val_loss']

    csv_log_out = open(file_name + ".csv", 'w')
    csv_log_out.write(",".join(headers) + "\n")

    if model.out_size > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    validation_accuracies = []
    validation_losses = []
    best_loss = sys.maxsize
    best_epoch = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        preds = []
        truths = []

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        i = 0
        epoch_stats = {'epoch': epoch}
        model.train()
        start_time = time.time()
        for inputs, labels in tqdm(training_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, penultimate_activ, conv_active = model(inputs)
            if model.out_size > 1:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            with torch.no_grad():
                if model.out_size > 1:
                    _, predicted = torch.max(outputs.data, 1) # torch.max returns tuple with (values, indices)
                    preds.extend(F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
                else:
                    predicted = torch.round(torch.sigmoid(outputs)).int()
                    preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().ravel())
                truths.extend(labels.detach().cpu().numpy().ravel())

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            #print(loss, "{}/{}={}".format((predicted == labels).sum().item(), labels.size(0), (predicted == labels).sum().item()/labels.size(0)))

            wandb.log(
                {
                    "acc": (predicted == labels).sum().item() / labels.size(0),
                    "loss": loss.item()
                }
            )
            i += 1
        end_time = time.time()
        print("Total time in seconds: ", end_time-start_time)
        epoch_stats['train_acc'] = train_correct * 1.0 / train_total
        epoch_stats['train_auc'] = roc_auc_score(truths, preds)
        epoch_stats['train_loss'] = running_loss / train_total

        wandb.log(
            {
                'train_acc': train_correct * 1.0 / train_total,
                'train_auc': roc_auc_score(truths, preds),
                'train_loss': running_loss / train_total,
                "train_time": end_time - start_time
            }
        )

        # Save the model and current state!
        model_path = os.path.join(output_filepath, "epoch_{}.checkpoint".format(epoch))
        best_model_path = os.path.join(output_filepath, "best_epoch.checkpoint")

        mstd = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': mstd,
            'optimizer_state_dict': optimizer.state_dict(),
            'channels': hyperparameters["channels"],
            'filter_size': hyperparameters["window_size"],
            'stride': hyperparameters["stride"],
            'embd_dim': hyperparameters["embed_size"],
        }, model_path)

        # Test Set Eval
        model.eval()
        eval_train_correct = 0
        eval_train_total = 0
        running_loss = 0

        i = 0
        preds = []
        truths = []
        with torch.no_grad():
            for inputs, labels in tqdm(validation_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, penultimate_activ, conv_active = model(inputs)

                if model.out_size > 1:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    preds.extend(F.softmax(outputs, dim=-1).data[:, 1].detach().cpu().numpy().ravel())
                else:
                    #print(outputs.shape, labels.float().shape)
                    if args.dataset_type == "Vanilla":
                        loss = criterion(outputs, labels.float())
                    else:
                        loss = criterion(outputs, labels.float()[0])
                    predicted = torch.round(torch.sigmoid(outputs)).int()
                    preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().ravel())
                truths.extend(labels.detach().cpu().numpy().ravel())
                running_loss += loss.item()

                eval_train_total += labels.size(0)
                eval_train_correct += (predicted == labels).sum().item()

        val_loss = running_loss / eval_train_total
        val_accuracy = eval_train_correct * 1.0 / eval_train_total
        print("{}; Validation loss: {}; Validation accuracy: {}".format(epoch, val_loss, val_accuracy))

        epoch_stats['val_acc'] = val_accuracy
        epoch_stats['val_auc'] = roc_auc_score(truths, preds)
        epoch_stats['val_loss'] = val_loss

        wandb.log(
            {
                'val_acc': val_accuracy,
                'val_auc': roc_auc_score(truths, preds),
                'val_loss': val_loss
            }
        )
        csv_log_out.write(",".join([str(epoch_stats[h]) for h in headers]) + "\n")
        csv_log_out.flush()

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': mstd,
                'optimizer_state_dict': optimizer.state_dict(),
                'channels': hyperparameters["channels"],
                'filter_size': hyperparameters["window_size"],
                'stride': hyperparameters["stride"],
                'embd_dim': hyperparameters["embed_size"],
            }, best_model_path)

        # Check that the validation loss has been decreasing
        print(epoch, best_epoch, args.patience)
        if epoch - best_epoch >= args.patience:
            print(
                "The model hasn't improved for {} epochs. Stop training.\n Best loss: {}: Last epochs losses:{}".format(
                    args.patience, best_loss, validation_losses[-args.patience:]))
            break  # Stop training

        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

    csv_log_out.close()

