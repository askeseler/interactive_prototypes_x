import sys
import os
sys.path.insert(0,'..')
from models import *
import torch
import numpy as np

refresh_rate = 1
base_architecture = 'resnet34'
no_bounding_boxes = False
img_size = 224
prototype_shape = [2000, 128, 1, 1]
num_classes = 200
last_layer_trainer = Trainer(gpus=1, max_epochs = 20, callbacks=[], progress_bar_refresh_rate = refresh_rate, checkpoint_callback=False, logger=False)

def experiment_deselection_masking(ppnet, dset, out_root):
    """ Deselection of background prototypes as indicated by groundtruth"""
    accs_masking = []# Accuracies before, after masking, after last layer fixing
    background_foreground = ppnet.background_foreground_prototypes()
    idxs_background_prototypes = np.where(np.array(background_foreground) == 0)[0]
    accs_masking.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    ppnet.prototype_mask[idxs_background_prototypes] = 0
    accs_masking.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    ppnet.set_training_phase("last_layer_mask_fixing_0")
    last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    accs_masking.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    accs_masking = np.array(accs_masking)
    with open(os.path.join(out_root, 'accs_masking.npy'), 'wb') as f:
        np.save(f, accs_masking)
    fig, ax = plt.subplots(1, dpi=200)
    ax.plot(accs_masking)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["before deselection", "after deselection", "after last layer training"])
    ax.set_ylabel("Test Accuracy")
    fig.savefig(os.path.join(out_root, "deselection_of_" + str(len(idxs_background_prototypes)) + "_change_in_accuracy.png"))
    print("outroot")
    print("The number of detected background prototypes is:")
    print(len(idxs_background_prototypes), end = "\n\n")

def experiment_deselection_deleting(ppnet, dset, out_root):
    accs = []
    background_foreground = ppnet.background_foreground_prototypes()
    idxs_background_prototypes = np.where(np.array(background_foreground) == 0)[0]
    accs.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    ppnet.delete_prototypes(ppnet.background_prototype_indices)
    accs.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    ppnet.set_training_phase("last_layer_mask_fixing_0")
    last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    accs.append(last_layer_trainer.test(ppnet, dset.test_dataloader())[0]["accuracy"])
    accs = np.array(accs)
    with open(os.path.join(out_root, 'accs_masking.npy'), 'wb') as f:
        np.save(f, accs)
    fig, ax = plt.subplots(1, dpi=200)
    ax.plot(accs)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["before deselection", "after deselection", "after last layer training"])
    ax.set_ylabel("Test Accuracy")
    fig.savefig(os.path.join(out_root, "deselection_of_" + str(len(idxs_background_prototypes)) + "_change_in_accuracy.png"))
    print("The number of detected background prototypes is:")
    print(len(idxs_background_prototypes), end = "\n\n")

def main_deselection(args, models_root = "../weights/after_push_fix/", deselection_by = "masking"):
    print("DATASET")
    print(args.dataset)
    dset = CUB200(debug = False, with_ground_truth = True, data_path = args.dataset)
    saved_models = os.listdir(models_root)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figs_deselection_masking", exist_ok=True)

    for path in saved_models:
        out_root = os.path.join("results", "figs_deselection_masking", deselection_by, path.split(".")[0])
        if os.path.isdir(out_root) and not(args.override):
            print("Skipping. Results already exist for:")
            print(out_root)
            continue
        os.makedirs(out_root, exist_ok=True)

        train_push_loader = dset.train_push_dataloader()
        logger = SummaryWriter("./tensorboard/deselection/" + deselection_by + "/" + path.split(".")[0])
        features = DeepEncoder(base_architecture, prototype_shape)
        ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                      num_classes=num_classes, training_phase="warm",
                      push_dataloader=train_push_loader, logger=logger)
        ppnet.load_state_dict(torch.load(os.path.join(models_root, path)), strict = False)
        ppnet.freeze()
        _ = ppnet.push(train_push_loader)
        histogram_all_prototypes, histogram_no_background_prototypes = ppnet.plot_histograms_bgfg()
        np.random.seed(9)
        examples_background_prototypes = ppnet.plot_background_prototypes(nx = 5, ny = 4, heatmap_threshold = .075)

        histogram_weights = ppnet.plot_weights_of_prototypes()
        pdf_weights = ppnet.plot_pdf_background_foreground_prototypes()
        histogram_all_prototypes.savefig(os.path.join(out_root, "histogram_all_prototypes.png"))
        histogram_no_background_prototypes.savefig(os.path.join(out_root, "histogram_no_background_prototypes.png"))
        examples_background_prototypes.savefig(os.path.join(out_root, "examples_background_prototypes.png"))
        histogram_weights.savefig(os.path.join(out_root, "histogram_weights.png"))
        pdf_weights.savefig(os.path.join(out_root, "pdf_weights.png"))
        ppnet.unfreeze()
        if not args.plots_only:
            if deselection_by == "masking":
                experiment_deselection_masking(ppnet, dset, out_root)
            elif deselection_by == "deleting":
                experiment_deselection_deleting(ppnet, dset, out_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--plots_only', dest='plots_only', action='store_true', default=False)
    parser.add_argument('--override', dest='override', action='store_true', default=False)
    args = parser.parse_args()
    main_deselection(args)
