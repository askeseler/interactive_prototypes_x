import pandas as pd
import tqdm
import sys

sys.path.insert(0, '..')
from models import *
import datetime
from pytorch_lightning.callbacks import Callback

img_size = 224
prototype_shape = [2000, 128, 1, 1]
num_classes = 200
base_architecture = 'resnet34'
refresh_rate = 0  # Quiet or progressbar?


class LogRemainingBackgroundPrototypes(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        _, new_prototype_indices = pl_module.push(pl_module.push_dataloader, no_side_effects=True)
        groundtruth = np.array(pl_module.push_dataloader.dataset.groundtruth)
        background_foreground = np.array([groundtruth[i[0]][i[1], i[2]] for i in new_prototype_indices])
        remaining_background = np.sum(np.array(background_foreground) == pl_module.bg_thres_perc_fg)
        print("n_remaining_prototypes = " + str(remaining_background))
        pl_module.custom_logger.add_scalar("remaining_background", remaining_background, pl_module.current_epoch)

def check_same_prototypes(prototypes_before_deselection, prototypes_after_deselection):
    found = []
    for x in tqdm.tqdm(prototypes_after_deselection):
        identical = False
        for y in prototypes_before_deselection:
            if str(x) == str(y):
                identical = True
        found.append(identical)
    return found

def main_deselection_loss(args):
    """ Trains the network with a fixed encoder using the deselection loss based on the distance to the deselection points (previously background prototypes)"""
    hashs = list(os.listdir("../weights/after_push_fix"))
    hashs.sort()
    hash = hashs[args.weights_idx]
    print(hash)
    os.makedirs("results/deselection_loss_experiments/" + hash, exist_ok=True)

    loss_functions = ["deselection_by_prototype_distance_max"]  # ,"deselection_by_prototype_distance_mean"]
    columns = []
    for l in loss_functions:
        columns.append("before deselection")
        columns.append("after_deselection")

    df = pd.DataFrame(columns=columns, index=["Groundtruth = 0.0", "Groundtruth < .1", "Groundtruth < .5", "same as before", "accuracy"])
    dset = CUB200(debug=args.debug, with_ground_truth=True, data_path=args.dataset)
    for i, l in enumerate(loss_functions):
        df = deselection_loss_experiment(hash, l, 2 * i, df, dset, delete_first=args.delete_bg_prototypes, )

    if args.delete_bg_prototypes:
        df.to_csv("results/deselection_loss_experiments/" + hash + "/loss_function_results_delete_first.csv")
    else:
        df.to_csv("results/deselection_loss_experiments/" + hash + "/loss_function_results.csv")


def deselection_loss_experiment(hash, loss_function, exp_col, df, dset, delete_first=False, reinit_prototypes=False):
    if delete_first:
        root = "deselection_loss_delete_first/" + hash + " / " + datetime.datetime.now().strftime("%Y_%m_%d") + " / exp_" + loss_function
    else:
        root = "deselection_loss_experiments/" + hash + " / " + datetime.datetime.now().strftime("%Y_%m_%d") + " / exp_" + loss_function
    logger = SummaryWriter("./tensorboard/" + root)

    train_push_loader = dset.train_push_dataloader()
    features = DeepEncoder(base_architecture, prototype_shape)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                  num_classes=num_classes, training_phase="warm",
                  push_dataloader=train_push_loader, logger=logger)
    ppnet.deselection_loss = loss_function
    trainer1 = Trainer(gpus=1, max_epochs=5, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    deselection_trainer = Trainer(gpus=1, max_epochs=100, callbacks=[LogRemainingBackgroundPrototypes()], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)

    ppnet.load_state_dict(torch.load("../weights/after_push_fix/" + hash), strict = False)
    _ = ppnet.push(train_push_loader)
    if delete_first:
        ppnet.delete_prototypes(ppnet.background_prototype_indices)
        # Some command line output:
        _, new_prototype_indices = ppnet.push(ppnet.push_dataloader, no_side_effects=True)
        groundtruth = np.array(ppnet.push_dataloader.dataset.groundtruth)
        background_foreground = np.array([groundtruth[i[0]][i[1], i[2]] for i in new_prototype_indices])
        remaining_background = np.sum(np.array(background_foreground) <= ppnet.bg_thres_perc_fg)
        print("n_remaining_prototypes = " + str(remaining_background))
    elif reinit_prototypes:
        ppnet.reinit_prototypes(ppnet.background_prototype_indices)
    acc_before = trainer1.test(ppnet, dset.test_dataloader())[0]["accuracy"]
    background_foreground = ppnet.background_foreground_prototypes()
    df.values[0, exp_col] = np.sum(np.array(background_foreground) <= ppnet.bg_thres_perc_fg)
    df.values[1, exp_col] = np.sum(np.array(background_foreground) < 0.1)
    df.values[2, exp_col] = np.sum(np.array(background_foreground) < 0.5)
    ppnet.set_training_phase("deselection_1")

    prototypes_before_deselection = ppnet.prototype_indices[ppnet.background_foreground_prototypes() <= ppnet.bg_thres_perc_fg]
    deselection_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    acc_after = trainer1.test(ppnet, dset.test_dataloader())[0]["accuracy"]
    _ = ppnet.push(train_push_loader)

    outpath = "../weights/" + loss_function + "/" + hash
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    torch.save(ppnet.state_dict(), outpath)

    background_foreground = ppnet.background_foreground_prototypes()
    prototypes_after_deselection = ppnet.prototype_indices[ppnet.background_foreground_prototypes() <= ppnet.bg_thres_perc_fg]

    df.values[0, exp_col + 1] = np.sum(np.array(background_foreground) <= ppnet.bg_thres_perc_fg)
    df.values[1, exp_col + 1] = np.sum(np.array(background_foreground) < 0.1)
    df.values[2, exp_col + 1] = np.sum(np.array(background_foreground) < 0.5)

    same_as_before = np.sum(check_same_prototypes(prototypes_before_deselection, prototypes_after_deselection))
    df.values[3, exp_col + 1] = same_as_before
    df.values[4, exp_col] = acc_before
    df.values[4, exp_col + 1] = acc_after
    return df

def main_last_layer_after_deselection(args):
    root_in = "../weights/deselection_by_prototype_distance_max"
    root_out = "../weights/deselection_by_prototype_distance_max_fixed"
    os.makedirs(root_out, exist_ok=True)
    max_epochs = 500
    hashs = list(os.listdir(root_in))
    hashs.sort()
    hash = hashs[args.weights_idx]
    dset = CUB200(debug=args.debug, with_ground_truth=True, data_path=args.dataset)
    
    weights_after_deselection = os.path.join(root_in, hash)
    logger = SummaryWriter("../tensorboard/deselection_loss_experiments/last_layer_training/" + hash)
    train_push_loader = dset.train_push_dataloader()
    last_layer_trainer = Trainer(gpus=1, max_epochs=max_epochs, callbacks=[], progress_bar_refresh_rate=refresh_rate)

    features = DeepEncoder(base_architecture, prototype_shape)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                num_classes=num_classes,push_dataloader=train_push_loader, logger=logger,
                training_phase = "last_layer_fixing_after_masking")
    print("load")
    ppnet.load_state_dict(torch.load(weights_after_deselection), strict = False)
    print("push")
    _ = ppnet.push(train_push_loader)  # appends unwanted prototypes to deselection_vectors
    ppnet.delete_prototypes(ppnet.background_prototype_indices)
    print("start training")
    last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    torch.save(ppnet.state_dict(), os.path.join(root_out, hash))


