import tqdm
import sys
sys.path.insert(0,'..')
from models import *
import datetime
from pytorch_lightning.callbacks import Callback
import pickle

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
        remaining_background = np.sum(np.array(background_foreground) <= pl_module.bg_thres_perc_fg)
        print(remaining_background)
        pl_module.custom_logger.add_scalar("remaining_background",remaining_background, pl_module.current_epoch)

def check_same_prototypes(prototypes_before_deselection, prototypes_after_deselection):
    found = []
    for x in tqdm.tqdm(prototypes_after_deselection):
        identical = False
        for y in prototypes_before_deselection:
            if str(x) == str(y):
                identical = True
        found.append(identical)
    return found

def main_deselection_training(args):
    iterative_deselection_training(args, args.n_deselections)

def iterative_deselection_training(args, repetitions = 5):
    if args.dataset_clever_hans == "":
        dset = CUB200(debug=args.debug, with_ground_truth = True, data_path = args.dataset)
    else:
        dset = PlantNet(data_path = args.dataset_clever_hans, debug=debug, with_ground_truth=True)
    hashs = list(os.listdir("../weights/after_push_fix"))
    hashs.sort()
    hash = hashs[args.weights_idx]
    print(hash)
    if args.reinit_prototypes:
        root = "iterative_deselection_reinit/" + hash
    else:
        root = "iterative_deselection/" + hash
    logger = SummaryWriter("./tensorboard/" + root)
    train_push_loader = dset.train_push_dataloader()

    features = DeepEncoder(base_architecture, prototype_shape)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                  num_classes=num_classes, training_phase="deselection_0",
                  push_dataloader=train_push_loader, logger=logger)
    ppnet.hyperparameters["deselection"]["last_layer"] = args.lr_deselection_last_layer
    ppnet.hyperparameters["deselection"]["prototype_vectors"] = args.lr_deselection_prototype_vectors
    ppnet.load_state_dict(torch.load("../weights/after_push_fix/" + hash), strict=False)
    ppnet.deselection_loss = "deselection_by_prototype_distance_max"
    ppnet.bg_thres_perc_fg = args.bg_thres_perc_fg

    for i in range(repetitions):
        # Train with deselection loss
        ppnet.set_training_phase("deselection_" + str(i))
        deselection_trainer = Trainer(gpus=1, max_epochs=30, callbacks=[LogRemainingBackgroundPrototypes()], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)

        #determine background prototypes and define deselection points accordingly
        _ = ppnet.push(train_push_loader)#appends unwanted prototypes to deselection_vectors
        if args.reinit_prototypes:# optional: randomly reassign those prototypes
            ppnet.reinit_prototypes(ppnet.background_prototype_indices)

        prototypes_before_deselection = ppnet.prototype_indices[ppnet.background_foreground_prototypes() == 0.0]
        print("In repetition " + str(i) + " there are ", end = "")
        print(np.sum(ppnet.background_foreground_prototypes() <= ppnet.bg_thres_perc_fg),end = " ")
        print("background prototypes before deselection")

        ppnet.custom_logger.add_scalar("deselection_points_per_repetition", len(ppnet.deselection_vectors), i)
        ppnet.custom_logger.add_scalar("background_prototypes_per_repetition", len(prototypes_before_deselection), i)
        deselection_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
        prototypes_after_deselection = ppnet.prototype_indices[ppnet.background_foreground_prototypes() <= args.bg_thres_perc_fg]

        print("In repetition " + str(i) + " thre are ", end = "")
        print(np.sum(ppnet.background_foreground_prototypes() <= ppnet.bg_thres_perc_fg), end = " ")
        print("background prototypes after deselection")

        same_as_before = np.sum(check_same_prototypes(prototypes_before_deselection, prototypes_after_deselection))
        ppnet.custom_logger.add_scalar("same_as_before_repetition", same_as_before, i)
    ppnet.custom_logger.add_scalar("background_prototypes_per_repetition", len(prototypes_after_deselection), repetitions)
    os.makedirs("../weights/iterative_deselection/", exist_ok = True)
    os.makedirs("../args/iterative_deselection/", exist_ok = True)
    torch.save(ppnet.state_dict(), "../weights/iterative_deselection/" + hash)
    with open("../args/iterative_deselection/" + hash + ".json", "w") as f:
        f.write(json.dumps(dict(args.__dict__)))

def main_last_layer_iterative_deselection(args, root_weights = "../weights/iterative_deselection/", max_epochs = 300):
    hashs = os.listdir(root_weights)#iterative_deselection/A42B47C9DC9647BF.pth...
    dset = CUB200(debug=args.debug, with_ground_truth = True, data_path = args.dataset)
    fix_last_layer(dset, root_weights, hashs[args.weights_idx], max_epochs, debug = args.debug)
    
def fix_last_layer(dset, root_weights, weights_pth, max_epochs = 1, debug = True):
    max_epochs = 1 if debug else max_epochs
    dict = torch.load(os.path.join(root_weights, weights_pth))
    logger = SummaryWriter("./tensorboard/iterative_deselection_last_layer/")
    train_push_loader = dset.train_push_dataloader()
    
    # Init model
    features = DeepEncoder(base_architecture, prototype_shape)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                  num_classes=num_classes, training_phase="last_layer_after_deselect",
                  push_dataloader=train_push_loader, logger=logger)
    ppnet.load_state_dict(dict, strict = False)

    ppnet.set_loss_function_weights(args.w_cluster, args.w_sep, args.w_l1, args.w_crossentropy, args.w_deselection)
    
    # Begin experiment
    last_layer_trainer = Trainer(gpus=1, max_epochs=max_epochs, callbacks=[], progress_bar_refresh_rate=refresh_rate)
    print("PUSH")
    _ = ppnet.push(train_push_loader)  # appends unwanted prototypes to deselection_vectors
    ppnet.delete_prototypes(ppnet.background_prototype_indices)
    ppnet.set_training_phase("last_layer_fixing_after_iterative_deselect")
    print("TRAIN")
    last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    
    # Save
    outroot = "../weights/fixing_after_iterative_deselection/"
    os.makedirs(outroot, exist_ok = True)
    print("SAVE")
    torch.save(ppnet.state_dict(), os.path.join(outroot, weights_pth))
