import torch
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0,'..')
from models import *

def fit_prototypes_and_predict(args):
    base_architecture = 'resnet34'
    no_bounding_boxes = True
    debug = args.debug
    if debug:
        print("DEBUGGING")
    refresh_rate = 1 if args.progressbar else 0
    img_size = 224
    prototype_shape = (2000, 128, 1, 1)
    num_classes = 200
    hash = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    if args.train_on_clever_hans:
        hash += "_train_on_clever_hans"
    hash += args.experiment_suffix

    logger = SummaryWriter("./tensorboard/" + "first_push_fix/" + hash)
    print(hash)

    features = DeepEncoder(base_architecture, prototype_shape)

    if args.dataset_clever_hans == "":
        dset = CUB200(data_path = args.dataset, debug=debug, with_ground_truth=True)
    else:
        if args.train_on_clever_hans:
            dset = PlantNet(data_path = args.dataset_clever_hans, debug=debug, with_ground_truth=True)
        else:
            dset = PlantNet(data_path = args.dataset, debug=debug, with_ground_truth=True)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                      num_classes=num_classes, training_phase="warm",
                      push_dataloader=dset.train_push_dataloader(), last_layer_optimizer_lr= args.last_layer_optimizer_lr,
                       logger=logger)

    warm_trainer = Trainer(gpus=1, max_epochs=5, callbacks=[], progress_bar_refresh_rate=refresh_rate)
    joint_trainer = Trainer(gpus=1, max_epochs=args.joint_epochs, callbacks=[], progress_bar_refresh_rate=refresh_rate)
    push_trainer = Trainer(gpus=1, max_epochs=10, callbacks=[], progress_bar_refresh_rate=refresh_rate)
    last_layer_trainer = Trainer(gpus=1, max_epochs=args.last_layer_epochs, callbacks=[], progress_bar_refresh_rate=refresh_rate)

    print("\n\n\n\n---------------- WARMING -----------------------------")
    ppnet.set_training_phase("warm")
    warm_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    _ = warm_trainer.test(ppnet, dset.test_dataloader())

    print("\n\n\n\n---------------- JOINT TRAINING -----------------------------")
    ppnet.set_training_phase("joint")
    joint_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    _ = joint_trainer.test(ppnet, dset.test_dataloader())


    print("\n\n\n\n---------------- PUSH -----------------------------")
    prototype_update, _ = ppnet.push(dset.train_push_dataloader(), no_side_effects=True)
    ppnet.prototype_vectors.data = torch.tensor(prototype_update, dtype=torch.float32).cuda()
    res = push_trainer.test(ppnet, dset.test_dataloader())[0]
    ppnet.custom_logger.add_scalar("push/accuracy_after_push", res["accuracy"], ppnet.current_global_epoch)
    ppnet.set_training_phase("last_layer")
    print("\n\n\n\n---------------- LAST LAYER -----------------------------")
    last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    ppnet.custom_logger.add_scalar("push/accuracy_after_fixing_of_last_layer", res["accuracy"], ppnet.current_global_epoch)


    print("\n\n\n\n------------------- SAVING -------------------------------")
    root_out = "../weights/after_push_fix"
    os.makedirs(root_out, exist_ok=True)
    filename = hash + ".pth"
    outpath = os.path.join(root_out, filename)
    torch.save(ppnet.state_dict(), outpath)

if __name__=="__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--progressbar', dest='progressbar', action='store_true', default=False)
    parser.add_argument('--dataset', action = 'store', type = str, help = 'path to dataset', default="~/datasets/cub200/images/")
    args = parser.parse_args()
    fit_prototypes_and_predict(args)