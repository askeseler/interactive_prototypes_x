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

from collections import namedtuple
args = namedtuple("args","debug dataset")(debug=True, dataset = "~/datasets/cub200/images/")

hash = "01C28165A164DEEE.pth"
root = "prototype_refinement_experiments/" + hash + " / " + datetime.datetime.now().strftime("%Y_%m_%d")

def main_refinement_by_pixel_distance(args):
    logger = SummaryWriter("./tensorboard/" + root)
    base_architecture = 'resnet34'
    dset = CUB200(debug=args.debug, with_ground_truth=True, data_path=args.dataset)

    train_push_loader = dset.train_push_dataloader()
    features = DeepEncoder(base_architecture, prototype_shape)
    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                num_classes=num_classes, training_phase="warm",
                push_dataloader=train_push_loader, logger=logger)
    #ppnet.deselection_loss = loss_function
    trainer1 = Trainer(gpus=1, max_epochs=5, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    deselection_trainer = Trainer(gpus=1, max_epochs=100, progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    ppnet.load_state_dict(torch.load("../weights/after_push_fix/" + hash), strict = False)
    ppnet.set_training_phase("refinement_by_pixel_distance")
    deselection_trainer.fit(ppnet, train_push_loader, dset.test_dataloader())