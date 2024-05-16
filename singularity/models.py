import sys

from pytorch_lightning import Trainer
import copy
import warnings

warnings.filterwarnings("ignore")

import torch.utils.data
import numpy as np
import logging
# from settings import base_architecture, img_size, prototype_shape, num_classes
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets import *

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.mixture import GaussianMixture as GMM
import os
import random
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
def fig2rgb_array(fig):
    """ Converts a matplotlib figure to an rgb array such that it may be displayed as an ImageDisplay
    Args:
        fig: Matplotlib figure
    Returns:
        arr: Image of the plot in the form of a numpy array
    """
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

class DeepEncoder(pl.LightningModule):
    def __init__(self, base_architecture, prototype_shape):
        super(DeepEncoder, self).__init__()
        base_architecture_to_features = {'resnet18': resnet18_features,
                                         'resnet34': resnet34_features,
                                         'resnet50': resnet50_features,
                                         'resnet101': resnet101_features,
                                         'resnet152': resnet152_features,
                                         'densenet121': densenet121_features,
                                         'densenet161': densenet161_features,
                                         'densenet169': densenet169_features,
                                         'densenet201': densenet201_features,
                                         'vgg11': vgg11_features,
                                         'vgg11_bn': vgg11_bn_features,
                                         'vgg13': vgg13_features,
                                         'vgg13_bn': vgg13_bn_features,
                                         'vgg16': vgg16_features,
                                         'vgg16_bn': vgg16_bn_features,
                                         'vgg19': vgg19_features,
                                         'vgg19_bn': vgg19_bn_features}
        self.prototype_shape = list(prototype_shape)
        self.features = base_architecture_to_features[base_architecture](pretrained=True)
        self.define_addon_layers()
        self.initialize_weights()

    def define_addon_layers(self):
        features = self.features
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture not implemented')

        self.add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
            nn.Sigmoid()
        )

    def initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.add_on_layers(x)
        return x


class PPNet(pl.LightningModule):
    def __init__(self, features, img_size, prototype_shape, num_classes, training_phase,
                 push_dataloader=None, init_prototype_class_identity=True, push_preprocessing=None, logger=None,
                 last_layer_optimizer_lr = 1e-4):
        """ Initializes the ProtoPNet classification head
        Args:
            features: An encoder that maps images to the latent space (Lightning module).
            img_size: Size of the input images (height, width)
            prototype_shape: Shape of the prototype vetors. Shape = [num_prototypes, num_feature_maps_latent_layer, 1,1]
            num_classes: Number of classes to be predicted. Make sure that the number of prototypes as defined by prototype_shape is chosen such that num_prototypes % n_classes == 0.
            training_phase: Initial training phase
            push_dataloader: Dataloader for the push operations
            push_preprocessing: Preprocessing function for the data in push_dataloader. If provided it will be applied before the forward step.
            logger: Tensorboard writer.
        """

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.n_feature_maps_latent_layer = prototype_shape[1]
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.push_preprocessing = push_preprocessing

        # assert (self.num_prototypes % self.num_classes == 0)
        # a multihot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        if init_prototype_class_identity:
            num_prototypes_per_class = self.num_prototypes // self.num_classes
            for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        self.prototype_class_identity = nn.Parameter(self.prototype_class_identity, requires_grad=False)
        # make it a parameter such that its saved automatically

        # this has to be named features to allow the precise loading
        self.features = features
        # self.define_addon_layers()

        # Prototype layer
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.last_push_prototype_vectors = None
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # Last layer
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self.set_last_layer_incorrect_connection(incorrect_strength=-.5)  # init last layer

        self.set_loss_function_weights(w_cluster=.8, w_sep=-0.08, w_l1=1e-4, w_crossentropy=1, w_deselection = 50)

        # Init training methods
        # self.push_epochs = [i for i in range(10000) if i % 10 == 0]
        # self.ppnet_multi = ppnet_multi = torch.nn.DataParallel(self)
        self.set_training_phase(training_phase)
        self.joint_lr_step_size = 5
        self.scheduler_steps_at_push = 5
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.push_dataloader = push_dataloader

        self.current_warm_epoch = 0
        self.current_joint_epoch = 0
        self.current_push_epoch = 0
        self.current_global_epoch = -1  # Current global epoch is not affected by using another trainer

        self.prototype_indices = np.zeros((self.num_prototypes, 3), dtype=np.int32) - 1
        self.heatmaps = np.ndarray((self.num_prototypes, 7, 7))
        self.prototype_image_paths = np.ndarray(self.num_prototypes, dtype=object)
        self.prototype_groundtruth_paths = np.ndarray(self.num_prototypes, dtype=object)
        self.prototype_mask = torch.ones(self.num_prototypes, requires_grad=False)

        self.bg_thres_perc_fg = 0.0#Upper limit of foreground pixels allowed in background grid cells
        self.background_prototype_indices = None
        self.deselection_vectors = None
        self.record = True  # log when True
        self.custom_logger = logger
        self.training_batch_idx = None
        self.bg_prototypes_deleted = False

        self.protos_to_refine = None
        self.protos_to_refine_to_class = None

        self.hyperparameters = {}#TODO keep all parameters of the optimizers here
        self.hyperparameters["deselection"] = {}
        self.hyperparameters["deselection"]["prototype_vectors"] = 3e-3
        self.hyperparameters["deselection"]["last_layer"] = 1e-4

    def set_training_phase(self, training_phase):
        """ Setter method for the training phase. The training phase is a string used for optimizer_configuration, setting of requires_grad and logging.
         Args: Training phase. The training phase must be a string that contains either warm, joint, last_layer or deselection.
                                If any other string is used it must not contain these substrings."""
        self.training_phase = training_phase
        self.freeze_layers()

    def configure_optimizers(self):
        """ Configure optimizers for the Trainer. Uses self.step_scheduler()."""
        print("configure optimizers")
        if "warm" in self.training_phase:
            warm_optimizer_lrs = {'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
            warm_optimizer_specs = [{'params': self.features.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                                    {'params': self.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
                                    ]
            warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
            return warm_optimizer
        elif "joint" in self.training_phase or "push" in self.training_phase:
            joint_optimizer_lrs = {'features': 1e-4, 'add_on_layers': 3e-3, 'prototype_vectors': 3e-3}
            joint_optimizer_specs = [{'params': self.features.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},  # bias are now also being regularized
                                     {'params': self.features.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
                                     {'params': self.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
                                     ]
            joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
            joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=self.joint_lr_step_size, gamma=0.1)
            scheduler = {
                'scheduler': joint_lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
            self.step_scheduler(joint_lr_scheduler)
            return ([joint_optimizer], [scheduler])
        elif "last_layer" in self.training_phase:
            last_layer_optimizer_specs = [{'params': self.last_layer.parameters(), 'lr': self.last_layer_optimizer_lr}]
            return torch.optim.Adam(last_layer_optimizer_specs)
        elif "deselection" in self.training_phase:
            optimizer_specs = [{'params': self.last_layer.parameters(), 'lr': self.hyperparameters["deselection"]["last_layer"]},
                               {'params': self.prototype_vectors, 'lr': self.hyperparameters["deselection"]["prototype_vectors"]}]
            return torch.optim.Adam(optimizer_specs)
        elif "refinement" in self.training_phase:
            optimizer_specs = [{'params': self.last_layer.parameters(), 'lr': 1e-4},
                               {'params': self.prototype_vectors, 'lr': 3e-3}]
            return torch.optim.Adam(optimizer_specs)


    def step_scheduler(self, scheduler):
        """ The learning rate must be adjusted by stepping the scheduler manually
            such that training can be resumed when interrupted (new trainer and hence optimizer).
        Args:
            scheduler: Pytorch LearningRateScheduler
            """
        # optimizer step is executed after epoch. We want to do it also before first epoch (hence +1)
        # https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        for _ in range(self.current_push_epoch + self.current_joint_epoch + 1):
            # print("scheduler step")
            scheduler.step()

    def push_prototypes(self):
        """ Pushes prototypes to the closest patch encodings"""
        self.freeze()
        with torch.no_grad():
            self.prototypes_before_push = copy.deepcopy(self.prototype_vectors)
            prototype_update, _ = self.push(self.push_dataloader, no_side_effects=True)
            self.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
            self.prototypes_after_push = copy.deepcopy(self.prototype_vectors)
        self.unfreeze()

    def on_train_epoch_end(self):
        """ Increments epoch counters for warm, push and joint learning"""
        # Set current epoch for each training phase
        if "push" in self.training_phase:
            self.current_push_epoch += 1
        elif "joint" in self.training_phase:
            self.current_joint_epoch += 1
        elif "warm" in self.training_phase:
            self.current_warm_epoch += 1

    def on_epoch_start(self):
        """ Freeze according to training_epoch """
        self.freeze_layers()

    def freeze_layers(self):
        if "warm" in self.training_phase:
            self.warm_only()
        elif "joint" in self.training_phase:
            self.joint()
        elif "push" in self.training_phase:
            self.joint()
        elif "last_layer" in self.training_phase:
            self.last_only()
        elif "deselection" in self.training_phase:
            self.deselection()
        elif "refinement" in self.training_phase:
            self.refinement()

    def on_train_epoch_start(self):
        """ Increments current global epoch on training epoch start"""
        if self.record:
            self.current_global_epoch += 1

    def training_step(self, batch, batch_idx):
        """ Performs training step:
        Args:
            batch: training batch
            batch_idx: index of batch
        Returns:
            loss_and_performance: Dictionary with loss and performance metrics"""
        self.training_batch_idx = batch_idx
        loss_and_performance = self.train_or_test(batch, batch_idx)
        if batch_idx % 100 == 0:
            if "deselection_cost" in loss_and_performance.keys():
                print("deselection_cost", end=" ")
                print(loss_and_performance["deselection_cost"].item(), end="  ")
                print("accuracy", end=" ")
                print(loss_and_performance["accuracy"].item())
                # print("soft_constraint", end = " ")
                # print(loss_and_performance["soft_constraint"])
        return loss_and_performance

    def validation_step(self, batch, batch_idx):
        """ Performs test step:
        Args:
            batch: test batch
            batch_idx: index of batch
        Returns:
            loss_and_performance: Dictionary with loss and performance metrics"""
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """ Performs test step:
        Args:
            batch: test batch
            batch_idx: index of batch
        Returns:
            loss_and_performance: Dictionary with loss and performance metrics"""
        self.eval()
        loss_and_performance = self.train_or_test(batch, batch_idx)
        return loss_and_performance

    def validation_epoch_end(self, step_outputs):
        """ Trigger custom log on validation epoch end
        step_outputs: List of dicts returned by validation step"""

        self.custom_log(step_outputs, "val")

    def training_epoch_end(self, step_outputs):
        """ Trigger custom log on training epoch end
        step_outputs: List of dicts returned by training step"""
        self.custom_log(step_outputs, "train")

    def test_epoch_end(self, step_outputs):
        """ Trigger custom log on test epoch end
        step_outputs: List of dicts returned by test step"""
        logs = self.custom_log(step_outputs, "test", log=True)
        self.log_dict(logs)
        return logs

    def custom_log(self, step_outputs, train_or_val="train", log=True):
        """ Logs all keys and the corresponding averaged values for each dictionary in step_outputs.
        Args:
            step_outputs: A list of dictionaries all of which have the same keys
            train_or_val: Training phase defined as a string. Is used for logging.
            log: Log via self.custom_logger or average metrics only
        Returns:
            out: Dict with averaged values from step_outputs
        """
        out = {}
        for k in step_outputs[0].keys():
            out[k] = np.mean([i[k].item() for i in step_outputs])

        if log and self.record:
            for k, v in out.items():
                self.custom_logger.add_scalar(k + "/" + train_or_val + "/" + self.training_phase, v, self.current_global_epoch)
        return out

    def train_or_test(self, batch, batch_idx):
        """ Performs a training or test step
        Args:
            batch: batch of data
            batch_idx: index of current batch
        Returns:
            loss_dict: Dict with loss and performance metrics
            """
        input, target = batch[0].to(device), batch[1].to(device)
        output, min_distances, conv_features, distances = self.forward(input)
        heatmaps = None
        part_locs = None

        if len(batch) >= 6:
            part_locs = batch[5]
            heatmaps = self.distance_2_similarity(distances)

        loss_dict = self.loss_function(min_distances, target, output, part_locs, heatmaps)

        _, predicted = torch.max(output.data, 1)
        n_examples = target.size(0)
        accuracy = (predicted == target).sum() / n_examples
        loss_dict["accuracy"] = accuracy
        return loss_dict

    def _l2_convolution(self, x, prototype_vectors, ones):
        """ Applies self.prototype_vectors as l2-convolution filters on input x
        Args:
            x: The latent grid encoding
        Returns:
            distances: L2 distances between patch encodings (latent grid vectors) and self.prototype_vectors
        """
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def distance_2_similarity(self, distances):
        """ Prototype activation function to convert distances into similarity
        Args:
            Distances: Distances between grid vectors and prototypes
        Returns:
            Similarity: Similarity value
        """
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        ''' Forward step of the prototype classification head.
        Args:
            x: The batch of input images
        Returns:
            logits: Output logits
            min_distances: Distance between prototype vectors and image encodings where the distance is the minimal L2 distance to any patch encoding.
        '''
        batch_size = x.shape[0]
        conv_features = self.features(x)  # torch.Size([100, 128, 7, 7])
        distances = self._l2_convolution(conv_features, self.prototype_vectors, self.ones)  # torch.Size([100, 2000, 1, 1])
        min_distances = torch.squeeze(-F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3])))  # torch.Size([100, 2000])
        prototype_activations = self.distance_2_similarity(min_distances)  # torch.Size([100, 2000])#
        prototype_activations = prototype_activations * torch.unsqueeze(self.prototype_mask, 0).repeat((batch_size, 1)).to(device)
        logits = self.last_layer(prototype_activations)  # torch.Size([100, 200])

        return logits, min_distances, conv_features, distances

    def push_forward(self, x):
        ''' Forward step for the push method. Computes the .
        Args:
            x: The batch of input images
        '''
        with torch.no_grad():
            self = self.to(device)
            conv_output = self.features(x)
            distances = self._l2_convolution(conv_output, self.prototype_vectors, self.ones)
        return conv_output, distances

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        ''' Initializes the weights of the last layer. Neurons that connect prototypes activation of a class to the respective output neurons are initialized with one.
            The other neurons are initialized with incorrect_stength. The weights of the last layer are subject to the L1 term in the loss function that is minimized such that
            all weights not connecting the prototype activations to the respective output neuron (class) approach zero.
        Args:
            incorrect_strength: The scalar for initialization of weights between the prototype activation of a class and output neurons of another class.
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def warm_only(self):
        """ Defines the layers for which a gradient is computed in the warming phase """
        for p in self.features.features.parameters():
            p.requires_grad = False
        for p in self.features.add_on_layers.parameters():
            p.requires_grad = True
        self.prototype_vectors.requires_grad = True
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def joint(self):
        """ Defines the layers for which a gradient is computed in the joint learning phase """
        for p in self.features.features.parameters():
            p.requires_grad = True
        for p in self.features.add_on_layers.parameters():
            p.requires_grad = True
        self.prototype_vectors.requires_grad = True
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def last_only(self):
        """ Defines the layers for which a gradient is computed in the learning phase where the last layer is trained only"""
        for p in self.features.features.parameters():
            p.requires_grad = False
        for p in self.features.add_on_layers.parameters():
            p.requires_grad = False
        self.prototype_vectors.requires_grad = False
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def refinement(self):  # NEW NEW NEW
        """ Defines the layers for which a gradient is computed in the deselection phase"""
        for p in self.features.features.parameters():
            p.requires_grad = True
        for p in self.features.add_on_layers.parameters():
            p.requires_grad = True
        self.prototype_vectors.requires_grad = True
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def deselection(self):  # NEW NEW NEW
        """ Defines the layers for which a gradient is computed in the deselection phase"""
        for p in self.features.features.parameters():
            p.requires_grad = False
        for p in self.features.add_on_layers.parameters():
            p.requires_grad = False
        self.prototype_vectors.requires_grad = True
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def set_loss_function_weights(self, w_cluster=None, w_sep=None, w_l1=None, w_crossentropy=None, w_deselection = None):
        """ Setter function for the weights of the loss function
        Args:
            w_cluster: Weight of the cost that incentivizes good clustering
            w_sep: Weight of the cost that incentivizes the separation of clusters of different prototypes (/classes)
            w_l1: Weight of the cost that approaches zero when only the evidence for a prototype of a class is connected
                  with the respective output neurons (last layer weights)
        """
        self.w_cluster = w_cluster if w_cluster else self.w_cluster
        self.w_sep = w_sep if w_sep else self.w_sep
        self.w_l1 = w_l1 if w_l1 else self.w_l1
        self.w_crossentropy = w_crossentropy if w_crossentropy else self.w_crossentropy
        self.w_deselection = w_deselection

    def pixel_dists(self, proto_locs, part_locs):
        """ Computes the distance in physical space between the provided prototype locations and the ground truth part locations
        Args:
            proto_locs: Prototype Locations. Shape [n_prototypes, batch_size, 2]
            part_locs: Part Locations. Shape [n_parts, batch_size, 2].
            missing_values: Shape [batch_size, n_parts]
        Returns:
            mean_dists: Average distances between prototypes and parts in batch. Shape [n_parts, n_prototypes]
            dists: Shape [n_images, n_parts, n_prototypes]
        """
        missing_values = torch.all(torch.isnan(part_locs), axis=-1)
        # Repeat prototype locations for each part
        n_parts = 6
        proto_locs1 = torch.unsqueeze(proto_locs, 1)
        proto_locs1 = torch.Tensor.expand(proto_locs1, [proto_locs1.shape[0], n_parts,
                                                        proto_locs1.shape[2],proto_locs1.shape[3]])
        #proto_locs1.shape = torch.Size([2000, 6, 1000, 2])
        
        # Repeat part locations for each prototype
        n_prototypes = proto_locs1.shape[0]
        part_locs1 = torch.torch.unsqueeze(part_locs, 0)
        part_locs1 = torch.Tensor.expand(part_locs1, [n_prototypes, part_locs1.shape[1], part_locs1.shape[2], part_locs1.shape[3]])
        #part_locs1.shape = torch.Size([2000, 6, 1000, 2])

        #flatten all but last dimension (x and y)
        part_locs2 = part_locs1.reshape([part_locs1.shape[0]*part_locs1.shape[1]*part_locs1.shape[2], part_locs1.shape[3]])
        proto_locs2 = proto_locs1.reshape([proto_locs1.shape[0]*proto_locs1.shape[1]*proto_locs1.shape[2], proto_locs1.shape[3]])
        #part_locs2.shape = torch.Size([12000000, 2])

        dists = torch.nn.functional.pairwise_distance(proto_locs2.float(), part_locs2.float())
        dists = dists.reshape([proto_locs1.shape[0],proto_locs1.shape[1],proto_locs1.shape[2]]).permute(2,1,0)

        dists[missing_values] = torch.nan
        mean_dists = torch.nanmean(dists, axis = 0)

        return mean_dists, dists

    def heatmap_max_locs(self, heatmaps):
        """ Get indeces from heatmaps"""
        hm = heatmaps.reshape([heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2] * heatmaps.shape[3]])
        flat_idxs = torch.argmax(hm, dim = -1)
        ys = flat_idxs // 7
        xs = flat_idxs % 7
        idxs = torch.stack([ys, xs]).permute(2,1,0)
        return idxs

    def loss_function(self, min_distances, target, output, part_locs = None, heatmaps = None):
        """ Computes the loss function.
        Args:
            min_distances: The distances to the closest image patch for each prototype; shape = [batch_size, n_prototypes].
            target: Target labels; shape = [batch_size]
            output: Output logits
        Returns:
            loss: The loss for the current batch
        """
        max_dist = self.n_feature_maps_latent_layer  # as values are between zero and one
        cross_entropy = torch.nn.functional.cross_entropy(output, target)  # labels are provided as one class-integer per sample of the batch

        # calculate cluster cost: Are the encodings for the prototypes of the correct class far away from the prototypes of the class?
        prototypes_of_correct_class = torch.t(  # Look up of columns for each target (yields batch_size many multi hot vectors)
            self.prototype_class_identity[:, target]).cuda()  # the prototypes that are meant to represent this class; shape = [batch_size, n_prototypes]
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class,
                                          dim=1)  # mask with prototypes_of_correct_class; maximal closeness across relevant prototypes (At least one prototype shall be close)
        cluster_cost = torch.mean(max_dist - inverted_distances)  # closeness is goal -> invert sign of term; minimize cluster cost

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class  # shape = [batch_size, n_prototypes]
        closeness = max_dist - min_distances
        inverted_distances_to_nontarget_prototypes, _ = torch.max(closeness * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        # avg_separation_cost = torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
        # avg_separation_cost = torch.mean(avg_separation_cost)

        l1_mask = 1 - torch.t(self.prototype_class_identity).cuda()
        l1 = (self.last_layer.weight * l1_mask).norm(p=1)

        if "deselection" in self.training_phase:
            #unwanted_prototypes = self.background_prototype_indices  # shape = [n_prototypes];
            assert type(self.last_push_prototype_vectors) != type(None)
            unwanted = torch.squeeze(self.deselection_vectors)
            prototypes = torch.squeeze(self.prototype_vectors)

            if self.deselection_loss == "deselection_by_prototype_distance_max":
                max_similarity = self.distance_2_similarity(torch.tensor(0))
                deselection_cost = torch.max(self.distance_2_similarity(torch.cdist(unwanted, prototypes, p=2))) / max_similarity  # even the most similar one should not be similar
            elif self.deselection_loss == "deselection_by_prototype_distance_mean":
                deselection_cost = torch.mean(self.distance_2_similarity(torch.cdist(unwanted, prototypes, p=2)))
            elif self.deselection_loss == "deselection_by_prototype_distance_min":
                deselection_cost = torch.min(self.distance_2_similarity(torch.cdist(unwanted, prototypes, p=2)))

            soft_constraint = torch.sum(torch.squeeze(self.prototype_vectors)[torch.squeeze(self.prototype_vectors) > 1.0] - 1)
            soft_constraint -= torch.sum(torch.squeeze(self.prototype_vectors)[torch.squeeze(self.prototype_vectors) < 0.0])
            loss = self.w_crossentropy * cross_entropy + self.w_l1 * l1 + self.w_deselection * deselection_cost + soft_constraint

            if self.current_epoch == 0 and not self.bg_prototypes_deleted and self.training_batch_idx:  # Additional logging for first epoch
                background_prototypes = torch.squeeze(self.prototype_vectors[self.background_prototype_indices])
                foreground_prototypes = torch.squeeze(self.prototype_vectors[self.foreground_prototype_indices])

                bg_proto_dist_to_deselection_points = torch.cdist(unwanted, background_prototypes)
                min_bg_proto_dist_to_deselection_points = torch.min(bg_proto_dist_to_deselection_points)
                max_bg_proto_dist_to_deselection_points = torch.max(bg_proto_dist_to_deselection_points)
                fg_proto_dist_to_deselection_points = torch.cdist(unwanted, foreground_prototypes)
                min_fg_proto_dist_to_deselection_points = torch.min(fg_proto_dist_to_deselection_points)
                max_fg_proto_dist_to_deselection_points = torch.max(fg_proto_dist_to_deselection_points)
                self.custom_logger.add_scalar("first_epoch/min_bg_proto_dist_to_deselection_pts", min_bg_proto_dist_to_deselection_points, self.training_batch_idx)
                self.custom_logger.add_scalar("first_epoch/max_bg_proto_dist_to_deselection_pts", max_bg_proto_dist_to_deselection_points, self.training_batch_idx)
                self.custom_logger.add_scalar("first_epoch/min_fg_proto_dist_to_deselection_pts", min_fg_proto_dist_to_deselection_points, self.training_batch_idx)
                self.custom_logger.add_scalar("first_epoch/max_fg_proto_dist_to_deselection_pts", max_fg_proto_dist_to_deselection_points, self.training_batch_idx)
                self.custom_logger.add_scalar("first_epoch/deselection_cost", deselection_cost, self.training_batch_idx)

            return {"loss": loss, "cross_entropy": cross_entropy, "cluster_cost": cluster_cost, "separation_cost": separation_cost, "l1": l1, "deselection_cost": deselection_cost,
                    "soft_constraint": soft_constraint}
        elif "refinement_by_pixel_distance" in self.training_phase:
            # A subset of the prototypes of each class should be close to one of the parts
            if type(self.protos_to_refine) == type(None):
                self.protos_to_refine = torch.Tensor([np.arange(6) + x * 10 for x in range(200)]).flatten().type(torch.long)#TODO parameterize
            if type(self.protos_to_refine_to_class) == type(None):
                self.protos_to_refine_to_class = torch.vstack([torch.eye(6,6) for _ in range(200)]).to(device)
            heatmaps = heatmaps[:,self.protos_to_refine]

            proto_locs = self.heatmap_max_locs(heatmaps)
            mean_dists, dists = self.pixel_dists(proto_locs, part_locs)#shape=[6,1200]
            mean_dists = mean_dists[torch.where(self.protos_to_refine_to_class.T)]

            mean_dists = torch.mean(mean_dists)
            loss = mean_dists + self.w_crossentropy * cross_entropy + self.w_cluster * cluster_cost + self.w_sep * separation_cost + self.w_l1 * l1
            return {"loss": loss, "cross_entropy": cross_entropy, "cluster_cost": cluster_cost, "separation_cost": separation_cost, "l1": l1, "mean_dists":mean_dists}
        else:
            loss = self.w_crossentropy * cross_entropy + self.w_cluster * cluster_cost + self.w_sep * separation_cost + self.w_l1 * l1
            return {"loss": loss, "cross_entropy": cross_entropy, "cluster_cost": cluster_cost, "separation_cost": separation_cost, "l1": l1}

    def background_score_prototypes(self, dataloader, logscale=False):
        """ Computes how much evidence for each prototype there is in areas of the images that are labeled as background.
            Evidence is inferred by upscaling the inverse distance from the latent feature maps.
        Args:
            dataloader: Dataloader with ground truth
            logscale: Transfrom distances to similarity before scoring
        """
        self.freeze()
        background_score = torch.zeros(2000)
        for push_iter, batch in enumerate(dataloader):
            imgs, ground_truth = batch[0].to(device), batch[4]
            ground_truth = torch.unsqueeze(ground_truth, 1)
            ground_truth = ground_truth.repeat(1, 2000, 1, 1)
            features, distances = self.push_forward(imgs)
            if logscale:
                distances = torch.log((distances + 1) / (distances + 0.0001))
            prototype_background_evidence = distances.cpu().detach() * (1 - ground_truth)
            background_score += torch.mean(prototype_background_evidence, (0, 2, 3))  # mean
        background_score /= len(dataloader)
        self.unfreeze()
        return background_score

    def background_foreground_prototypes(self):
        """ Retrieves the type of the prototype (background/foreground) of the prototype in pixel space (type of corresponding image patch after push)
        Returns:
            background_foreground: 1D numpy array of floats (range 0,1). Each scalar indicates the fraction of foreground pixels in the upscaled image patch.
                                    Zeros indicate that no pixel in the ground truth of the corresponding upscaled image patch is foreground.
                                    A value of one indicates that all pixels in the upscaled latent patch of the prototype are foreground.
            """
        groundtruth = np.array(self.push_dataloader.dataset.groundtruth)
        background_foreground = np.array([groundtruth[i[0]][i[1], i[2]] for i in self.prototype_indices])
        return background_foreground

    def set_unwanted_prototypes(self):
        groundtruth = np.array(self.push_dataloader.dataset.groundtruth)
        background_foreground = np.array([groundtruth[i[0]][i[1], i[2]] for i in self.prototype_indices])
        self.background_prototype_indices = torch.tensor(background_foreground <= self.bg_thres_perc_fg).type(torch.bool).cuda()
        self.foreground_prototype_indices = torch.tensor(background_foreground > self.bg_thres_perc_fg).type(torch.bool).cuda()
        self.full_foreground_prototype_indices = torch.tensor(background_foreground == 1).type(torch.bool).cuda()

        # Append deselection points (current unwanted prototypes to self.prototype_vectors)
        if type(self.deselection_vectors) == type(None):
            self.deselection_vectors = torch.nn.Parameter(self.last_push_prototype_vectors[self.background_prototype_indices].detach())
        else:
            self.deselection_vectors = torch.cat((self.deselection_vectors, self.last_push_prototype_vectors[self.background_prototype_indices]), 0).detach()
        self.ones_deselection_vectors = torch.nn.Parameter(torch.ones(self.deselection_vectors.shape), requires_grad=False)
        # self.zero_grad()

        print("reinit model params")
        print(self.prototype_vectors.shape)
        print(self.ones.shape)
        print(self.prototype_mask.shape)

    def delete_prototypes(self, prototype_indices):
        """ Delete prototypes according to self.background_prototype_indices
        Args:
            prototype_indices: Multi hot vector with as many entries as there are prototypes (1D Numpy array)
            """
        prototype_indices = ~prototype_indices
        self.num_prototypes = torch.sum(prototype_indices).item()
        self.prototype_class_identity.data = self.prototype_class_identity.data[prototype_indices]
        self.prototype_vectors.data = self.prototype_vectors.data[prototype_indices]
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototype_indices]
        self.ones.data = self.ones.data[prototype_indices]
        self.prototype_mask = self.prototype_mask[prototype_indices]
        self.prototype_shape[0] = len(prototype_indices)
        self.prototype_indices = self.prototype_indices[prototype_indices.cpu().detach().numpy()]  # Changes length of prototype indices!
        self.bg_prototypes_deleted = True

    def reinit_prototypes(self, prototype_indices):
        random = torch.rand(self.prototype_vectors.shape).to(device)
        self.prototype_vectors.data[prototype_indices] = random.data[prototype_indices]
        self.prototype_vectors.to(device)

    def overlay_heatmap(self, heatmap, img, heatmap_threshold=.15, mode=1):
        """ Overlay heatmap and image
        Args:
            heatmap: 2D numpy array
            img: PIL image (RGB)
        Returns:
            upsampled: upsampled heatmap (same size as img)
            combined: image and overlayed heatmap"""

        def apply_colormap(inp, colormap=plt.cm.inferno, vmin=None, vmax=None):
            norm = plt.Normalize(vmin, vmax)
            return colormap(norm(inp))

        def normalize(x):
            x = x - np.min(x)
            x = x / np.max(x)
            return x

        if mode == 0:
            heatmap = np.log((heatmap + 1) / (heatmap + 0.0001))
            upsampled = cv2.resize(heatmap, dsize=img.size, interpolation=cv2.INTER_CUBIC)
            combined = 0.5 * (apply_colormap(upsampled, plt.cm.inferno) + apply_colormap(np.array(img)[:, :, 0], plt.cm.gray))
        elif mode == 1:
            # Threshold heatmap and overlay as cyan
            upsampled = img.resize((224, 224))
            heatmap_scaling_uint8 = int(255 / np.max(np.log((self.heatmaps + 1) / (self.heatmaps + 0.0001))))
            heatmap = np.log((heatmap + 1) / (heatmap + 0.0001)) * heatmap_scaling_uint8
            heatmap = np.array(Image.fromarray(heatmap).resize((224, 224)), dtype=np.float) / 255
            heatmap = np.repeat(np.expand_dims(heatmap, -1), 3, -1)
            heatmap = (heatmap > heatmap_threshold) * 255
            heatmap[:, :, 0] = 0
            combined = normalize(upsampled + 0.5 * heatmap)
        else:
            raise Exception("No such mode")
        return upsampled, combined

    def plot_background_prototypes(self, threshold=0, ny=10, nx=17, shuffle=True, mode=1, heatmap_threshold=.15, plot_foreground = False):
        """ Plots background prototypes
        Args:
            threshold: Threshold for the allowed foreground evidence. Either zero or a value smaller than one.
            ny: Number of rows in the plot
            nx: Number of columns in the plot
        """
        fig, ax = plt.subplots(ny, nx, figsize=(20, 10), dpi=200)
        background_foreground = self.background_foreground_prototypes()
        idxs_background_prototypes = np.where(np.array(background_foreground) <= threshold)[0]
        if plot_foreground:
            idxs_background_prototypes = np.where(np.array(background_foreground) > threshold)[0]
        prototype_indices_in_img = self.prototype_indices[idxs_background_prototypes][:, 1:]
        heatmaps = self.heatmaps[idxs_background_prototypes]
        image_paths = self.prototype_image_paths[idxs_background_prototypes]
        groundtruth = self.prototype_groundtruth_paths[idxs_background_prototypes]
        i_shuffle = np.arange(len(idxs_background_prototypes))
        if shuffle:
            np.random.shuffle(i_shuffle)

        idx = 0
        stop = False
        for y in range(ny):
            if stop:
                break
            for x in range(nx):
                if idx >= len(idxs_background_prototypes):
                    break
                i = i_shuffle[idx]
                img = Image.open(image_paths[i]).resize((224, 224))
                rgbimg = Image.new("RGB", img.size)
                rgbimg.paste(img)
                img = rgbimg
                _, combined = self.overlay_heatmap(heatmaps[i], img, heatmap_threshold=heatmap_threshold, mode=mode)
                ax[y, x].imshow(combined)
                yp, xp = prototype_indices_in_img[i]
                rect = patches.Rectangle((xp * 32 -1, yp * 32 -1), 32, 32, linewidth=1, edgecolor='r', facecolor='none')
                ax[y, x].add_patch(rect)
                ax[y, x].axis("off")
                idx += 1
        return fig

    def plot_pdf_background_foreground_prototypes(self):
        """ Plots probability density function of last layer weigths for foreground and background prototypes """

        def interp_gmm(x, samples, n_components=2):
            gmm = GMM(n_components=n_components, max_iter=10000, random_state=10, covariance_type='full')
            clf = gmm.fit(samples.reshape(-1, 1))
            x_interp = np.linspace(np.min(x), np.max(x), 100)
            y_interp = np.exp(clf.score_samples(x_interp.reshape(-1, 1))).flatten()
            return x_interp, y_interp

        # Get background/foreground prototype and respective weights
        background_foreground = self.background_foreground_prototypes()
        idxs_background_prototypes = np.where(np.array(background_foreground) <= self.bg_thres_perc_fg)[0]
        idxs_foreground_prototypes = list(np.arange(2000))
        for e in idxs_background_prototypes:
            idxs_foreground_prototypes.remove(e)
        weights = self.last_layer.weight.detach().cpu().numpy()[np.where(self.prototype_class_identity.cpu().detach().numpy().T)]

        # Compute histograms
        y, x = np.histogram(weights[idxs_background_prototypes], 20, density=True)
        x = x[1:]
        y1, x1 = np.histogram(weights[idxs_foreground_prototypes], 20, density=True)
        x1 = x1[1:]

        # Plot PDF with bin edges as scatter
        fig, ax = plt.subplots(1, dpi=200)
        ax.scatter(x, y)
        x_interp, y_interp = interp_gmm(x, weights[idxs_background_prototypes])
        ax.plot(x_interp, y_interp, label="Non-Object Prototypes")
        ax.scatter(x1, y1)
        x1_interp, y1_interp = interp_gmm(x1, weights[idxs_foreground_prototypes])
        ax.plot(x1_interp, y1_interp, label="Object Prototypes")

        ax.legend(prop={'size': 14})
        ax.set_xlabel("Weight", fontsize=14)
        ax.set_ylabel("Probability Density", fontsize=14)
        return fig

    def plot_histograms_bgfg(self, alt=False):
        background_foreground = self.background_foreground_prototypes()
        fig1, ax = plt.subplots(1, 2, dpi=200, figsize=(12, 4))
        fig = fig1
        ax[0].set_xlabel("foreground evidence")
        ax[0].set_ylabel("frequency")
        ax[0].hist(background_foreground)
        ax[1].set_xlabel("foreground evidence")
        ax[1].set_ylabel("frequency")
        ax[1].hist(background_foreground[background_foreground > 0.0])
        ax[0].set_title("before deselection")
        ax[1].set_title("after deselection")
        fig.tight_layout()

        if alt:
            fig1, ax = plt.subplots(1, dpi=200, figsize=(8, 4))
            ax.set_xlabel("foreground evidence")
            ax.set_ylabel("frequency")
            ax.hist(background_foreground)

            fig, ax = plt.subplots(1, dpi=200, figsize=(8, 4))
            ax.set_xlabel("foreground evidence")
            ax.set_ylabel("frequency")
            ax.hist(background_foreground[background_foreground > 0.0])

        return fig1, fig

    def plot_images_with_prototype(self, label_class=20, heatmap_factor=.5, heatmap_threshold=.2, resample = 1, start = None, stop = None, plot_rectangle = True):
        """ Plot images with cyan region that indicates an area with high activation for the prototype 
            and a red box for the image patch that corresponds to the grid cell with the highest prototype activation.
        Args:
            label_class: Class for which the prototypes are plotted
            heatmap_factor: Visibility of cyan region

            """

        def normalize(x):
            x = x - np.min(x)
            x = x / np.max(x)
            return x

        prototypes_per_class = self.num_prototypes // self.num_classes
        if not start:
            start = label_class * prototypes_per_class
            stop = start + prototypes_per_class
        arrs = []
        heatmap_scaling_uint8 = int(255 / np.max(np.log((self.heatmaps + 1) / (self.heatmaps + 0.0001))))
        print(heatmap_scaling_uint8)
        for img, idxs, heatmap in zip(self.prototype_image_paths[start:stop], self.prototype_indices[start:stop],
                                      self.heatmaps[start:stop]):
            fig, ax = plt.subplots(1, figsize=(2, 2), dpi=200)
            img = Image.open(img)
            img_shape = img.size
            img = img.resize((224, 224))

            # Threshold heatmap
            heatmap = np.log((heatmap + 1) / (heatmap + 0.0001)) * heatmap_scaling_uint8
            heatmap = np.array(Image.fromarray(heatmap).resize((224, 224)), dtype=np.float) / 255
            heatmap = np.repeat(np.expand_dims(heatmap, -1), 3, -1)
            heatmap = (heatmap > heatmap_threshold) * 255
            heatmap[:, :, 0] = 0

            yp, xp = idxs[1:]
            rect = patches.Rectangle((xp * 32, yp * 32), 32, 32, linewidth=3, edgecolor='r', facecolor='none')
            ax.imshow(normalize(img + heatmap_factor * heatmap))
            if plot_rectangle:
                ax.add_patch(rect)
            ax.axis("off")
            fig.tight_layout()
            arr = np.array(Image.fromarray(fig2rgb_array(fig)).resize(img_shape, resample = resample)) / 255

            plt.close()
            arrs.append(arr)
        return arrs
    def plot_weights_of_prototypes(self):
        background_foreground = self.background_foreground_prototypes()
        idxs_background_prototypes = np.where(np.array(background_foreground) <= self.bg_thres_perc_fg)[0]
        weights = self.last_layer.weight.detach().cpu().numpy()[np.where(self.prototype_class_identity.T)]
        fig, ax = plt.subplots(1, dpi=200)
        ax.hist(weights)
        ax.set_xlabel("weight")
        ax.set_ylabel("n prototypes")
        return fig

    def plot_prototypes_of_class(self, label_class=20, ny=5, nx=2):
        arrs = self.plot_images_with_prototype(label_class, .5, .2)
        fig, ax = plt.subplots(ny, nx, figsize=(2, 4), dpi=200)
        i = 0
        for y in range(ny):
            for x in range(nx):
                ax[y, x].imshow(arrs[i])
                ax[y, x].axis("off")
                i += 1
        fig.tight_layout(pad=0.1)
        return fig

    def push(self, dataloader, no_side_effects=False):
        """ Retrieves the closest patch encodings to the prototypes. The distance between image encodings (latent grid vectors) and prototypes of the image class is computed.
            The preliminary vectors for the closest patch encoding are iteratively updated if a closer patch is found.
        Args:
            dataloader: Push dataloader
        Returns:
            prototype_update: Matrix of the closest patch encodings for each prototype. Can be used to update the current prototypes.
        """
        n_prototypes = self.num_prototypes
        num_classes = self.num_classes
        previous_prototype_indices = self.prototype_indices.copy()
        global_min_proto_dist = np.full(n_prototypes, np.inf)  # saves the patch representation that gives the current smallest distance

        prototype_shape = self.prototype_shape
        global_min_fmap_patches = np.zeros(prototype_shape)
        for push_iter, batch in enumerate(dataloader):
            search_batch_input, search_y = batch[0], batch[1]
            self.eval()
            if type(self.push_preprocessing) != type(None):
                search_batch = self.push_preprocessing(search_batch_input)
            else:
                search_batch = search_batch_input

            search_batch = search_batch.cuda()
            features, distances = self.push_forward(search_batch)

            features = np.copy(features.detach().cpu().numpy())
            distances = np.copy(distances.detach().cpu().numpy())

            class_to_img_index = {key: [] for key in range(num_classes)}
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index[img_label].append(img_index)

            prototype_shape = self.prototype_shape
            max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

            for j in range(n_prototypes):
                target_class = torch.argmax(self.prototype_class_identity[j]).item()  # each prototype has exactely one class
                if len(class_to_img_index[target_class]) == 0:  # push only to image encodings of correct class; if there are none in batch continue.
                    continue
                distances_correct_class = distances[class_to_img_index[target_class]][:, j, :, :]  # shape [samples_in_batch_of_class_j, 7, 7]

                batch_min_distances_correct_class = np.amin(distances_correct_class)  # minimum value of all images in batch that are of the prototypes target class
                if batch_min_distances_correct_class < global_min_proto_dist[j]:
                    subset_argmin_distances_correct_class = list(np.unravel_index(np.argmin(distances_correct_class, axis=None), distances_correct_class.shape))  # 3d coordinates of min val
                    # replace coordinate of image in subset (of correct class) by coordinate of original image idx in batch ...
                    batch_argmin_distances_correct_class = copy.deepcopy(subset_argmin_distances_correct_class)
                    batch_argmin_distances_correct_class[0] = class_to_img_index[target_class][batch_argmin_distances_correct_class[0]]

                    # retrieve the corresponding feature map patch
                    img_idx_batch = batch_argmin_distances_correct_class[0]
                    fmap_height_start = batch_argmin_distances_correct_class[1]
                    fmap_width_start = batch_argmin_distances_correct_class[2]

                    # were not interested in the value at minimum distance but the patch encoding (output of features/prototype Layer input)
                    batch_min_fmap_patch_j = features[img_idx_batch, :, fmap_height_start:fmap_height_start + 1, fmap_width_start:fmap_width_start + 1]
                    global_min_proto_dist[j] = batch_min_distances_correct_class
                    global_min_fmap_patches[j] = batch_min_fmap_patch_j

                    # save coordinates of image in dataset and patch position
                    self.prototype_indices[j, 0] = int(push_iter * dataloader.batch_size + img_idx_batch)
                    self.prototype_indices[j, 1] = int(fmap_height_start)
                    self.prototype_indices[j, 2] = int(fmap_width_start)

                    # save heatmaps
                    self.heatmaps[j] = distances_correct_class[subset_argmin_distances_correct_class[0]]
                    self.prototype_image_paths[j] = batch[2][img_idx_batch]
                    self.prototype_groundtruth_paths[j] = batch[3][img_idx_batch]

        if no_side_effects:  # Do not update prototype indices, prototype vectors, no call of set_unwanted_prototypes
            new_prototype_indices = self.prototype_indices.copy()
            self.prototype_indices = previous_prototype_indices
            prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
            del class_to_img_index
            return prototype_update, new_prototype_indices
        else:
            self.last_push_prototype_vectors = self.prototype_vectors.detach().clone()
            prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
            del class_to_img_index
            self.set_unwanted_prototypes()
            return prototype_update

    def push_epoch(self, iteration, last_layer_trainer, push_trainer, train_loader, test_loader):
        print("\n\n\n\n-------------------  PUSH EPOCH (iteration " + str(iteration) + ") -------------------------------")
        ppnet = self.to(device)

        ppnet.record = False  # Turn off logging
        print("-------------------> Testing before push")
        ppnet.features = ppnet.features.to(device)
        res = push_trainer.test(ppnet, test_loader)[0]
        ppnet.custom_logger.add_scalar("push/accuracy_before_push", res["accuracy"], ppnet.current_global_epoch)

        ppnet.push_prototypes()
        print("-------------------> Testing after push")
        res = push_trainer.test(ppnet, test_loader)[0]
        ppnet.custom_logger.add_scalar("push/accuracy_after_push", res["accuracy"], ppnet.current_global_epoch)

        print("-------------------> Training of last layer")
        ppnet.set_training_phase("last_layer")
        last_layer_trainer.fit(ppnet, train_loader, test_loader)
        res = last_layer_trainer.test(ppnet, test_loader)[0]
        ppnet.custom_logger.add_scalar("push/accuracy_after_fixing_of_last_layer", res["accuracy"], ppnet.current_global_epoch)

        print("-------------------> End-To-End Training")
        ppnet.record = True  # Turn on logging
        # train for several epochs via push trainer...
        ppnet.set_training_phase("push")
        _ = push_trainer.fit(ppnet, train_loader, test_loader)


def main(args):
    base_architecture = 'resnet34'
    no_bounding_boxes = True
    debug = args.debug
    refresh_rate = 1 if args.progressbar else 0
    img_size = 224
    prototype_shape = [2000, 128, 1, 1]
    num_classes = 200
    n_push = 1
    hash = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    print(hash)

    if args.push_phase:
        logger = SummaryWriter("./tensorboard/" + "including_push/" + hash)
    else:
        logger = SummaryWriter("./tensorboard/" + "first_push_fix/" + hash)

    features = DeepEncoder(base_architecture, prototype_shape)

    dset = CUB200(data_path=args.dataset, debug=debug, with_ground_truth=True)

    ppnet = PPNet(features=features, img_size=img_size, prototype_shape=prototype_shape,
                  num_classes=num_classes, training_phase="warm",
                  push_dataloader=dset.train_push_dataloader(), logger=logger)

    warm_trainer = Trainer(gpus=1, max_epochs=5, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    joint_trainer = Trainer(gpus=1, max_epochs=5, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    push_trainer = Trainer(gpus=1, max_epochs=10, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)
    last_layer_trainer = Trainer(gpus=1, max_epochs=20, callbacks=[], progress_bar_refresh_rate=refresh_rate, checkpoint_callback=False, logger=False)

    print("\n\n\n\n---------------- WARMING -----------------------------")
    ppnet.set_training_phase("warm")
    warm_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    _ = warm_trainer.test(ppnet, dset.test_dataloader())

    print("\n\n\n\n---------------- JOINT TRAINING -----------------------------")
    ppnet.set_training_phase("joint")
    joint_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())
    _ = joint_trainer.test(ppnet, dset.test_dataloader())

    if args.push_phase:
        for i in range(n_push):
            ppnet.push_epoch(i, last_layer_trainer, push_trainer, dset.train_dataloader(), dset.test_dataloader())
    else:
        print("\n\n\n\n---------------- PUSH -----------------------------")
        prototype_update, _ = ppnet.push(dset.train_push_dataloader(), no_side_effects=True)
        ppnet.prototype_vectors.data = torch.tensor(prototype_update, dtype=torch.float32).cuda()
        res = push_trainer.test(ppnet, dset.test_dataloader())[0]
        ppnet.custom_logger.add_scalar("push/accuracy_after_push", res["accuracy"], ppnet.current_global_epoch)
        ppnet.set_training_phase("last_layer")
        print("\n\n\n\n---------------- LAST LAYER -----------------------------")
        last_layer_trainer.fit(ppnet, dset.train_dataloader(), dset.test_dataloader())

    root_out = "weights/after_push_fix"
    os.makedirs(root_out, exist_ok=True)
    filename = hash + ".pth"
    outpath = os.path.join(root_out, filename)
    torch.save(ppnet.state_dict(), outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--progressbar', dest='progressbar', action='store_true', default=False)
    parser.add_argument('--push_phase', dest='push_phase', action='store_true', default=False)
    parser.add_argument('--repetitions', dest='repetitions', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--dataset', action='store', type=str, help='path to dataset', default="~/datasets/cub200/images/")

    args = parser.parse_args()
    for _ in range(args.repetitions):
        main(args)




