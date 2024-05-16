import argparse
from fit_prototypes_and_predict import *
from deselection_masking import *
from deselection_loss_experiments import *
from iterative_deselection_training import *
from refinement_by_pixel_distance import *

if __name__=="__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument('--progressbar', dest='progressbar', action='store_true', default=False)
    parser.add_argument('--dataset', action = 'store', type = str, help = 'path to dataset', default="~/datasets/cub200/images/")
    parser.add_argument('--dataset_clever_hans', action = 'store', type = str, help = 'path to dataset', default="")
    parser.add_argument('--train_on_clever_hans', action="store_true", default = False)
    parser.add_argument('--experiment_suffix', action='store', type=str, help = "Will be appended to has of run and name of weights", default="")
    parser.add_argument('--experiment', action = 'store', type = str, help = 'Choose from fit_prototypes_and_pedict, deselection_masking ...', default="fit_prototypes_and_pedict")

    # Args of fitting prototypes
    parser.add_argument('--last_layer_epochs', dest='last_layer_epochs', type=int, default=100)
    parser.add_argument('--joint_epochs', dest='joint_epochs', type=int, default=20)
    parser.add_argument('--last_layer_optimizer_lr', dest='last_layer_optimizer_lr', type=float, default=1e-4)

    # Args of deselection masking:
    parser.add_argument('--plots_only', dest='plots_only', action='store_true', default=False)
    parser.add_argument('--override', dest='override', action='store_true', default=False)

    # Args for deselection loss
    parser.add_argument('--delete_bg_prototypes', dest = 'delete_bg_prototypes', action='store_true', default=False)
    parser.add_argument('--reinit_prototypes', dest = 'reinit_prototypes', action='store_true', default=False)
    parser.add_argument('--weights_idx', dest = 'weights_idx', type = int, default = 0, help='The index of the hash in alphabetic order for the weights to begin with (after_push_fix)')
    parser.add_argument('--bg_thres_perc_fg', dest = 'bg_thres_perc_fg', type=float, default = 0.0, help='Upper limit of foreground pixels allowed in grid cells to be interpreted as foreground')
    parser.add_argument('--n_deselections', dest='n_deselections', type=int, default = 5)
    parser.add_argument('--lr_deselection_last_layer', dest = "lr_deselection_last_layer", type=float, default = 1e-4)
    parser.add_argument('--lr_deselection_prototype_vectors', dest = "lr_deselection_prototype_vectors", type=float, default = 1e-3)

    parser.add_argument('--w_cluster', dest = 'w_cluster', type = float, default = .8)
    parser.add_argument('--w_sep', dest ='w_sep', type = float, default = -0.08)
    parser.add_argument('--w_l1', dest = 'w_l1', type= float, default=1e-4)
    parser.add_argument('--w_crossentropy', dest = 'w_crossentropy', type=float, default = 1.0)
    parser.add_argument('--w_deselection', dest = 'w_deselection', type=float, default = 50.0)

    args = parser.parse_args()
    if args.experiment == "fit_prototypes_and_predict":
        fit_prototypes_and_predict(args)
    elif args.experiment == "deselection_masking":
        main_deselection(args, deselection_by="masking")
    elif args.experiment == "deselection_deleting":
        main_deselection(args, deselection_by="deleting")
    elif args.experiment == "deselection_loss":
        main_deselection_loss(args)
    elif args.experiment == "iterative_deselection":
        main_deselection_training(args)
    elif args.experiment == "last_layer_after_deselection":
        main_last_layer_after_deselection(args)
    elif args.experiment == "last_layer_iterative_deselection":
        main_last_layer_iterative_deselection(args)
    elif args.experiment == "refinement_by_pixel_distance":
        main_refinement_by_pixel_distance(args)
    else:
        print("No such experiment. Exiting ...")


