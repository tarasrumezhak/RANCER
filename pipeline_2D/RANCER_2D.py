import json

import torch
from ddsmoothing.certificate import L2Certificate
from torch.utils.data import DataLoader

from datasets.utils import load_toy_dataset
from models.utils import load_toy_model
from optimization.optimizers import OptimizeIsotropicSmoothingParameters, AncerOptimizer, RancerOptimizer

if __name__ == "__main__":

    with open('config.json') as json_file:
        params = json.load(json_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_toy_dataset(params)

    data_loader = DataLoader(dataset, batch_size=params["batch_size"])

    dataset_length = len(dataset)
    print("dataset_length: ", dataset_length)

    certificate = L2Certificate(batch_size=params["batch_size"], device=device)

    model = load_toy_model(params, device)

    # perform the isotropic optimization
    isotropic_file = 'saved_parameters/isotropic/isotropic_parameters'
    same_predictions_file = 'saved_parameters/isotropic/same_predictions'
    isotropic_obj = OptimizeIsotropicSmoothingParameters(
        model, data_loader, device=device
    )

    initial_thetas = params['initial_theta'] * torch.ones(dataset_length).to(device)

    initial_same_prediction = torch.empty(dataset_length, dtype=torch.bool, device='cuda')

    print("initial_thetas: ", initial_thetas.shape, initial_thetas)

    print("Started running isotropic optimization...")
    isotropic_obj.run_optimization(
        certificate, params['iso_learning_rate'], initial_thetas, initial_same_prediction,
        params['iso_iterations'], params['iso_num_samples'], isotropic_file, same_predictions_file
    )

    # open the isotropic file
    isotropic_thetas = torch.load(isotropic_file, map_location=device)
    same_predictions = torch.load(same_predictions_file, map_location=device)

    print("isotropic_thetas: ", isotropic_thetas)
    print("same_predictions: ", same_predictions)

    for i in range(isotropic_thetas.shape[0]):
        if not same_predictions[i]:
            isotropic_thetas[i] = torch.clamp(isotropic_thetas[i], min=0, max=1)

    print("isotropic_thetas: ", isotropic_thetas)

    ancer_optimization_folder = "saved_parameters/ancer"

    ancer_obj = AncerOptimizer(
        model, data_loader, device=device
    )

    print("Started running ANCER optimization")
    ancer_obj.run_optimization_ancer(
        isotropic_thetas, ancer_optimization_folder,
        params['rancer_iterations'], certificate, params['rancer_learning_rate'],
        params['rancer_num_samples'], params['rancer_regularization_weight']
    )
    print("Finished running ANCER optimization")

    rancer_optimization_folder = "saved_parameters/rancer/thetas"
    rancer_matrices_folder = "saved_parameters/rancer/matrices"

    rancer_obj = RancerOptimizer(
        model, data_loader, device=device
    )

    print("Started running RANCER optimization")
    rancer_obj.run_optimization_hessian(
        isotropic_thetas, rancer_optimization_folder,
        params['rancer_iterations'], certificate, rancer_matrices_folder, params['rancer_learning_rate'],
        params['rancer_num_samples'], params['rancer_regularization_weight']
    )
    print("Optimization is fully done!")
