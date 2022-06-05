import json
import math

import numpy as np
import torch
from ddsmoothing.certificate import L2Certificate
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from datasets.utils import load_toy_dataset
from models.utils import load_toy_model


def plot_decision_boundary(pred_func, X, i, ax1):
    # Set min and max values and give it some padding
    padding = 2
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    ax1.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax1.scatter(X[i, 0], X[i, 1], c='white')


def rotation_matrix_to_angle(A):
    test_vec = np.array([1, 0])
    rotated_vec = A * test_vec.T

    inner = np.inner(test_vec, rotated_vec)
    norms = np.linalg.norm(test_vec) * np.linalg.norm(rotated_vec)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)

    return deg


def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
    pred = model(x)
    pred = torch.argmax(pred, dim=1)

    return pred.cpu().detach().numpy()


if __name__ == '__main__':
    with open('config.json') as json_file:
        params = json.load(json_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    isotropic_file = 'saved_parameters/isotropic/isotropic_parameters'
    same_predictions_file = 'saved_parameters/isotropic/same_predictions'
    ancer_optimization_folder = "saved_parameters/ancer"
    rancer_optimization_folder = "saved_parameters/rancer/thetas"
    rancer_matrices_folder = "saved_parameters/rancer/matrices"

    model = load_toy_model(params, device)

    certificate = L2Certificate(params['batch_size'], device=device)

    isotropic_thetas = torch.load(isotropic_file, map_location=device)
    same_predictions = torch.load(same_predictions_file, map_location=device)

    dataset = load_toy_dataset(params)

    dataset_length = len(dataset)
    print("dataset_length: ", dataset_length)

    for i in range(dataset_length):
        width_iso, height_iso = isotropic_thetas[i], isotropic_thetas[i]
        print(f"Isotropic: width = {width_iso}, height = {height_iso}")

        print("same_predictions: ", same_predictions[i])

        ancer_theta_i = torch.relu(
            torch.load(ancer_optimization_folder + '/theta_test_' + str(i) + '.pt')
        )

        width_ancer = ancer_theta_i[0][0][0]
        height_ancer = ancer_theta_i[0][0][1]
        print(f"Ancer: width = {width_ancer}, height = {height_ancer}")

        hessian_optimization_theta_i = torch.relu(
            torch.load(rancer_optimization_folder + '/theta_test_' + str(i) + '.pt')
        )

        width_hessian_optimization = hessian_optimization_theta_i[0][0][0]
        height_hessian_optimization = hessian_optimization_theta_i[0][0][1]
        print(f"Rancer: width = {width_hessian_optimization}, height = {height_hessian_optimization}")

        rotation_matrix = torch.load(rancer_matrices_folder + '/rotation_matrix_test_' + str(i) + '.pt')
        rotation_matrix = rotation_matrix.cpu().numpy()

        rotation_angle_hessian = rotation_matrix_to_angle(rotation_matrix)[0][0]
        print("rotation_angle: ", rotation_angle_hessian)

        x, y = dataset[i][0][0][0][0], dataset[i][0][0][0][1]

        ell_iso = Ellipse(xy=(x, y), width=width_iso * 2, height=height_iso * 2, angle=0, edgecolor='yellow',
                          linewidth=2,
                          fill=False)

        ell_ancer = Ellipse(xy=(x, y), width=width_ancer * 2, height=height_ancer * 2,
                            angle=0,
                            edgecolor='blue', linewidth=2,
                            fill=False)


        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            ox, oy = origin
            px, py = point

            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy


        second_point_ancer = (x + width_ancer.cpu(), y)
        third_point_ancer = (x, y + height_ancer.cpu())

        second_point = (x + width_hessian_optimization.cpu(), y)
        third_point = (x, y + height_hessian_optimization.cpu())
        second_point_rot = np.dot(rotation_matrix, np.array(second_point) - np.array([x, y])) + np.array([x, y])
        third_point_rot = np.dot(rotation_matrix, np.array(third_point) - np.array([x, y])) + np.array([x, y])

        ell_rancer = Ellipse(xy=(x, y), width=width_hessian_optimization * 2, height=height_hessian_optimization * 2,
                             angle=-rotation_angle_hessian,
                             edgecolor='pink', linewidth=2,
                             fill=False)

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(ell_iso)
        ax.add_patch(ell_ancer)
        ax.add_patch(ell_rancer)

        ax.plot((x, second_point_ancer[0]), (y, second_point_ancer[1]), 'bo-')
        ax.plot((x, third_point_ancer[0]), (y, third_point_ancer[1]), 'bo-')

        ax.plot((x, second_point_rot[0][0]), (y, second_point_rot[0][1]), 'o-', color='pink')
        ax.plot((x, third_point_rot[0][0]), (y, third_point_rot[0][1]), 'o-', color='pink')

        ell_iso.set_clip_box(ax.bbox)
        ell_iso.set_alpha(1)
        ell_iso.set(label="Isotropic")
        ell_ancer.set_clip_box(ax.bbox)
        ell_ancer.set_alpha(1)
        ell_ancer.set(label="ANCER")
        ell_rancer.set_clip_box(ax.bbox)
        ell_rancer.set_alpha(1)
        ell_rancer.set(label="RANCER")

        ax.legend(loc="upper right")

        plot_decision_boundary(lambda x: predict(x), dataset[i][0][0].cpu().detach().numpy(), 0,
                               ax)

        plt.show()
