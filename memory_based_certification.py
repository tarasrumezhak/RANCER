import csv

import numpy as np
import torch
from scipy.optimize import root_scalar, Bounds, minimize
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm

check_values = np.linspace(0.1, 0.99, 10)


def check_ellipsoid_intersection(A, a, B, b):
    """ Checks whether the ellipsoids defined as:

        S_A = {x: (x - a)^T A (x - a) <= 1} and
        S_B = {x: (x - b)^T B (x - b) <= 1}

        intersect at any point.
    """
    A_np = A.flatten().detach().numpy()
    a_np = a.flatten().detach().numpy()
    B_np = B.flatten().detach().numpy()
    b_np = b.flatten().detach().numpy()

    K_t = lambda t: 1 - np.sum(((b_np - a_np) ** 2) * ((t * (1 - t) * A_np * B_np) / (t * A_np + (1 - t) * B_np)))

    if K_t(0.5) < 0:
        return False

    func_value = np.min([K_t(v) for v in check_values])

    if func_value < 0:
        return False

    bounds = Bounds([0.001], [0.999])
    x_0 = 0.5
    res = minimize(K_t, x_0, method="trust-constr", options={'verbose': 1}, bounds=bounds)

    if res.fun[0] < 0:
        return False
    else:
        return True


# For RANCER =====================================================
def check_rotated_ellipsoid_intersection(A, a, B, b):
    """ Checks whether the ellipsoids defined as:

        S_A = {x: (x - a)^T A (x - a) <= 1} and
        S_B = {x: (x - b)^T B (x - b) <= 1}

        intersect at any point.
    """
    A_np = A.detach().numpy()
    a_np = a.detach().numpy()
    B_np = B.detach().numpy()
    b_np = b.detach().numpy()

    K_t = lambda t: np.matmul((b_np - a_np), np.matmul(
        np.linalg.inv((1 / (1 - t)) * np.linalg.inv(B_np) + (1 / t) * np.linalg.inv(A_np)), (b_np - a_np).T))

    if K_t(0.5) < 0:
        return False

    func_value = np.min([K_t(v) for v in check_values])

    if func_value < 0:
        return False

    bounds = Bounds([0.001], [0.999])
    x_0 = 0.5
    res = minimize(K_t, x_0, method="trust-constr", options={'verbose': 1}, bounds=bounds)

    if res.fun < 0:  # [0]
        return False
    else:
        return True


def project_on_diagonal_ellipsoid(A, b):
    """ Projects the vector y into the set S = {x: x^T A x <= c}
    inputs:
      A: numpy array corresponding to the diagonal of the covariance matrix
      b: numpy vector to be projected

    outputs:
      projection: numpy array of the projected version of y
      a bool: True if the projected vector of y is in the desired set.
    """

    def check(A, b):
        return np.dot(A, b ** 2) <= 1 + 1e-3

    def solve(A, b):
        f = lambda t: ((b ** 2 * A) / (1 + t * A) ** 2).sum() - (1 + 1e-3)

        return root_scalar(f, method='bisect', bracket=[0.01, 1000], xtol=1e-3).root

    # Check if y belongs to our region
    if check(A, b):
        # print('your vector is already in the desired set')
        return b, True

    # print('projecting your vector ...')
    t = solve(A, b)
    projection = b / (1 + t * A)
    # print('Done!')
    return projection, check(A, projection)


# For RANCER =====================================================
def project_on_rotated_ellipsoid(A, b):
    """ Projects the vector y into the set S = {x: x^T A x <= c}
    inputs:
      A: numpy array corresponding to the diagonal of the covariance matrix
      b: numpy vector to be projected

    outputs:
      projection: numpy array of the projected version of y
      a bool: True if the projected vector of y is in the desired set.
    """

    def check(A, b):
        return np.matmul(b, np.matmul(A, b.T)) <= 1 + 1e-3

    def solve(A, b):
        f = lambda t: np.matmul(b, np.matmul(
            np.matmul(np.linalg.inv(t * A + np.eye(2)), np.matmul(A, np.linalg.inv(t * A + np.eye(2)).T)), b.T)) - (
                              1 + 1e-3)

        return root_scalar(f, method='bisect', bracket=[0.0001, 1000],
                           xtol=1e-3).root  # bisect,  change bracket range according to different size

    if check(A, b):
        return b, True

    t = solve(A, b)
    projection = b / (1 + t * A)
    return projection, check(A, projection)


def get_ancer_sigma(sigma_folder_path: str, i: int):
    theta_i = torch.relu(
        torch.load(
            sigma_folder_path + '/sigma_test_' + str(i) + '.pth',
            map_location=torch.device('cpu'))
    )

    return theta_i


def main(args):
    g = args.results_file_path

    print("loading data...")

    f = open(str(g), "r")
    index, label, prediction, min_radius, max_radius, correct, proxy_radius = [], [], [], [], [], [], []
    with open(str(g)) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row_i, row in tqdm(enumerate(reader)):
            min_radius.append(float(row["radius"]))

            correct.append(int(row["correct"]))
            prediction.append(int(row["predict"]))
            label.append(int(row["label"]))
            index.append(int(row["idx"]))
            proxy_radius.append(float(row["proxyvol"]))

            # obtain the maximum l_p radius by using the gap computed from the min one
            optim_sigmas = get_ancer_sigma(args.optimized_sigmas, row_i)
            max_radius.append(min_radius[-1] * (optim_sigmas.max() / optim_sigmas.min()))

    print("loaded all data.")

    dataset = CIFAR10(root='./train/datasets', train=False, download=True, transform=ToTensor())

    saved_images, saved_predictions, saved_min_radii, saved_max_radii, saved_proxy_rad, keep_original_sigmas = [], [], [], [], [], []
    # anything_detected = False
    for i in tqdm(range(len(min_radius))):

        idx, pred, min_rad, max_rad, proxy_rad = index[i], prediction[i], min_radius[i], max_radius[i], proxy_radius[i]
        img, _ = dataset[idx]

        if args.verbose:
            print(f"----------- New point {idx} ----------")

        # a variable indicating whether we should get the original sigmas or consider it a ball
        keep_sigma = True

        if len(saved_images) != 0:
            # Get the differences
            diff = torch.norm(img.reshape(1, -1) - torch.stack(saved_images).reshape(len(saved_max_radii), -1), dim=1)

            where_max_overlap = diff < (torch.tensor(saved_max_radii) + max_rad)

            # Check whether this image is with overlap with any other instances
            if where_max_overlap.any():
                if args.verbose:
                    print("- Maximums overlap")

                where_max_overlap_diff_class_idx = \
                    torch.where((torch.tensor(saved_predictions) != pred) & where_max_overlap)[0]

                if len(where_max_overlap_diff_class_idx) > 0:
                    if args.verbose:
                        print(
                            "-- Maximums between different predictions overlap! Test based on ellipsoid intersection...")

                    # load the sigmas of the new point we're inferrencing on
                    # and build the B matrix as per the paper
                    B_sigmas = get_ancer_sigma(args.optimized_sigmas, i)
                    B_sigmas = (1 / B_sigmas) / (min_rad / B_sigmas.min())
                    b = img
                    keep_sigma = False

                    # for each of them, check the ellipsoids intersection
                    for overall_idx in where_max_overlap_diff_class_idx:
                        # ignore points where we abstained at that ellipsoid previously
                        if saved_predictions[overall_idx] == -1:
                            continue

                        A_min_radius = saved_min_radii[overall_idx]

                        # if it's the original sigmas, load them; otherwise
                        # just take A_sigmas to be a ball of radius saved_min_radii
                        if keep_original_sigmas[overall_idx]:
                            A_sigmas = get_ancer_sigma(args.optimized_sigmas, overall_idx.item())
                            A_gap = A_min_radius / A_sigmas.min()
                            A_sigmas = (1 / A_sigmas) / A_gap
                        else:
                            A_gap = 1
                            A_sigmas = torch.ones_like(B_sigmas) / A_min_radius

                        a, _ = dataset[overall_idx]

                        # check if ellipsoids intersect
                        if not check_ellipsoid_intersection(A_sigmas, a, B_sigmas, b):
                            continue

                        if args.verbose:
                            print("--- Intersection found. Fix with ellipsoid method")

                        # they do intersect, correct based on the ellipsoid projection method
                        if torch.linalg.norm(A_sigmas.flatten() * (b - a).flatten()) <= 1:
                            if args.verbose:
                                print("--- Failed for point inside an ellipsoid")

                            # box adjustment did not work, will abstain
                            pred = -1
                            min_rad = 0
                            max_rad = 0
                            proxy_rad = 0
                            break

                        # ELLIPSOID PROJECTION METHOD
                        delta_a, success = project_on_diagonal_ellipsoid(
                            A_sigmas.flatten().detach().numpy(),
                            (b - a).flatten().detach().numpy()
                        )

                        new_b = torch.Tensor(delta_a).reshape(a.shape) + a
                        min_rad = min(min_rad, torch.linalg.norm(new_b - b).item())
                        max_rad = min_rad
                        proxy_rad = min_rad

        if isinstance(max_rad, torch.Tensor):
            max_rad = max_rad.item()

        saved_images.append(img)
        saved_predictions.append(pred)
        saved_min_radii.append(min_rad)
        saved_max_radii.append(max_rad)
        saved_proxy_rad.append(proxy_rad)
        keep_original_sigmas.append(keep_sigma)

        if args.verbose:
            print("Done with point")

    print("You are Done!, --------> Saving results")

    f = open(args.new_results_file_path, 'w')
    print("idx\tlabel\tpredict\tradius\tvolume\tproxyvol\tcorrect\tsigma\tproduct\ttime", file=f, flush=True)
    for i in range(len(index)):
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{:.3}\t{}\t{}\t{}\t{}".format(
                index[i],
                label[i],
                saved_predictions[i],
                saved_min_radii[i],
                1,
                saved_proxy_rad[i],
                int(label[i] == saved_predictions[i]),
                1,
                1,
                1
            ),
            file=f,
            flush=True
        )

    print("You are officially done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--results-file-path", type=str, required=True,
        help="path to the certification results file"
    )
    parser.add_argument(
        "--optimized-sigmas", type=str, required=True,
        help="path to the ANCER optimized sigmas folder"
    )
    parser.add_argument(
        "--new-results-file-path", type=str, required=True,
        help="path to the post-processed results file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true"
    )

    args = parser.parse_args()
    main(args)
