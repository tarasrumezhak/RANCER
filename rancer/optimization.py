import numpy as np
import torch
from ddsmoothing.certificate import Certificate
from torch.autograd import Variable


def optimize_rancer(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, device: str = "cuda:0"
):
    """Optimize batch using ANCER, assuming isotropic initialization point.

    Args:
        model (torch.nn.Module): trained network
        batch (torch.Tensor): inputs to certify around
        certificate (Certificate): instance of desired certification object
        learning_rate (float): optimization learning rate for ANCER
        isotropic_theta (torch.Tensor): initialization isotropic value per
            input in batch
        iterations (int): number of iterations to run the optimization
        samples (int): number of samples per input and iteration
        kappa (float): relaxation hyperparameter
        device (str, optional): device on which to perform the computations

    Returns:
        torch.Tensor: optimized anisotropic thetas
    """
    batch_size = batch.shape[0]
    img_size = np.prod(batch.shape[1:])

    isotropic_original = isotropic_theta.clone().cpu().detach().numpy()[:, 0, 0, 0]

    min_bound = isotropic_original * 0.5
    max_bound = isotropic_original * 1.5

    min_bound = torch.Tensor(min_bound).to(device)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)

    max_bound = torch.Tensor(max_bound).to(device)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)

    # define a variable, the optimizer, and the initial sigma values
    theta = Variable(isotropic_theta, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    initial_theta = theta.detach().clone()

    base_classifier_output = model(batch).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()

    def my_loss2(point):
        # return - torch.log(model(point)[0][base_classifier_output_local])
        return criterion(model(point)[0], base_classifier_output_local)

    hessian_batch = []

    for i, sample in enumerate(batch):
        base_classifier_output_local = base_classifier_output[i]  # [0][0]
        hessian = torch.autograd.functional.hessian(my_loss2, sample)

        hessian_batch.append(torch.unsqueeze(hessian, 0))

    hessian_batch = torch.cat(hessian_batch, dim=0)
    hessian_batch = hessian_batch.reshape((batch_size, 3072, 3072))

    hessian_batch = (hessian_batch + torch.transpose(hessian_batch, 1, 2)) / 2

    hessian_mask = torch.abs(hessian_batch) < 0.01
    hessian_batch[hessian_mask] = 0

    eig_val, eig_vec = torch.linalg.eigh(hessian_batch)
    R = eig_vec

    # reshape vectors to have ``samples`` per input in batch
    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    # solve iteratively by projected gradient ascend
    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)

        # Reparameterization trick
        noise_orig = certificate.sample_noise(new_batch, theta_repeated)

        noise = noise_orig.reshape((batch_size, samples, 3072))
        noise_t = torch.transpose(noise, 2, 1)
        rotated_noise_t = torch.matmul(R, noise_t)
        rotated_noise = torch.transpose(rotated_noise_t, 2, 1)
        rotated_noise = rotated_noise.reshape((batch_size, samples, 3, 32, 32))
        rotated_noise = rotated_noise.reshape((batch_size * samples, 3, 32, 32))

        out = model(
            new_batch + rotated_noise
        ).reshape(batch_size, samples, -1).mean(dim=1)

        vals, _ = torch.topk(out, 2)
        gap = certificate.compute_proxy_gap(vals)

        prod = torch.prod(
            (theta.reshape(batch_size, -1)) ** (1 / img_size), dim=1)
        proxy_radius = prod * gap

        radius_maximizer = - (
                proxy_radius.sum() +
                kappa *
                (torch.min(theta.view(batch_size, -1), dim=1).values * gap).sum()
        )
        radius_maximizer.backward()
        optimizer.step()
        optimizer.zero_grad()

        # project to the initial theta
        with torch.no_grad():
            torch.max(theta, initial_theta, out=theta)

        theta = torch.clamp(theta, min=min_bound, max=max_bound)

    return theta, R
