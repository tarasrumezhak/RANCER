import numpy as np
import torch
from ddsmoothing import Certificate
from torch.autograd import Variable


def optimize_isotropic_dds(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        sig_0: torch.Tensor, iterations: int, samples: int,
        device: str = 'cuda:0'
) -> tuple:
    batch_size = batch.shape[0]

    sig = Variable(sig_0, requires_grad=True).view(batch_size, 1, 1, 1)

    for param in model.parameters():
        param.requires_grad_(False)

    base_classifier_output_log_prob = model(batch)
    base_classifier_output_prob = torch.exp(base_classifier_output_log_prob)
    base_classifier_output = torch.argmax(base_classifier_output_prob, dim=-1)

    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    for _ in range(iterations):
        sigma_repeated = sig.repeat((1, samples, 1, 1)).view(-1, 1, 1, 1)

        eps = certificate.sample_noise(
            new_batch, sigma_repeated)

        smoothed_classifier_output_log_prob = model(new_batch + eps)
        smoothed_classifier_output_prob = torch.exp(smoothed_classifier_output_log_prob)

        out = smoothed_classifier_output_prob.reshape(batch_size, samples, -1).mean(1)

        vals, _ = torch.topk(out, 2)

        radius = sig.reshape(-1) / 2 * certificate.compute_proxy_gap(vals)

        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += learning_rate * grad[0]  # Gradient Ascent step

    smooth_classifier_output = torch.argmax(out, dim=-1)

    same_prediction = smooth_classifier_output == base_classifier_output.reshape(-1)

    # For training purposes after getting the sigma
    for param in model.parameters():
        param.requires_grad_(True)

    return sig.reshape(-1), same_prediction


def ancer_optimization(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, device: str = "cuda:0"
) -> tuple:
    batch_size = batch.shape[0]
    img_size = np.prod(batch.shape[1:])

    isotropic_original = isotropic_theta.clone().cpu().detach().numpy()[:, 0, 0, 0]  # [0][0][0][0]

    # define a variable, the optimizer, and the initial sigma values
    theta = Variable(isotropic_theta, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    initial_theta = theta.detach().clone()

    # base_classifier_output_log_prob = model(batch)
    # # print("batch: ", batch)
    # base_classifier_output_prob = torch.exp(base_classifier_output_log_prob)

    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)

        noise = certificate.sample_noise(new_batch, theta_repeated)

        out = torch.exp(model(
            new_batch + noise
        )).reshape(batch_size, samples, -1).mean(dim=1)

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

    min_bound = isotropic_original * 0.85
    max_bound = isotropic_original * 1.15

    min_bound = torch.Tensor(min_bound).to(device)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)

    max_bound = torch.Tensor(max_bound).to(device)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)

    # print("min_bound: ", min_bound.shape, min_bound)

    # TODO: removed clipping
    # theta = torch.clamp(theta, min=min_bound, max=max_bound)

    return theta


def rancer_optimization(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, device: str = "cuda:0"
) -> tuple:
    batch_size = batch.shape[0]
    img_size = np.prod(batch.shape[1:])

    isotropic_original = isotropic_theta.clone().cpu().detach().numpy()[:, 0, 0, 0]

    # define a variable, the optimizer, and the initial sigma values
    theta = Variable(isotropic_theta, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    initial_theta = theta.detach().clone()

    base_classifier_output_log_prob = model(batch)
    base_classifier_output_prob = torch.exp(base_classifier_output_log_prob)
    base_classifier_output = torch.argmax(base_classifier_output_prob, dim=-1)

    criterion = torch.nn.NLLLoss()

    def my_loss(point):
        return criterion(model(point), base_classifier_output_local)

    hessian_batch = []

    for i, sample in enumerate(batch):
        base_classifier_output_local = base_classifier_output[i][0][0]

        loss = my_loss(sample[0][0])
        hessian = torch.autograd.functional.hessian(my_loss, sample[0][0])
        hessian_batch.append(torch.unsqueeze(hessian, 0))

    hessian_batch = torch.cat(hessian_batch, dim=0)

    eig_val, eig_vec = torch.linalg.eigh(hessian_batch)
    R = eig_vec

    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    R = torch.unsqueeze(R, 1)
    R_repeated = R.repeat(1, samples, 1, 1).view((batch_size * samples, 1, 2, 2))

    # solve iteratively by projected gradient ascend
    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)  # [0]

        noise = certificate.sample_noise(new_batch, theta_repeated)
        noise_t = torch.transpose(noise, 2, 3)

        rotated_noise = torch.matmul(R_repeated, noise_t)

        rotated_noise_t = torch.transpose(rotated_noise, 2, 3)

        rotated_noise = rotated_noise_t

        out = torch.exp(model(
            new_batch + rotated_noise
        )).reshape(batch_size, samples, -1).mean(dim=1)

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

        with torch.no_grad():
            torch.max(theta, initial_theta, out=theta)

    min_bound = isotropic_original * 0.85
    max_bound = isotropic_original * 1.15

    min_bound = torch.Tensor(min_bound).to(device)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)
    min_bound = torch.unsqueeze(min_bound, 1)

    max_bound = torch.Tensor(max_bound).to(device)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)
    max_bound = torch.unsqueeze(max_bound, 1)

    # theta = torch.clamp(theta, min=min_bound, max=max_bound)

    return theta, R
