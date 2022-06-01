import torch
import numpy as np
from torch.autograd import Variable
from ddsmoothing.certificate import Certificate


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

    isotropic_original = isotropic_theta.clone().cpu().detach().numpy()[:, 0, 0, 0]  # [0][0][0][0]

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
    print("base_classifier_output: ", base_classifier_output)

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

    thetas_min = []
    thetas_max = []
    gaps = []
    proxy_radiuses = []

    iterations = 100

    # solve iteratively by projected gradient ascend
    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)

        # Reparameterization trick
        noise_orig = certificate.sample_noise(new_batch, theta_repeated)

        # print("noise_orig shape: ", noise_orig.shape)
        # print("noise_orig norm: ", torch.linalg.norm(noise_orig))

        noise = noise_orig.reshape((batch_size, samples, 3072))
        # print("noise shape: ", noise.shape)
        noise_t = torch.transpose(noise, 2, 1)
        rotated_noise_t = torch.matmul(R, noise_t)
        rotated_noise = torch.transpose(rotated_noise_t, 2, 1)
        rotated_noise = rotated_noise.reshape((batch_size, samples, 3, 32, 32))
        rotated_noise = rotated_noise.reshape((batch_size * samples, 3, 32, 32))


        # print("rotated_noise shape: ", rotated_noise.shape)
        # print("rotated_noise norm: ", torch.linalg.norm(rotated_noise))
        # ff
        # print("rotated_noise shape: ", rotated_noise.shape)

        # print("noise diff: ", torch.max(noise_orig - rotated_noise))

        out = model(
            new_batch + rotated_noise
        ).reshape(batch_size, samples, -1).mean(dim=1)

        # plt.imshow(new_batch[0].detach().cpu().numpy())
        # plt.show()

        # plt.imshow(np.dstack((batch[0][0].cpu().detach().numpy(), batch[0][1].cpu().detach().numpy(), batch[0][2].cpu().detach().numpy())))
        # plt.show()

        # print("showed")


        # print("out: ",  torch.round(out, decimals=3))

        # out2 = model(
        #     new_batch + noise_orig
        # ).reshape(batch_size, samples, -1).mean(dim=1)
        # print("out2: ",  torch.round(out2, decimals=3))

        # smoothed_classifier = Smooth(model, 10, theta, R, certificate)
        # counts = smoothed_classifier._sample_noise(batch, num=100, batch_size=1, device=device)
        # print("counts: ", np.around(counts/100, decimals=3))

        vals, _ = torch.topk(out, 2)
        gap = certificate.compute_proxy_gap(vals)

        # gaps.append(gap.detach().cpu())

        prod = torch.prod(
            (theta.reshape(batch_size, -1))**(1/img_size), dim=1)
        proxy_radius = prod * gap


        # proxy_radiuses.append(proxy_radius.detach().cpu())

        radius_maximizer = - (
            proxy_radius.sum() +
            kappa *
            (torch.min(theta.view(batch_size, -1), dim=1).values*gap).sum()
        )
        radius_maximizer.backward()
        optimizer.step()
        optimizer.zero_grad()

        # project to the initial theta
        with torch.no_grad():
            torch.max(theta, initial_theta, out=theta)

        # thetas_min.append(theta.min().detach().cpu())
        # thetas_max.append(theta.max().detach().cpu())

    # print("base_classifier_output: ", base_classifier_output)
    # print("out: ", out.argmax(1))

    # print("proxy_radius: ", proxy_radius, proxy_radius.sum())

    # print("thetas_min: ", thetas_min)

    # plt.plot(np.array(thetas_min))
    # plt.title("Thetas min")
    # # plt.show()
    # plt.savefig("visual/thetas_min_identity5.png")
    # plt.close()

    # plt.plot(np.array(thetas_max))
    # plt.title("Thetas max")
    # # plt.show()
    # plt.savefig("visual/thetas_max_identity5.png")
    # plt.close()

    # plt.plot(np.array(gaps))
    # plt.title("Gaps")
    # # plt.show()
    # plt.savefig("visual/gaps_identity5.png")
    # plt.close()

    # plt.plot(np.array(proxy_radiuses))
    # plt.title("Gaps")
    # # plt.show()
    # plt.savefig("visual/proxy_radiuses_identity5.png")
    # plt.close()

    # ff

    theta = torch.clamp(theta, min=min_bound, max=max_bound)

    # print("R before return: ", R.shape)

    return theta, R
