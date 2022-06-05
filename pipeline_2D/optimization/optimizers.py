import torch
from ddsmoothing import Certificate, OptimizeSmoothingParameters
from optimization.optimization import optimize_isotropic_dds, ancer_optimization, rancer_optimization
from tqdm import tqdm


class OptimizeIsotropicSmoothingParameters(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        """Optimize isotropic smoothing parameters over a dataset given by
        the loader in test_loader.

        Args:
            model (torch.nn.Module): trained base model
            test_loader (torch.utils.data.DataLoader): dataset of inputs
            device (str, optional): device on which to perform the computations
        """
        super().__init__()
        self.model = model
        self.device = device
        self.loader = test_loader
        self.data_samples = 0

        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.data_samples += len(idx)

        self.log(
            "There are in total {} instances in the testloader".format(
                self.data_samples
            )
        )

    def save_theta(self, thetas: torch.Tensor, filename: str):
        """Save the optimized isotropic thetas

        Args:
            thetas (torch.Tensor): optimized thetas
            filename (str): filename to the folder where the thetas should be
                saved
        """
        torch.save(thetas, filename)
        self.log('Optimized smoothing parameters are saved')

    def save_same_predictions_indicators(self, same_predictions: torch.Tensor, filename: str):

        torch.save(same_predictions, filename)
        self.log('Same prediction indicators are saved')

    def run_optimization(
            self, certificate: Certificate, lr: float,
            theta_0: torch.Tensor, initial_same_prediction: torch.Tensor,
            iterations: int,
            num_samples: int, filename: str = './', same_predictions_file: str = './'
    ):
        """Run the Isotropic DDS optimization for the dataset

        Args:
            certificate (Certificate): instance of desired certification object
            lr (float, optional): optimization learning rate for Isotropic DDS
            theta_0 (torch.Tensor): initialization value per input of the test
                loader
            iterations (int): Description
            num_samples (int): number of samples per input and iteration
            filename (str, optional): name of the file of the saved thetas
        """
        print("inside isotropic_obj.run_optimization")

        theta_0 = theta_0.reshape(-1)

        assert torch.numel(theta_0) == self.data_samples, \
            "Dimension of theta_0 should be the number of " + \
            "examples in the testloader"

        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)

            thetas, same_prediction = optimize_isotropic_dds(
                model=self.model, batch=batch,
                certificate=certificate, learning_rate=lr,
                sig_0=theta_0[idx], iterations=iterations,
                samples=num_samples, device=self.device
            )

            theta_0[idx] = thetas.detach()
            initial_same_prediction[idx] = same_prediction.detach()

        self.save_theta(theta_0, filename)
        self.save_same_predictions_indicators(initial_same_prediction, same_predictions_file)


class AncerOptimizer(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.loader = test_loader
        self.num_samples = 0

        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.num_samples += len(idx)

        self.log(
            "There are in total {} instances in the testloader".format(
                self.num_samples
            )
        )

    def save_theta(
            self, thetas: torch.Tensor, idx: torch.Tensor, path: str,
            flag: str = 'test'
    ):
        """Save the optimized thetas at idx

        Args:
            thetas (torch.Tensor): optimized thetas
            idx (torch.Tensor): indices of the thetas of thetas to be
                saved
            path (str): path to the folder where the thetas should be
                saved
            flag (str, optional): flag to add to the output name
        """
        for i, j in enumerate(idx):
            torch.save(
                thetas[i],
                path + '/theta_' + flag + '_' + str(j.item()) + '.pt'
            )

        self.log(f'Optimized parameters at indices {idx} saved.')

    def run_optimization_ancer(
            self, isotropic_thetas: torch.Tensor, output_folder: str,
            iterations: int, certificate: Certificate, lr: float = 0.04,
            num_samples: int = 100, regularization_weight: float = 2
    ):
        print("Started running ANCER optimization")
        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            thetas = torch.ones_like(batch) * \
                     isotropic_thetas[idx].reshape(-1, 1, 1, 1)

            thetas = ancer_optimization(
                self.model, batch, certificate, lr, thetas, iterations,
                num_samples, kappa=regularization_weight,
                device=self.device
            )

            self.save_theta(thetas.detach(), idx, output_folder)


class RancerOptimizer(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.loader = test_loader
        self.num_samples = 0

        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.num_samples += len(idx)

        self.log(
            "There are in total {} instances in the testloader".format(
                self.num_samples
            )
        )

    def save_theta(
            self, thetas: torch.Tensor, idx: torch.Tensor, path: str,
            flag: str = 'test'
    ):
        """Save the optimized thetas at idx

        Args:
            thetas (torch.Tensor): optimized thetas
            idx (torch.Tensor): indices of the thetas of thetas to be
                saved
            path (str): path to the folder where the thetas should be
                saved
            flag (str, optional): flag to add to the output name
        """
        for i, j in enumerate(idx):
            torch.save(
                thetas[i],
                path + '/theta_' + flag + '_' + str(j.item()) + '.pt'
            )

        self.log(f'Optimized parameters at indices {idx} saved.')

    def save_rotation_matrix(
            self, rotation_matrices: torch.Tensor, idx: torch.Tensor, path: str,
            flag: str = 'test'
    ):
        for i, j in enumerate(idx):
            # print("R before saving: ", rotation_matrices[i].shape)
            torch.save(
                rotation_matrices[i],
                path + '/rotation_matrix_' + flag + '_' + str(j.item()) + '.pt'
            )

        self.log(f'Rotation matrices at indices {idx} saved.')

    def run_optimization_hessian(
            self, isotropic_thetas: torch.Tensor, output_folder: str,
            iterations: int, certificate: Certificate, rotation_matrices_folder: str, lr: float = 0.04,
            num_samples: int = 100, regularization_weight: float = 2
    ):
        print("Started running RANCER optimization")
        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            thetas = torch.ones_like(batch) * \
                     isotropic_thetas[idx].reshape(-1, 1, 1, 1)

            thetas, rotation_matrices = rancer_optimization(
                self.model, batch, certificate, lr, thetas, iterations,
                num_samples, kappa=regularization_weight,
                device=self.device
            )

            # save the optimized thetas
            self.save_theta(thetas.detach(), idx, output_folder)

            # save rotation matrices
            self.save_rotation_matrix(rotation_matrices.detach(), idx, rotation_matrices_folder)
