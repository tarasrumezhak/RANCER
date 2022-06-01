import torch
from tqdm import tqdm
import time

from ddsmoothing import OptimizeSmoothingParameters, Certificate
from .optimization import optimize_rancer



class OptimizeRANCERSmoothingParameters(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        """Optimize anisotropic smoothing parameters over a dataset given by
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

    def save_rotation_matrix(self, rotation_matrices: torch.Tensor, idx: torch.Tensor, path: str,
                             flag: str = 'test'
                             ):
        for i, j in enumerate(idx):
            # print("R before saving: ", rotation_matrices[i].shape)
            torch.save(
                rotation_matrices[i],
                path + '/rotation_matrix_' + flag + '_' + str(j.item()) + '.pt'
            )

    def run_optimization(
            self, isotropic_thetas: torch.Tensor, output_folder: str,
            iterations: int, certificate: Certificate, lr: float = 0.04,
            num_samples: int = 100, regularization_weight: float = 2
    ):
        """Run the ANCER optimization for the dataset

        Args:
            isotropic_theta (torch.Tensor): initialization isotropic value per
                input in batch
            output_folder (str): path to the folder where the thetas should be
                saved
            iterations (int): number of iterations to run the optimization
            certificate (Certificate): instance of desired certification object
            lr (float, optional): optimization learning rate for ANCER
            num_samples (int): number of samples per input and iteration
            regularization_weight (float, optional): relaxation hyperparameter
        """
        # a = 0
        for batch, _, idx in tqdm(self.loader):
            # print("a = ", a)
            # if a <= 56:
            #   a += 1
            #   continue
            # a += 1
            batch = batch.to(self.device)
            thetas = torch.ones_like(batch) * \
                     isotropic_thetas[idx].reshape(-1, 1, 1, 1)

            start = time.time()
            thetas, rotation_matrices = optimize_rancer(
                self.model, batch, certificate, lr, thetas, iterations,
                num_samples, kappa=regularization_weight,
                device=self.device
            )
            end = time.time()
            print("rancer time: ", end - start)

            print("here")
            print("thetas: ", thetas.shape)
            print("rotation_matrices: ", rotation_matrices.shape)


            # # label = 3

            # # =========================================
            # smoothed_classifier = Smooth(self.model, 10, thetas, rotation_matrices, certificate)
            # print("smoothed_classifier initialized")
            # prediction, gap = smoothed_classifier.certify(batch, 100, 100, 0.001, 1)
            # # correct = int(prediction == label)

            # # Computing volumes http://oaji.net/articles/2014/1420-1415594291.pdf
            # radius = thetas.min().item() * gap

            # print("radius: ", radius)
            # # print("correct: ", correct)
            # # ff
            # # =========================================



            rotation_matrices_folder = r"../rotation_matrices_ce_loss"

            # save the optimized thetas
            self.save_theta(thetas.detach(), idx, output_folder)
            self.save_rotation_matrix(rotation_matrices.detach(), idx, rotation_matrices_folder)

