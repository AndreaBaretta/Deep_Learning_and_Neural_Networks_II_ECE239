import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
# from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        # YOUR CODE HERE:
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        norm_t_s = torch.linspace(0, 1, T)
        beta_t = beta_1 * (1 - norm_t_s) + beta_T * norm_t_s
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1/torch.sqrt(alpha_t)
        lo = alpha_t.repeat(T,1).tril()
        up = torch.ones_like(lo).triu(diagonal=1)
        alpha_t_bar = (up + lo).prod(axis=1)
        sqrt_alpha_bar = alpha_t_bar.sqrt()
        sqrt_oneminus_alpha_bar = (1 - alpha_t_bar).sqrt()

        # ==================================================== #
        return {
            'beta_t': beta_t[t_s - 1],
            'sqrt_beta_t': sqrt_beta_t[t_s - 1],
            'alpha_t': alpha_t[t_s - 1],
            'sqrt_alpha_bar': sqrt_alpha_bar[t_s - 1],
            'oneover_sqrt_alpha': oneover_sqrt_alpha[t_s - 1],
            'alpha_t_bar': alpha_t_bar[t_s - 1],
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar[t_s - 1]
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  

        one_hot = torch.zeros(len(conditions), 10)
        one_hot[torch.arange(len(conditions)), conditions] = 1
        mask_p = torch.rand(len(conditions)) > self.dmconfig.mask_p
        one_hot *= mask_p[:, None]
        t_s = torch.randint(0, self.dmconfig.T, len(conditions))
        epsilon = torch.normal(0, 1, *images.shape)
        sched = self.scheduler(t_s)
        x_t = sched['sqrt_alpha_bar']*images + sched['sqrt_oneminus_alpha_bar']*epsilon
        epsilon_theta = self.network.forward(x_t, t_s, one_hot)
        noise_loss = self.loss_fn(epsilon_theta, epsilon)

        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        pass



        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images