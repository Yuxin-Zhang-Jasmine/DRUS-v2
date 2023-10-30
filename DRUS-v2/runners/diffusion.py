import os
from math import sqrt
import numpy as np
import torch
from functions.ckpt_util import download
from functions.denoising import efficient_generalized_steps
from guided_diffusion.script_util import create_model
import mat73
from scipy.io import savemat


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None

        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        if self.config.model.use_fp16:
            model.convert_to_fp16()
        ckpt = os.path.join(self.args.log_path, self.args.ckpt)
        print('Path of the current ckpt: ' + ckpt)
        if not os.path.exists(ckpt):
            print('The model does not exist, downloading an Imagenet 3c ckpt...')
            download(
                'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                    self.config.data.image_size, self.config.data.image_size), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        model = torch.nn.DataParallel(model)
        self.sample_sequence(model, cls_fn)

    def sample_sequence(self, model, cls_fn=None):
        args, config = self.args, self.config
        print("data channels : " + str(self.config.data.channels))
        print("model in_channels : " + str(self.config.model.in_channels))
        print('The corresponding MATLAB path: ' + self.args.matlab_path)

        # ** define the dasLst and the gammaLst **
        # ------with the fixed gammas, it varys from 5 to 1000 (the results are saved in 1_itSensitivity)--------
        timestepsLst = list(range(5, 70, 5)) + list(range(70, 110, 10)) + [120, 140, 160, 200, 250, 333, 500, 1000]
        tmp = len(timestepsLst)
        timestepsLst = timestepsLst * 4
        timestepsLst = timestepsLst * 13
        
        dasLst = ['simu_reso'] * tmp + ['simu_cont'] * tmp + ['expe_reso'] * tmp + ['expe_cont'] * tmp
        dasLst = dasLst * 13
        if self.args.deg == "DRUSdeno":
            gammaLst = [0.11]*tmp + [0.12]*tmp + [0.16]*tmp + [0.025]*tmp
        elif self.args.deg == "DRUS":	    
            gammaLst = [30]*tmp + [65]*tmp + [90]*tmp + [55]*tmp
        else:
            raise ValueError
        gammaLst = gammaLst * 13
        # -------------------------------------------------------------------------------------------------------

        # ------with the fixed gammas and it=50, repeat the picmus vitro restoration 500 times ------------------
        # --------------------------------(the results are saved in 2_histogram)---------------------------------
        timestepsLst = [50]
        tmp = len(timestepsLst)
        timestepsLst = timestepsLst * 4
        timestepsLst = timestepsLst * 500

        dasLst = ['simu_reso'] * tmp + ['simu_cont'] * tmp + ['expe_reso'] * tmp + ['expe_cont'] * tmp
        dasLst = dasLst * 500
        if self.args.deg == "DRUSdeno":
            gammaLst = [0.11] * tmp + [0.12] * tmp + [0.16] * tmp + [0.025] * tmp
        elif self.args.deg == "DRUS":
            gammaLst = [30] * tmp + [65] * tmp + [90] * tmp + [55] * tmp
        else:
            raise ValueError
        gammaLst = gammaLst * 500
        # -------------------------------------------------------------------------------------------------------

        # ------with it=50, repeat the picmus vivo restoration 20 times while search for the best gammas---------
        # --------------------------------(the results are saved in 5_picmusVivoImg/both)------------------------
        timestepsLst = [50]*20*20
        dasLst = ['expe_cross']*10 + ['expe_long']*10
        dasLst = dasLst * 20
        if self.args.deg == "DRUSdeno":
            gammaLst = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]*2*20
        elif self.args.deg == "DRUS":
            gammaLst = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        3, 4, 5, 6, 7, 0.5, 1, 1.5, 2, 2.5]*20
        else:
            raise ValueError
        # -------------------------------------------------------------------------------------------------------

        # -with it=50, repeat the additional Carotid Cross restoration 20 times while search for the best gammas-
        # --------------------------------(the results are saved in 6_additionalVivoImg)-------------------------
        timestepsLst = [50] * 5 * 5 * 20
        dasLst = ['test1_cross'] * 5 + ['test2_cross'] * 5 + ['test3_cross'] * 5 + ['test4_cross'] * 5 + [
            'test5_cross'] * 5
        dasLst = dasLst * 20

        if self.args.deg == "DRUS":
            gammaLst = gammaLst + [12, 16, 20, 24, 28] * 5
        elif self.args.deg == "DRUSdeno":
            gammaLst = gammaLst + [0.01, 0.02, 0.03, 0.04, 0.05] * 5
        else:
            raise ValueError
        gammaLst = gammaLst * 20
        # -------------------------------------------------------------------------------------------------------

        # ---------------------with it=50, repeat the simulated fetus restoration 20 times ----------------------
        # --------------------------------(the results are saved in 7_discussionSimu)----------------------------
        phantomIdies = ['1', '2', '3']
        phantomIdx = phantomIdies[1]  # select from [0 - kidney | | 1 - fetus | | 2 - simuComplex]
        timestepsLst = [50] * 7 * 20

        dasLst = ['simulation_Hty_2_mu', 'simulation_Hty_2_mu', 'simulation_Hty_2_mu', 'simulation_Hty_2_mu',
                  'simulation_Hty_2_bo', 'simulation_Hty_2_as', 'simulation_Hty_2_ap']
        if self.args.deg == "HtH":
            dasLst = dasLst * 20
            gammaLst = [0, 0.2, 0.5, 1, 1 * 13.6, 1 * 13.6, 1 * 13.6] * 20
        else:
            raise ValueError
        # -------------------------------------------------------------------------------------------------------

        # ** get SVD results of the model matrix **
        print(f'Loading the SVD of the degradation matrix (' + self.args.deg + ')')
        if self.args.deg in ["DRUS", "HtH"]:
            from functions.svd_replacement import ultrasound0

            if self.args.deg == "DRUS":
                svdPath = self.args.matlab_path + 'SVD/02_picmus/'
                Up = torch.from_numpy(mat73.loadmat(svdPath + 'Ud.mat')['Ud'])
                lbdp = torch.from_numpy(mat73.loadmat(svdPath + 'Sigma.mat')['Sigma'])
                Vp = torch.from_numpy(mat73.loadmat(svdPath + 'Vd.mat')['Vd'])

            elif self.args.deg == "HtH":
                svdPath = self.args.matlab_path + 'SVD/01_simulation/'
                Up = torch.from_numpy(mat73.loadmat(svdPath + 'V.mat')['V'])
                lbdp = torch.from_numpy(mat73.loadmat(svdPath + 'LBD.mat')['LBD'])
                Vp = torch.from_numpy(mat73.loadmat(svdPath + 'Vp.mat')['Vp'])
            else:
                raise ValueError
            H_funcs = ultrasound0(config.data.channels, Up, lbdp, Vp, self.device)

        elif self.args.deg in ["DRUSdeno"]:
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)

        else:
            print("ERROR: problem_model (--deg) type not supported")
            quit()

        idx_so_far = 0
        print(f'Start restoration')
        for _ in range(len(dasLst)):
            timesteps = timestepsLst[idx_so_far]
            dasSaveName = dasLst[idx_so_far] + '.mat'
            # ** load the observation y_0 **
            if self.args.deg in ("DRUS", "DRUSdeno"):
                if self.args.deg in "DRUS":
                    y_0 = torch.from_numpy(mat73.loadmat(self.args.matlab_path + 'Observation/02_picmus/DRUS/' + dasSaveName)['By'])
                elif self.args.deg == "DRUSdeno":
                    y_0 = torch.from_numpy(mat73.loadmat(self.args.matlab_path + 'Observation/02_picmus/Deno/' + dasSaveName)['By'])
                else:
                    raise ValueError
                y_0 = (y_0.view(1, -1)).repeat(1, config.data.channels).to(self.device)

                gamma = gammaLst[idx_so_far] * 1
                if gamma <= 0:
                    gamma = 0.1
                if self.config.model.in_channels == 3:
                    gamma = gamma * sqrt(3)

            elif self.args.deg in ("HtH"):
                dataPath = self.args.matlab_path + 'Observation/01_simulation/' + phantomIdx + '/'
                if self.args.deg == "HtH":
                    y_0 = torch.from_numpy(mat73.loadmat(dataPath + dasSaveName)['o_Hty'])
                else:
                    raise ValueError
                y_0 = (y_0.view(1, 65536*config.data.channels)).to(self.device)
                gamma = gammaLst[idx_so_far]

            else:
                raise ValueError

            # ===========================================================================================================

            # ** Begin DDRM **
            x = torch.randn(
                y_0.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                x, _ = self.sample_image(x, model, H_funcs, y_0, gamma, last=False, cls_fn=cls_fn, timesteps=timesteps)
            for i in [-1]:  # range(len(x)):
                for j in range(x[i].size(0)):
                    # ** save the DDRM restored image as .mat **
                    savemat(os.path.join(self.args.image_folder, f"{idx_so_far + j + 1}_{i}.mat"),
                            {'x': x[i][j].detach().cpu().numpy()})

            idx_so_far += y_0.shape[0]  # iterate multiple images
            print(f'Finish {idx_so_far}')

    def sample_image(self, x, model, H_funcs, y_0, gamma, last=False, cls_fn=None, classes=None, timesteps=50):
        skip = self.num_timesteps // timesteps
        seq = range(0, self.num_timesteps, skip)
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, gamma, etaB=self.args.etaB,
                                        etaA=self.args.eta, etaC=self.args.eta, cls_fn=cls_fn,
                                        classes=classes)
        if last:
            x = x[0][-1]
        return x
