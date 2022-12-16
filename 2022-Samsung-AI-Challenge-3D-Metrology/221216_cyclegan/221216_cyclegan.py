import torch
import torch.nn as nn
import numpy as np

class resnet_block(nn.Module):
    def __init__(self, dim):
        super(resnet_block, self).__init__()

        _resnet_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim)
        ]

        self.layer = nn.Sequential(*_resnet_block)

    def forward(self, x):
        out = self.layer(x) + x
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_ch):
        super(ResnetGenerator, self).__init__()

        self.init_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_ch, 16, kernel_size=7, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(True))

        self.donw_sampling_layer1 = nn.Sequential(
            # down sampling
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.donw_sampling_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True))

        self.res_block = nn.Sequential(*[resnet_block(64) for i in range(3)])

        self.up_samplig_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.up_samplig_layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, input_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.model(x)
        out1 = self.init_layer(x)
        out2 = self.donw_sampling_layer1(out1)
        out3 = self.donw_sampling_layer2(out2)
        out4 = self.res_block(out3)
        out5 = self.up_samplig_layer1(out4)
        out6 = self.up_samplig_layer2(out5)
        out7 = self.output_layer(out6)
        return out7

    def set_requires_grad(self, mode):
        for param in self.parameters():
            param.requires_grad = mode

class PatchGanDiscriminator(nn.Module):
    def __init__(self, input_ch):
        super(PatchGanDiscriminator, self).__init__()

        model = [
            nn.Conv2d(input_ch, 16, kernel_size=7, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.2),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),  # 1
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),  # 2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.2),

            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def set_requires_grad(self, mode):
        for param in self.parameters():
            param.requires_grad = mode

    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, gan_mode):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan_gp':
            self.loss = None

    def __call__(self, prediction, target_is_real):
        if self.gan_mode == 'lsgan':
            if target_is_real:
                target_tensor = self.real_label  # .to(self.device)
            else:
                target_tensor = self.fake_label  # .to(self.device)

            target_tensor = target_tensor.expand_as(prediction)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan_gp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def _gradient_penalty(netD, real_data, fake_data, type="mixed", constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=real_data.device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def train_generator():
    ...

def train_discriminator():
    ...

def valid_model():
    ...

def model_training(model, **k):
    best_rmse = 9999
    best_loss = 9999

    for epoch in range(k['epochs']):
        for step_i , data_tuple in enumerate(k['train_dataloader']):
            train_discriminator()
            if step_i % k['critic_iter']:
                train_generator(model,)

        rmse_loss = valid_model(model, k['valid_dataloder'])
        if rmse_loss < best_rmse:
            best_rmse = best_loss
            torch.save(model, './best_model.pth')


