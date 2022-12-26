import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import itertools
import cv2, PIL
import os, glob
import csv, platform

current_os = platform.system()
if current_os == "Linux":
    _path = '/home/kji/workspace/jupyter_kji/samsumg_sem_dataset'
    cfg = {
        'device': "cuda:5",
        "db_path": _path,
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'num_workers': 4,
        'n_fold': 5
    }
elif current_os == "Windows":
    _path = 'D:/git_repos/samsung_sem'
    cfg = {
        'device': "cuda:0",
        "db_path": _path,
        'epochs': 100,
        'batch_size': 4,
        'lr': 0.0002,
        'num_workers': 0,
        'n_fold': 5
    }


'''
    cnn classifier를 사용해서 case를 분류함
    Generator를 4가지 case를 나눠서 학습함
'''

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

class cycleGAN_model(nn.Module):
    def __init__(self, input_ch=3,
                 optim_lr=0.0002,
                 gan_mode='lsgan',
                 guided=False):
        import itertools

        super(cycleGAN_model, self).__init__()
        self.gan_mode = gan_mode
        self.guided = guided

        self.Gen = nn.ModuleDict({
            'A': ResnetGenerator(input_ch),
            'B': ResnetGenerator(input_ch)
        })

        # wandb.watch(self.Gen['A'], log='all')
        # wandb.watch(self.Gen['B'], log='all')

        self.Dis = nn.ModuleDict({
            'A': PatchGanDiscriminator(input_ch),
            'B': PatchGanDiscriminator(input_ch)
        })

        # wandb.watch(self.Dis['A'], log='all')
        # wandb.watch(self.Dis['B'], log='all')

        self.optimizer = {
            'G': torch.optim.Adam(itertools.chain(self.Gen['A'].parameters(), self.Gen['B'].parameters()), lr=optim_lr,
                                  betas=(0.5, 0.999)),
            'D_A': torch.optim.Adam(self.Dis['A'].parameters(), lr=optim_lr,
                                  betas=(0.5, 0.999)),
            'D_B': torch.optim.Adam(self.Dis['B'].parameters(), lr=optim_lr,
                                  betas=(0.5, 0.999))
        }

        self.schedular = {
            'G': torch.optim.lr_scheduler.LambdaLR(self.optimizer['G'], lr_lambda=lambda epoch: 0.95 ** epoch),
            'D_A': torch.optim.lr_scheduler.LambdaLR(self.optimizer['D_A'], lr_lambda=lambda epoch: 0.95 ** epoch),
            'D_B': torch.optim.lr_scheduler.LambdaLR(self.optimizer['D_B'], lr_lambda=lambda epoch: 0.95 ** epoch)
        }

        self.criterion = nn.ModuleDict({
            'cycle': nn.L1Loss(),
            'idt': nn.L1Loss(),
            'gan': GANLoss(self.gan_mode),
            'mse': nn.MSELoss(),
            'guided': nn.L1Loss()
        })

        self.lambda_idt = 0.5
        self.lambda_A = 10.0
        self.lambda_B = 10.0

    def forward(self, data_A, data_B, mode: str):
        if mode == 'gen':
            A_out = self.Gen['A'](data_A)
            B_out = self.Gen['B'](data_B)
        elif mode == 'dis':
            A_out = self.Dis['A'](data_A)
            B_out = self.Dis['B'](data_B)
        else:
            raise None
        return A_out, B_out

    def model_train_discriminator(self, real_A, real_B):
        self.train()

        fake_B, fake_A = self(real_A, real_B, 'gen')

        self.set_requires_grad('dis', True)

        self.optimizer['D_B'].zero_grad()

        pred_real_B, pred_real_A = self(real_B, real_A, 'dis')  # netA netB
        pred_fake_B, pred_fake_A = self(fake_B.detach(), fake_A.detach(), 'dis')

        # Discriminator B update
        loss_D_B_Real = self.criterion['gan'](pred_real_A, True)
        loss_D_B_fake = self.criterion['gan'](pred_fake_A, False)

        if self.gan_mode == 'lsgan':
            loss_D_B = (loss_D_B_fake + loss_D_B_Real) * 0.5
        elif self.gan_mode == 'wgan_gp':
            gradient_penalty_B = _gradient_penalty(self.Dis['B'], real_A, fake_A.detach())
            loss_D_B = loss_D_B_fake + loss_D_B_Real + gradient_penalty_B[0]

        loss_D_B.backward()
        self.optimizer['D_B'].step()

        # Discriminator A update
        self.optimizer['D_A'].zero_grad()

        loss_D_A_Real = self.criterion['gan'](pred_real_B, True)
        loss_D_A_fake = self.criterion['gan'](pred_fake_B, False)

        if self.gan_mode == 'lsgan':
            loss_D_A = (loss_D_A_Real + loss_D_A_fake) * 0.5
        elif self.gan_mode == 'wgan_gp':
            gradient_penalty_A = _gradient_penalty(self.Dis['A'], real_B, fake_B.detach())
            loss_D_A = loss_D_A_Real + loss_D_A_fake + gradient_penalty_A[0]

        loss_D_A.backward()
        self.optimizer['D_A'].step()

        loss_dic = {'dis_a': loss_D_A.item(),
                    'dis_b': loss_D_B.item()}

        return loss_dic

    def model_train_generator(self, real_A, real_B):
        self.train()

        fake_B, fake_A = self(real_A, real_B, 'gen')
        rec_B, rec_A = self(fake_A, fake_B, 'gen')

        self.set_requires_grad('dis', False)
        self.optimizer['G'].zero_grad()

        idt_A, idt_B = self(real_B, real_A, 'gen')

        loss_idt_A = self.criterion['idt'](idt_A, real_B) * self.lambda_B * self.lambda_idt
        loss_idt_B = self.criterion['idt'](idt_B, real_A) * self.lambda_A * self.lambda_idt

        dis_A_fake_B, dis_B_fake_A = self(fake_B, fake_A, 'dis')  # dis_A(fake_B) / dis_B(fake_A)

        loss_G_A = self.criterion['gan'](dis_A_fake_B, True)
        loss_G_B = self.criterion['gan'](dis_B_fake_A, True)

        loss_cycle_A = self.criterion['cycle'](rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.criterion['cycle'](rec_B, real_B) * self.lambda_B

        # Guied Loss (paired)
        if self.guided:
            loss_guided_A = self.criterion['guided'](fake_B, real_B)
            loss_guided_B = self.criterion['guided'](fake_A, real_A)
        else:
            loss_guided_A = 0
            loss_guided_B = 0
        ##########

        loss_Gen = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_guided_A + loss_guided_B
        loss_Gen.backward()

        self.optimizer['G'].step()

        loss_dic = {'gen': loss_Gen.item()}

        inference_image = {
            'real_a': real_A,
            'real_b': real_B,
            'atob_fake': fake_B,
            'btoa_fake': fake_A,
            'rec_a': rec_A,
            'rec_b': rec_B
        }

        return loss_dic, {key: self.tensortonp(inference_image[key]) for key in inference_image}

    def model_valid(self, real_A, real_B):
        self.eval()

        with torch.no_grad():
            fake_B, fake_A = self(real_A, real_B, 'gen')

            true = (real_B * 255).type(torch.uint8).float()
            fake_true = (fake_B * 255).type(torch.uint8).float()
            rmse_loss = torch.sqrt(self.criterion['mse'](fake_true, true))

        img_dict = {
            'real_A': real_A,
            'fake_B': fake_B,

            'real_B': real_B,
            'fake_A': fake_A,
        }

        return rmse_loss.item(), {key: self.tensortonp(img_dict[key]) for key in img_dict}

    def tensortonp(self, tensor):
        return (tensor.detach().cpu().numpy() * 255).astype(np.uint8)

    def set_requires_grad(self, net_type='dis', mode=True):
        if net_type == 'gen':
            net_dic = self.Gen
        elif net_type == 'dis':
            net_dic = self.Dis

        for key in net_dic:
            net_dic[key].set_requires_grad(mode)

    def schedular_step(self):
        self.schedular['G'].step()
        self.schedular['D_A'].step()
        self.schedular['D_B'].step()

    def model_save(self, PATH):
        temp_dict = {}
        key_list = [key for key in self.__dict__.keys() if not '_' in key[0]]
        key_list.extend([key for key in self.__dict__['_modules'].keys()])

        for key in key_list:
            if hasattr(self, key):
                value = getattr(self, key)
                if isinstance(value, dict):
                    if not key in temp_dict:
                        temp_dict[key] = {}
                    for sub_key in value.keys():
                        if not sub_key in temp_dict[key]:
                            temp_dict[key][sub_key] = value[sub_key].state_dict()
                elif isinstance(value, nn.ModuleDict):
                    if not key in temp_dict:
                        temp_dict[key] = value.state_dict()
                else:
                    if not key in temp_dict:
                        temp_dict[key] = value

        torch.save(temp_dict, PATH)

    def model_load(self, PATH, device):
        state_dict = torch.load(PATH, map_location=device)

        for cls_key in state_dict.keys():
            if hasattr(self, cls_key):
                value = getattr(self, cls_key)
                if isinstance(value, dict):
                    for sub_key in value.keys():
                        value[sub_key].load_state_dict(state_dict[cls_key][sub_key])
                elif isinstance(value, nn.ModuleDict):
                    value.load_state_dict(state_dict[cls_key])
                else:
                    setattr(self, cls_key, state_dict[cls_key])


def get_img_list(abs_path):
    # abs_path = '/home/kji/workspace/jupyter_kji/samsumg_sem_dataset'

    # Dataset path
    sim_depth_path = os.path.join(abs_path, 'simulation_data/Depth')
    sim_sem_path = os.path.join(abs_path, 'simulation_data/SEM')

    train_path = os.path.join(abs_path, 'train')

    # only Test
    test_path = os.path.join(abs_path, 'test/SEM')

    sim_depth_img_path_dic = dict()
    for case in os.listdir(sim_depth_path):
        if not case in sim_depth_img_path_dic:
            sim_depth_img_path_dic[case] = []
        for folder in os.listdir(os.path.join(sim_depth_path, case)):
            img_list = glob.glob(os.path.join(sim_depth_path, case, folder, '*.png'))
            for img in img_list:
                sim_depth_img_path_dic[case].append(img)
                sim_depth_img_path_dic[case].append(img)

    sim_sem_img_path_dic = dict()
    for case in os.listdir(sim_sem_path):
        if not case in sim_sem_img_path_dic:
            sim_sem_img_path_dic[case] = []
        for folder in os.listdir(os.path.join(sim_sem_path, case)):
            img_list = glob.glob(os.path.join(sim_sem_path, case, folder, '*.png'))
            sim_sem_img_path_dic[case].extend(img_list)

    train_avg_depth = dict()
    with open(os.path.join(train_path, "average_depth.csv"), 'r') as csvfile:
        temp = csv.reader(csvfile)
        for idx, line in enumerate(temp):
            if idx > 0:
                depth_key, site_key = line[0].split('_site')
                depth_key = depth_key.replace("d", "D")
                site_key = "site" + site_key
                if not depth_key in train_avg_depth:
                    train_avg_depth[depth_key] = dict()

                train_avg_depth[depth_key][site_key] = float(line[1])

    train_img_path_dic = dict()
    for depth in os.listdir(os.path.join(train_path, "SEM")):
        if not depth in train_img_path_dic:
            train_img_path_dic[depth] = []
        for site in os.listdir(os.path.join(train_path, "SEM", depth)):
            img_list = glob.glob(os.path.join(train_path, "SEM", depth, site, "*.png"))
            train_img_path_dic[depth].extend([[temp_img, train_avg_depth[depth][site]] for temp_img in img_list])

    test_img_path_list = glob.glob(os.path.join(test_path, "*.png"))

    result_dic = dict()
    result_dic['sim'] = dict()
    result_dic['sim']['sem'] = sim_sem_img_path_dic
    result_dic['sim']['depth'] = sim_depth_img_path_dic
    result_dic['train'] = train_img_path_dic
    result_dic['test'] = np.array(test_img_path_list)
    result_dic['train_avg_depth'] = train_avg_depth

    return result_dic

result_dic = get_img_list(cfg['db_path'])

'''
train sem ->  sim sem -> sim depth

case 별로 dataset을 나눠야됨.
'''

class gan_dataset(Dataset):
    def __init__(self, a_data_path, b_data_path, transform=None):
        super(gan_dataset, self).__init__()
        self.a_data_path = a_data_path
        self.b_data_path = b_data_path
        self.transform = transform

        self.a_size = len(a_data_path)
        self.b_size = len(b_data_path)

    def __getitem__(self, idx):
        if self.a_size > self.b_size:
            a_idx = idx
            b_idx = idx % self.b_size
        else:
            a_idx = idx % self.a_size
            b_idx = idx
        if isinstance(self.a_data_path[a_idx], str):
            a_path = self.a_data_path[a_idx]
        elif isinstance(self.a_data_path[a_idx], list):
            a_path = self.a_data_path[a_idx][0]

        if isinstance(self.b_data_path[b_idx], str):
            b_path = self.b_data_path[b_idx]
        elif isinstance(self.b_data_path[b_idx], list):
            b_path = self.b_data_path[b_idx][0]

        a_img = PIL.Image.open(a_path).convert("L")
        b_img = PIL.Image.open(b_path).convert("L")

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

        a_img = (np.array(a_img) / 255.)
        a_img = a_img.reshape(1, *a_img.shape).astype(np.float32)
        b_img = (np.array(b_img) / 255.)
        b_img = b_img.reshape(1, *b_img.shape).astype(np.float32)

        return a_img, b_img

    def __len__(self):
        return max(len(self.a_data_path), len(self.b_data_path))

def create_dataloader(a_key, b_key, t_ratio, result_dic, case=1):
    if 'sim' in a_key:
        a_list = result_dic['sim'][a_key.split('_')[-1]][f"Case_{case}"]
    else:
        a_list = result_dic['train'][f"Depth_{100 + 10 * case}"]
    if 'sim' in b_key:
        b_list = result_dic['sim'][b_key.split('_')[-1]][f"Case_{case}"]
    else:
        b_list = result_dic['train'][f"Depth_{100 + 10 * case}"]

    horizon_transform = transforms.RandomHorizontalFlip(1.0)
    rotate_transform = transforms.RandomRotation((180, 180))
    vertical_transform = transforms.RandomVerticalFlip(1.0)

    a_train_data_size = int(len(a_list) * t_ratio)
    b_train_data_size = int(len(b_list) * t_ratio)

    train_dataset = gan_dataset(a_list[:a_train_data_size], b_list[:b_train_data_size], None) + \
                    gan_dataset(a_list[:a_train_data_size], b_list[:b_train_data_size], horizon_transform) + \
                    gan_dataset(a_list[:a_train_data_size], b_list[:b_train_data_size], rotate_transform) + \
                    gan_dataset(a_list[:a_train_data_size], b_list[:b_train_data_size], vertical_transform)

    valid_dataset = gan_dataset(a_list[a_train_data_size:], b_list[b_train_data_size:], None) + \
                    gan_dataset(a_list[a_train_data_size:], b_list[b_train_data_size:], horizon_transform) + \
                    gan_dataset(a_list[a_train_data_size:], b_list[b_train_data_size:], rotate_transform) + \
                    gan_dataset(a_list[a_train_data_size:], b_list[b_train_data_size:], vertical_transform)

    return DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True), \
           DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)


from tqdm.auto import tqdm
# import wandb

def valid(model, valid_dataloader, device):
    rmse_list = []
    img_mean_list = []
    for step_i, data_tuple in enumerate(valid_dataloader):
        real_a = data_tuple[0].to(device, non_blocking=True)
        real_b = data_tuple[1].to(device, non_blocking=True)

        rmse_loss, img_dict = model.model_valid(real_a, real_b)
        rmse_list.append(rmse_loss)
        if step_i == 0:
            img_list = [img_dict[key][0][0] for key in img_dict]
            # img_list = [wandb.Image(PIL.Image.fromarray(np.concatenate((img_list[i], img_list[i+1]), axis=-1)).convert('L'), caption=key)
            #             for i, key in enumerate(img_dict.keys()) if i % 2 == 0]
            # wandb.log({
            #     "example image": img_list
            # })
        img_mean_list.extend(list(np.mean(img_dict['fake_B'], axis=(1, 2, 3))))
    return np.mean(rmse_list)

def training(case, epochs, device, type, checkpoint_path=None):
    best_rmse_loss = 9999
    critic_iter = 5
    best_epoch = 0

    if type == 'semtodepth':
        a_key = 'sim_sem'
        b_key = 'sim_depth'
    elif type == 'simtotrain':
        a_key = 'sim_sem'
        b_key = 'train'

    train_dataloader, valid_dataloader = create_dataloader(a_key=a_key,
                                                           b_key=b_key,
                                                           t_ratio=0.8,
                                                           result_dic=result_dic,
                                                           case=case)

    model = cycleGAN_model(1, optim_lr=0.0002, gan_mode='wgan_gp', guided=False)
    model.to(device)

    if checkpoint_path:
        model.model_load(checkpoint_path, device)


    # model.optimizer['G'].to(device)
    # model.optimizer['D_A'].to(device)
    # model.optimizer['D_B'].to(device)

    for epoch in range(epochs):
        loss_list = [[], [], []]
        for step_i, data_tuple in enumerate(train_dataloader):
            real_a = data_tuple[0].to(device, non_blocking=True)
            real_b = data_tuple[1].to(device, non_blocking=True)

            dis_loss = model.model_train_discriminator(real_a, real_b)
            loss_list[1].append(dis_loss['dis_a'])
            loss_list[2].append(dis_loss['dis_b'])
            if step_i % critic_iter == 0:
                gen_loss, img_dic = model.model_train_generator(real_a, real_b)
                loss_list[0].append(gen_loss['gen'])

                # wandb.log({
                #     'Gen_step_loss': gen_loss,
                #     'Dis_A_step_loss': dis_loss['dis_a'],
                #     'Dis_B_step_loss': dis_loss['dis_b']
                # })
            if step_i == 0:
                break
        rmse_loss = valid(model, valid_dataloader, device)
        print(f'epoch - {epoch}, gen loss - {gen_loss}, rmse loss - {rmse_loss}')
        # wandb.log({
        #     'Gen_loss': np.mean(loss_list[0]),
        #     'Dis_A_loss': np.mean(loss_list[1]),
        #     'Dis_B_loss': np.mean(loss_list[2]),
        #     'learning_rate': model.schedular['G'].get_lr(),
        #     'rmse_loss': rmse_loss
        # })

        if best_rmse_loss > rmse_loss:
            best_rmse_loss = rmse_loss
            model.model_save(f'./type-{type}_best_model.pth')

        model.schedular_step()
    print(f'training end, best epoch - {best_epoch}, best valid rmse loss - {best_rmse_loss}')


training(1, cfg['epochs'], cfg['device'], 'semtodepth', './savemodels/case1_t(semtodepth)_best_model.pth')