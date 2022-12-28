import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import itertools
import cv2, PIL
import os, glob
import csv, platform
import torchvision

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

class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        mobv3s = torchvision.models.mobilenet_v3_small(pretrained=True)
        feature = [nn.Sequential(nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False),
                   nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                   nn.Hardswish())]
        feature.extend([mobv3s.features._modules[module_key] for i, module_key in enumerate(mobv3s.features._modules.keys()) if i > 0])

        self.feature = nn.Sequential(*feature)
        self.avgpool = mobv3s.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=128, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=4, bias=True)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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


class test_dataset(Dataset):
    def __init__(self, path_list, transforms):
        super(test_dataset, self).__init__()
        self.path_list = path_list
        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.path_list[idx]
        img = None
        depth = 0
        img_name = None

        if isinstance(item, str):
            img = PIL.Image.open(item).convert("L")
            img_name = item.split('/')[-1].split('\\')[-1]
        elif isinstance(item, list):
            img = PIL.Image.open(item[0]).convert("L")
            img_name = item[0].split('/')[-1].split('\\')[-1]
            depth = item[1]

        if self.transforms:
            img = self.transforms(img)

        img = np.array(img).astype(np.float32) / 255.
        img = img.reshape(1, *img.shape)

        return img, depth, img_name

    def __len__(self):
        return len(self.path_list)

train_list = result_dic['train']['Depth_110'] + result_dic['train']['Depth_120'] + result_dic['train']['Depth_130'] + result_dic['train']['Depth_140']
train_db = test_dataset(train_list, None)
test_db = test_dataset(result_dic['test'], None)
submission_db = test_dataset(glob.glob('D:/git_repos/samsung_sem/sample_submission/*.png'), None)

train_db_dataloader = DataLoader(train_db, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
test_db_dataloader = DataLoader(test_db, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
submission_db_dataloader = DataLoader(submission_db, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

class ganmodels(nn.Module):
    def __init__(self, device):
        super(ganmodels, self).__init__()
        keys = ["semtodepth", "simtotrain"]
        self.models = nn.ModuleList([nn.ModuleDict({"semtodepth": cycleGAN_model(1), "simtotrain": cycleGAN_model(1)}) for i in range(4)])

        for i in range(4):
            for key in self.models[i]:
                self.models[i][key].model_load(f'./savedmodels/case{i+1}_t({key})_best_model.pth', device)


    def forward(self, img, cls_idx):
        # Train SEN Image To Simulation SEM Image
        img = self.models[cls_idx]['simtotrain'].Gen['B'](img)
        # Simulation SEM Image to Simulation Depth Image
        img = self.models[cls_idx]['semtodepth'].Gen['A'](img)

        return img

def submission_test(datalodaer, device):
    cls_model = torch.load("./savedmodels/best_cnn_classifer.pth", map_location=device)
    cls_model.eval()

    ganmodel = ganmodels(device)
    ganmodel.to(device)
    ganmodel.eval()

    for i, item in enumerate(datalodaer):
        sem_imgs = item[0].to(device)
        depths = item[1].to(device)
        img_names = item[2]

        pred_cls = torch.argmax(cls_model(sem_imgs), dim=1)
        for img, cls_idx, name in zip(sem_imgs, pred_cls, img_names):
            pred_depth = 140 + cls_idx.item() * 10
            depth_img = ganmodel(img.reshape(1, *img.shape), cls_idx)

            depth_img_uint8 = (depth_img * 255).type(torch.uint8).detach().cpu().numpy()
            mask = (depth_img_uint8 >= (pred_depth-1)).astype(np.float)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            d_mask = cv2.morphologyEx(mask[0][0], cv2.MORPH_CLOSE, kernel).reshape(*mask.shape)
            depth_img_uint8[d_mask.astype(np.bool_)] = pred_depth

            os.makedirs('./submission_pred', exist_ok=True)
            cv2.imwrite('./submission_pred/' + name, depth_img_uint8[0][0])

            print()

submission_test(submission_db_dataloader, cfg['device'])