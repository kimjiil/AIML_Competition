import matplotlib.pyplot as plt
import numpy as np
import torch
import os, glob, csv
import cv2
import pickle

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

result_dic = get_img_list('D:/git_repos/samsung_sem')

print()

def ret_mean(img_path):
    if isinstance(img_path, str):
        return np.mean(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

    if isinstance(img_path, list):
        return np.mean(cv2.imread(img_path[0], cv2.IMREAD_GRAYSCALE))

def create_train_data_dic(result_dic):
    train_pickle_path = './train_data_dic.pickle'
    if os.path.exists(train_pickle_path):
        with open(train_pickle_path, 'rb') as f:
            train_data_dic = pickle.load(f)
    else:
        train_data_dic = dict()
        train_data_dic['avg_depth'] = dict()
        train_data_dic['img_mean'] = dict()
        for key in result_dic['train']:
            train_data_dic['img_mean'][key] = list(map(ret_mean, result_dic['train'][key]))
            train_data_dic['avg_depth'][key] = [i[1] for i in result_dic['train'][key]]

        with open(train_pickle_path, 'wb') as f:
            pickle.dump(train_data_dic, f)
            print("train dataset pickle save!")

    return train_data_dic

def create_sim_data_dic(result_dic):
    sim_pickle_path = './sim_data_dic.pickle'
    if os.path.exists(sim_pickle_path):
        with open(sim_pickle_path, 'rb') as f:
            sim_data_dic = pickle.load(f)
    else:
        sim_data_dic = dict()
        sim_data_dic['sem'] = dict()
        for key in result_dic['sim']['sem']:
            sim_data_dic['sem'][key] = list(map(ret_mean, result_dic['sim']['sem'][key]))

        sim_data_dic['depth'] = dict()
        for key in result_dic['sim']['depth']:
            sim_data_dic['depth'][key] = list(map(ret_mean, result_dic['sim']['depth'][key]))

        with open(sim_pickle_path, 'wb') as f:
            pickle.dump(sim_data_dic, f)
            print('simulation dataset pickle save!')

    print()
    return sim_data_dic

def train_draw_plot(result_dic):

    train_data_dic = create_train_data_dic(result_dic)

    plt.figure(figsize=(6, 6))
    plot_idx = 1
    for data_key in train_data_dic:
        ax = plt.subplot(2, 1, plot_idx)
        for i, key in enumerate(train_data_dic[data_key]):
            plt.hist(train_data_dic[data_key][key], bins=100, density=True, label=key, color=plt.cm.tab20c(4*i), alpha=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        plot_idx += 1
        ax.set_xlim(105, 145)
        ax.set_title(data_key)

    plt.tight_layout()
    plt.show()

def simulation_draw_plot(result_dic):

    sim_data_dic = create_sim_data_dic(result_dic)

    plt.figure(figsize=(6, 6))
    plot_idx = 1
    for data_key in sim_data_dic:
        ax = plt.subplot(2, 1, plot_idx)

        for i, key in enumerate(sim_data_dic[data_key]):
            plt.hist(sim_data_dic[data_key][key], bins=100, density=True, color=plt.cm.tab20c(4*i), alpha=0.5, label=key)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        ax.set_title(data_key)
        ax.set_xlim(85, 130)
        plot_idx += 1

    plt.tight_layout()
    plt.show()

def sim_train_plot_draw(result_dic):
    sim_data_dic = create_sim_data_dic(result_dic)
    train_data_dic = create_train_data_dic(result_dic)

    print()
    plt.figure(figsize=(8, 6))

    ax = plt.subplot(2, 1, 1)
    for i, key in enumerate(sim_data_dic['sem']):
        plt.hist(sim_data_dic['sem'][key], bins=100, density=True, color=plt.cm.tab20c(4*i), label=key, alpha=0.5)
    ax.set_title("Simulation SEM Image")
    ax.set_xlim(85, 125)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    ax = plt.subplot(2, 1, 2)
    for i, key in enumerate(train_data_dic['img_mean']):
        plt.hist(train_data_dic['img_mean'][key], bins=100, density=True, color=plt.cm.tab20c(4*i), label=key, alpha=0.5)
    ax.set_title("Real SEM Image")
    ax.set_xlim(85, 125)
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))

    plt.tight_layout()
    plt.show()

# train_draw_plot(result_dic)
# simulation_draw_plot(result_dic)
sim_train_plot_draw(result_dic)