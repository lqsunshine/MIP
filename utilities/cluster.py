from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import pickle
import pathlib


def get_label_path_dict(image_list):
    label_path_dict = {}
    for image_path, label in image_list:
        if label not in label_path_dict.keys():
            label_path_dict[label] = [image_path]
        else:
            label_path_dict[label].append(image_path)
    return label_path_dict

def get_class_center_embed(opt,label,model,label_path_dict,get_normal_input):

    certain_class_images = [get_normal_input(image_path).to(opt.device) for image_path in label_path_dict[label]]
    certain_class_embeds = [model(img.unsqueeze(0))[0].squeeze() for img in certain_class_images]
    certain_class_embeds = torch.tensor([item.cpu().detach().numpy() for item in certain_class_embeds])


    one_kmeans =  KMeans(n_clusters=1)
    one_kmeans.fit(certain_class_embeds)
    cluster_center = torch.from_numpy(one_kmeans.cluster_centers_).to(opt.device)

    return cluster_center

def get_label_center_dict(opt, model,labels,image_list,get_normal_input):
    pkl_path = 'Dataset/'+ opt.dataset+ '_cluster_center.pkl'
    if not pathlib.Path(pkl_path).exists():
        label_center_dict = {}
        label_path_dict = get_label_path_dict(image_list)
        for label in tqdm(labels,desc='Computing the cluster center of each training class:'):
            cluster_center_embed = get_class_center_embed(opt,label,model,label_path_dict,get_normal_input)
            label_center_dict[label] = cluster_center_embed
        with open(pkl_path, 'wb') as f:
            pickle.dump(label_center_dict, f)
    else:
        fr = open(pkl_path, 'rb')  #
        label_center_dict = pickle.load(fr)  #
        fr.close()

    return label_center_dict