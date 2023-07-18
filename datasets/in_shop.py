from datasets.basic_dataset_scaffold import BaseDataset
import os, numpy as np, pandas as pd


def Give(opt, data_path):
    """
        This function generates a training, testing and evaluation dataloader for Metric Learning on the In-Shop Clothes dataset.
        For Metric Learning, training and test sets are provided by one text file, list_eval_partition.txt.
        So no random shuffling of classes.
        Args:
            opt: argparse.Namespace, contains all traininig-specific parameters.
        Returns:
            dict of PyTorch datasets for training, testing (by query and gallery separation) and evaluation.
        """
    # Load train-test-partition text file.
    # print(data_path + '/Eval/list_eval_partition.txt')
    data_info = np.array(
        pd.read_csv(data_path + '/Eval/list_eval_partition.txt', header=1, delim_whitespace=4))

    train, query, gallery = data_info[data_info[:, 2] == 'train'][:, :2], \
                            data_info[data_info[:, 2] == 'query'][:, :2], \
                            data_info[data_info[:, 2] == 'gallery'][:, :2]

    # Generate conversions
    lab_conv = {x: i for i, x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:, 1]])))}
    train[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:, 1]])

    lab_conv = {x: i for i, x in enumerate(
        np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:, 1], gallery[:, 1]])])))}
    query[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:, 1]])
    gallery[:, 1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:, 1]])

    # Generate Image-Dicts for training, query and gallery,
    # of shape {class_idx:[list of paths to images belong to this class] ...}
    train_image_dict = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(data_path + '/' + img_path)

    query_image_dict = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(data_path + '/' + img_path)

    gallery_image_dict = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(data_path + '/' + img_path)

    ### Uncomment this if super-labels should be used to generate resp.datasets
    # super_train_image_dict, counter, super_assign = {},0,{}
    # for img_path, _ in train:
    #     key = '_'.join(img_path.split('/')[1:3])
    #     if key not in super_assign.keys():
    #         super_assign[key] = counter
    #         counter += 1
    #     key = super_assign[key]
    #
    #     if not key in super_train_image_dict.keys():
    #         super_train_image_dict[key] = []
    #     super_train_image_dict[key].append(opt.data_path+'/'+img_path)
    # super_train_dataset = BaseTripletDataset(super_train_image_dict, opt, is_validation=True)

    train_dataset = BaseDataset(train_image_dict, opt)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    query_dataset = BaseDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset = BaseDataset(gallery_image_dict, opt, is_validation=True)


    query_poison_dataset = None
    gallery_poison_dataset = None
    test_clean_defense_dataset = None
    test_poison_defense_dataset = None

    if opt.backdoor: #for inference

        query_poison_dataset = BaseDataset(query_image_dict,  opt, is_test_poison=True) #

        gallery_poison_dataset = BaseDataset(gallery_image_dict,  opt, is_test_poison=True) #

        if opt.defense:
            test_clean_defense_dataset = BaseDataset(query_image_dict, opt,is_defenses=True)
            test_poison_defense_dataset = BaseDataset(query_image_dict, opt, is_test_poison=True, is_defenses=True)  #


    #using testing_query for test dataset
    # return {'training': train_dataset, 'testing_query': query_dataset, 'evaluation': eval_dataset,
    #         'testing_gallery': gallery_dataset,'testing_poison_query': query_poison_dataset,'testing_poison_gallery': gallery_poison_dataset}
    return {'training': train_dataset, 'testing': query_dataset,'testing_clean': query_dataset, 'evaluation': eval_dataset,
            'testing_poison': query_poison_dataset,'testing_clean_defense': test_clean_defense_dataset, 'testing_poison_defense': test_poison_defense_dataset}

    # return {'training':train_dataset, 'testing_query':query_dataset, 'evaluation':eval_dataset,
    # 'testing_gallery':gallery_dataset, 'super_evaluation':super_train_dataset}