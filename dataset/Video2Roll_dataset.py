import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from .balance_data import MultilabelBalancedRandomSampler

# Resize all input images to 1 x 100 x 900
# process = transforms.Compose([lambda x: x.resize((900,100)),
#                                lambda x: np.reshape(x,(100,900,1)),
#                                lambda x: np.transpose(x,[2,0,1])])

process = transforms.Compose([lambda x: x.resize((900, 100)), lambda x: np.reshape(x, (100, 900))])


class Video2RollDataset(Dataset):
    def __init__(
        self,
        img_root="../input_images",
        label_root="./labels",
        transform=None,
        subset="train",
        min_key=15,
        max_key=65,
    ):
        self.img_root = img_root  # images root dir
        self.label_root = label_root  # labels root dir
        self.transform = transform
        self.subset = subset
        # the minimum and maximum Piano Key values in the data, depending on the data stats
        self.min_key = min_key  # 3
        self.max_key = max_key  # 79
        self.load_data()

    def __getitem__(self, index):
        if self.subset == "train":
            input_file_list, label = self.data["train"][index]
        else:
            input_file_list, label = self.data["test"][index]
        input_img_list = []
        # 5 consecutive frames, set binary
        for input_file in input_file_list:
            input_img = Image.open(input_file).convert("L")
            binarr = np.array(input_img)
            input_img = Image.fromarray(binarr.astype(np.uint8))
            input_img_list.append(input_img)

        new_input_img_list = []
        for input_img in input_img_list:
            input_img = process(input_img)
            new_input_img_list.append(input_img)
        if self.transform is not None:
            new_input_img_list = list(
                self.transform(
                    image=new_input_img_list[0],
                    image1=new_input_img_list[1],
                    image2=new_input_img_list[2],
                    image3=new_input_img_list[3],
                    image4=new_input_img_list[4],
                ).values()
            )
        new_input_img_list = [img / 255.0 for img in new_input_img_list]
        # stack 5 consecutive frames
        final_input_img = np.stack(new_input_img_list)
        torch_input_img = torch.from_numpy(final_input_img).float()
        torch_label = torch.from_numpy(label).float()

        return torch_input_img, torch_label

    def __len__(self):
        if self.subset == "train":
            # return 20000
            return len(self.data["train"])
        else:
            return len(self.data["test"])

    def load_data(self):
        # self.folders: dictionary
        # key: train/test, values: list of tuples [(video_i_image_folder, video_i_label_folder)]
        self.folders = {}

        """
        old
        """
        # train_img_folder = glob.glob(self.img_root+'/training/*')
        # train_img_folder.sort(key=lambda x:int(x.split('/')[-1].split(' ')[4].split('.')[1]))
        # test_img_folder = glob.glob(self.img_root+'/testing/*')
        # test_img_folder.sort(key=lambda x:int(x.split('/')[-1].split(' ')[4].split('.')[1]))
        # train_label_folder = glob.glob(self.label_root+'/training/*')
        # train_label_folder.sort(key=lambda x: int(x.split('/')[-1].split(' ')[4].split('.')[1]))
        # test_label_folder = glob.glob(self.label_root+'/testing/*')
        # test_label_folder.sort(key=lambda x: int(x.split('/')[-1].split(' ')[4].split('.')[1]))

        """
        new
        """
        train_img_folder = glob.glob(self.img_root + "/training/*")
        train_img_folder.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
        test_img_folder = glob.glob(self.img_root + "/testing/*")
        test_img_folder.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
        train_label_folder = glob.glob(self.label_root + "/training/*")
        train_label_folder.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))
        test_label_folder = glob.glob(self.label_root + "/testing/*")
        test_label_folder.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))

        self.folders["train"] = [
            (train_img_folder[i], train_label_folder[i]) for i in range(len(train_img_folder))
        ]
        self.folders["test"] = [
            (test_img_folder[i], test_label_folder[i]) for i in range(len(test_img_folder))
        ]

        # self.data: dictionary
        # key: train/test, value: list of tuples [([frame_{i-2, i+2}_image_filename], frame_i_label)]
        self.data = {}
        self.data["train"] = []
        self.data["test"] = []
        self.train_labels = []
        count_zero = 0
        # load train data
        for img_folder, label_file in self.folders["train"]:
            # each folder contains all image frames of one video, format: frame{number}.jpg

            """old"""
            # img_files = glob.glob(img_folder + '/*.jpg')
            # img_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))

            """new"""
            img_files = glob.glob(img_folder + "/*.png")
            img_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

            # label is a pkl file. The key is frame number, value is the label vector of 88 dim
            labels = np.load(label_file, allow_pickle=True)
            for i, file in enumerate(img_files):
                """old"""
                # key = int(file.split('/')[-1].split('.')[0][5:])

                """new"""
                key = int(file.split("/")[-1].split(".")[0])

                label = np.where(labels[key] > 0, 1, 0)
                # count the number of frames that no key is activate
                if not np.any(label):
                    count_zero += 1
                    # continue
                new_label = label[self.min_key : self.max_key + 1]
                if i >= 2 and i < len(img_files) - 2:
                    file_list = [
                        img_files[i - 2],
                        img_files[i - 1],
                        file,
                        img_files[i + 1],
                        img_files[i + 2],
                    ]
                else:
                    continue
                self.data["train"].append((file_list, new_label))
                self.train_labels.append(new_label)
        print("number of all zero label in training:", count_zero)
        self.train_labels = np.asarray(self.train_labels)
        count_zero = 0

        # load test data
        for img_folder, label_file in self.folders["test"]:
            """old"""
            # img_files = glob.glob(img_folder + '/*.jpg')
            # img_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))

            """new"""
            img_files = glob.glob(img_folder + "/*.png")
            img_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

            labels = np.load(label_file, allow_pickle=True)
            for i, file in enumerate(img_files):
                """old"""
                # key = int(file.split('/')[-1].split('.')[0][5:])

                """new"""
                key = int(file.split("/")[-1].split(".")[0])

                label = np.where(labels[key] > 0, 1, 0)
                if not np.any(label):
                    count_zero += 1
                    # continue
                new_label = label[self.min_key : self.max_key + 1]
                if i >= 2 and i < len(img_files) - 2:
                    file_list = [
                        img_files[i - 2],
                        img_files[i - 1],
                        file,
                        img_files[i + 1],
                        img_files[i + 2],
                    ]
                else:
                    continue
                self.data["test"].append((file_list, new_label))
        print("number of all zero label in testing:", count_zero)

        print("length of training data:", len(self.data["train"]))
        print("length of testing data:", len(self.data["test"]))


if __name__ == "__main__":
    dataset = Video2RollDataset(subset="train")

    # g,h = dataset.__getitem__(200)
    # print(g.shape)
    # print(torch.nonzero(h))
    train_sampler = MultilabelBalancedRandomSampler(dataset.train_labels)
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    for i, data in enumerate(train_loader):
        print(i)
        imgs, label = data
        print(label.shape)
        # fig, (ax1) = plt.subplots(1)
        # ax1.imshow(label.cpu().numpy().T, plt.cm.gray)
        # plt.show()
        # print(torch.nonzero(label, as_tuple=True))
        print(torch.unique(torch.nonzero(label)[:, 1]))
        if i == 3:
            break
