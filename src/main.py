import os
import random

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.io import imread
from skimage.transform import resize, rescale, rotate
from torch.utils.data import Dataset
from torchvision.transforms import Compose

batch_size = 16
lr = 0.0001
workers = 2
image_size = 224
aug_scale = 0.05
aug_angle = 15


def crop_sample(x):
    volume, mask = x
    volume[volume < np.max(volume) * 0.1] = 0
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask


def normalize_volume(volume):
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume


class Dataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(
            self,
            images_dir,
            transform=None,
            image_size=256,
            subset="train",
            random_sampling=True,
            seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                    filter(lambda f: ".tif" in f, filenames),
                    key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=10)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        im1 = np.asarray(y_pred).astype(np.bool)
        im2 = np.asarray(y_true).astype(np.bool)

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        im_sum = im1.sum() + im2.sum()
        if im_sum == 0:
            return 1.0

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        # return loss function
        return 1, - (2. * intersection.sum() / im_sum)


def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask


class UNetNoBatch(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetNoBatch, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def data_loaders(batch_size, workers, image_size, aug_scale, aug_angle,
                 dataset="../input/lgg-mri-segmentation/kaggle_3m"):
    dataset_train, dataset_valid = datasets(dataset, image_size, aug_scale, aug_angle)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(images, image_size, aug_scale, aug_angle):
    train = Dataset(
        images_dir=images,
        subset="train",
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=images,
        subset="validation",
        image_size=image_size,
        random_sampling=False,
    )
    return train, valid


# setup CUDA to speedup the computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

loader_train, loader_valid = data_loaders(batch_size, workers, image_size, aug_scale, aug_angle)
loaders = {"train": loader_train, "valid": loader_valid}

""" Method used to obtain the results by varying the Learning Rate 
"""


def train_validate():
    for epochs in [100]:  # , 200
        for lr in [0.01, 0.0001, 0.1]:
            print("LR: {}, epochs: {}".format(lr, epochs))
            unet = UNet(in_channels=Dataset.in_channels,
                        out_channels=Dataset.out_channels)
            unet.to(device)
            loss_train = []
            loss_valid = []

            dsc_loss = DiceLoss()
            optimizer = optim.Adam(unet.parameters(), lr=lr)
            for epoch in range(epochs):
                step_train = 0
                step_validation = 0
                for phase in ["train", "valid"]:
                    if phase == "train":
                        unet.train()
                    else:
                        unet.eval()

                    cumulative_loss = 0

                    # cycle on the batches of the train and validation dataset
                    for i, data in enumerate(loaders[phase]):

                        if phase == "train":
                            step_train += 1

                        if phase == "valid":
                            step_validation += 1

                        x, y_true = data
                        x, y_true = x.to(device), y_true.to(device)

                        # reset the gradient
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):

                            # forward phase of the net
                            y_pred = unet(x)

                            # compute loss function
                            loss = dsc_loss(y_pred, y_true)
                            cumulative_loss += loss.item()

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                    if phase == "valid":
                        loss_valid.append(cumulative_loss / step_validation)

                    if phase == "train":
                        loss_train.append(cumulative_loss / step_train)

                    step_train = 0
                    step_validation = 0

            iterations = np.arange(0, len(loss_train), 1)

            print("Batch used: Loss train:{} , Loss validation:{} , Epochs:{}, Learning Rate:{}".format(
                loss_train[len(loss_train) - 1], loss_valid[len(loss_valid) - 1], epochs, lr))

            plt.plot(iterations, loss_train, 'b-')
            plt.plot(iterations, loss_valid, 'r-')

            plt.legend(["Loss training", "Loss validation"], loc="upper right")
            plt.xlabel('Epochs')
            plt.ylabel('Dice coefficient')
            plt.figure()
            plt.show()


""" Method used to obtain the results without using Batch Normalization
"""


def train_validate_no_batch():
    for epochs in [100]:
        for lr in [0.01, 0.0001, 0.1]:
            unet = UNetNoBatch(in_channels=Dataset.in_channels,
                               out_channels=Dataset.out_channels)
            unet.to(device)
            loss_train = []
            loss_valid = []
            dsc_loss = DiceLoss()
            optimizer = optim.Adam(unet.parameters(), lr=lr)
            for epoch in range(epochs):
                step_train = 0
                step_validation = 0

                for phase in ["train", "valid"]:
                    if phase == "train":
                        unet.train()
                    else:
                        unet.eval()

                    cumulative_loss = 0

                    # cycle on the batches of the train and validation dataset
                    for i, data in enumerate(loaders[phase]):

                        if phase == "train":
                            step_train += 1

                        if phase == "valid":
                            step_validation += 1

                        x, y_true = data
                        x, y_true = x.to(device), y_true.to(device)

                        # reset the gradient
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):

                            # forward phase of the net
                            y_pred = unet(x)

                            loss = dsc_loss(y_pred, y_true)
                            cumulative_loss += loss.item()

                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                    if phase == "valid":
                        loss_valid.append(cumulative_loss / step_validation)

                    if phase == "train":
                        loss_train.append(cumulative_loss / step_train)

                    step_train = 0
                    step_validation = 0

            iterations = np.arange(0, len(loss_train), 1)

            print(
                "Batch Normalization not used: Loss train:{} , Loss validation:{} , Epochs:{}, Learning Rate:{}".format(
                    loss_train[len(loss_train) - 1], loss_valid[len(loss_valid) - 1], epochs, lr))

            plt.plot(iterations, loss_train, 'b-')
            plt.plot(iterations, loss_valid, 'r-')

            plt.legend(["Loss training", "Loss validation"], loc="upper right")
            plt.xlabel('Epochs')
            plt.ylabel('Dice coefficient')
            plt.figure()
            plt.show()


""" Method used to compare the results with different Batch Size 
"""


def train_validate_batch_size():
    for epochs in [30]:
        for batch_size_in in [16, 32, 64, 128, 256, 512, 1024]:
            loader_train, loader_valid = data_loaders(batch_size_in, workers, image_size, aug_scale, aug_angle)
            loaders = {"train": loader_train, "valid": loader_valid}

            for lr in [0.0001, 0.1]:
                unet = UNet(in_channels=Dataset.in_channels,
                            out_channels=Dataset.out_channels)
                unet.to(device)
                loss_train = []
                loss_valid = []
                dsc_loss = DiceLoss()
                optimizer = optim.Adam(unet.parameters(), lr=lr)

                for epoch in range(epochs):
                    step_train = 0
                    step_validation = 0
                    for phase in ["train", "valid"]:
                        if phase == "train":
                            unet.train()
                        else:
                            unet.eval()

                        cumulative_loss = 0

                        # cycle on the batches of the train and validation dataset
                        for i, data in enumerate(loaders[phase]):

                            if phase == "train":
                                step_train += 1

                            if phase == "valid":
                                step_validation += 1

                            x, y_true = data
                            x, y_true = x.to(device), y_true.to(device)

                            # reset the gradient
                            optimizer.zero_grad()

                            with torch.set_grad_enabled(phase == "train"):

                                # forward phase of the net
                                y_pred = unet(x)

                                loss = dsc_loss(y_pred, y_true)
                                cumulative_loss += loss.item()

                                if phase == "train":
                                    loss.backward()
                                    optimizer.step()

                        if phase == "valid":
                            loss_valid.append(cumulative_loss / step_validation)

                        if phase == "train":
                            loss_train.append(cumulative_loss / step_train)

                        step_train = 0
                        step_validation = 0

                iterations = np.arange(0, len(loss_train), 1)
                print(
                    "Different Batch Size Used: Batch Size: {}, Loss train:{} , Loss validation:{} , Epochs:{}, Learning Rate:{}".format(
                        batch_size_in, loss_train[len(loss_train) - 1], loss_valid[len(loss_valid) - 1], epochs, lr))

                plt.plot(iterations, loss_train, 'b-')
                plt.plot(iterations, loss_valid, 'r-')

                plt.legend(["Loss training", "Loss validation"], loc="upper right")
                plt.xlabel('Epochs')
                plt.ylabel('Dice coefficient')
                plt.figure()
                plt.show()


# Run all the three methods used to obtain the results showed in the report
train_validate()
train_validate_no_batch()
train_validate_batch_size()
