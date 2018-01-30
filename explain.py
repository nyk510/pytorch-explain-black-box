# coding: utf-8

import ntpath
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models

from utils import get_logger

logger = get_logger()

use_cuda = torch.cuda.is_available()
if use_cuda:
    logger.info("run with gpu")

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def tv_norm(input_image, tv_beta):
    img = input_image[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    """
    画像の前処理を実行する.
    255 で割った後平均値を引き算し, 標準偏差で割算するため, 結局スケールが [-1, 1] に収まる.

    :param img:
    :return:
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img[:, :, ::-1]
    preprocessed_img /= 255.
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def generate_masked_images(mask, original, blurred):
    # 画像を演算するため正規化して float に変換する
    original = np.float32(original)
    original /= 255.

    mask = np.transpose(mask, (1, 2, 0))
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask

    heat_map = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map)
    heat_map /= 255.

    blurred = np.float32(blurred)
    blurred /= 255.

    cam = 1.0 * heat_map + original
    cam = cam / np.max(cam)

    perturbated = np.multiply(1 - mask, original) + np.multiply(mask, blurred)
    return perturbated, heat_map, mask, cam


def save(img, file_name, save_dir="output"):
    """
    mask 画像と元画像をかけ合わせた画像を生成し保存する.
    :param save_dir:
    :param prefix:
    :return:
    """
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    filename = os.path.join(save_dir, file_name)
    cv2.imwrite(filename, np.uint8(255 * img))


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model():
    model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    # 学習が行われないように gradient 計算を行わない用にする.
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model


def run(img_path,
        tv_beta=3,
        lr=0.1,
        max_iterations=500,
        l1_coefficient=0.01,
        tv_coefficient=0.2):
    """
    run main training script

    :param str img_path:
    :param int tv_beta:
    :param float lr: learning rate
    :param int max_iterations:
    :param float l1_coefficient:
    :param float tv_coefficient:
    :return:
    """

    model = load_model()
    original_img = cv2.imread(img_path, 1)
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img)
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask_init = np.ones((28, 28), dtype=np.float32)

    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.Upsample(size=(224, 224)).cuda()
    else:
        upsample = torch.nn.Upsample(size=(224, 224))
    optimizer = torch.optim.Adam([mask], lr=lr)

    target = torch.nn.Softmax()(model(img))
    # gpu -> cpu
    target = target.cpu().data.numpy()
    category = np.argmax(target)
    prob = np.max(target) * 100
    print("Category with highest probability: {category} - {prob:.2f}%".format(**locals()))
    print("Optimizing.. ")

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)

        noise = np.zeros((224, 224, 3), dtype=np.float32)

        # add noise to zero vector.
        # in the paper, noise variance is 4. but in this code, input image is normalized (i.e. devided by 255)
        # so we set 4 / 255 ~ 0.2
        noise = noise + cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax()(model(perturbated_input))
        loss = l1_coefficient * torch.mean(torch.abs(1 - mask)) + tv_coefficient * tv_norm(mask, tv_beta) + outputs[
            0, int(category)]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    # mask を (224, 224) に up sampling しその後 cpu に dump してから shape = (1, 224, 224) に変換
    upsampled_mask = upsample(mask).cpu().data.numpy()[0]
    print(upsampled_mask)

    input_filename = os.path.splitext(ntpath.basename(img_path))[0]
    logger.info(input_filename)

    perturbated, heat_map, mask, cam = generate_masked_images(upsampled_mask, original_img, blurred_img_numpy)

    # 入力された画像のファイル名を先頭に付けて保存する
    for img, name in zip([perturbated, heat_map, mask, cam], ["perturbated", "heat_map", "mask", "cam"]):
        fname_i = input_filename + name + ".png"
        save(img, file_name=fname_i, save_dir="output")


if __name__ == '__main__':
    run("./examples/flute.jpg")