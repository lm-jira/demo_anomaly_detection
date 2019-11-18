import argparse
from os.path import isfile

import cv2
import importlib
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import zmq

from common import send_array, recv_array


def cut_image(input_image, size=128):
    output_images = []
    width, height, _ = input_image.shape
    for i in range(0, width, size):
        for j in range(0, height, size):
            output_images.append(input_image[i:i+size, j:j+size])

    return output_images


def patch_image(image_batch, ncol, nrow):
    size = image_batch[0].shape[0]
    patched_img = np.zeros([ncol*size, nrow*size, 3])
    index = 0
    for i in range(ncol):
        for j in range(nrow):
            patched_img[i*size:(i+1)*size, j*size:(j+1)*size] = \
                image_batch[index]
            index += 1

    return patched_img


def path(d):
    try:
        assert isfile(d)
        assert d.endswith(".pb")
        return d
    except Exception:
        raise argparse.ArgumentTypeError(
            "Example {} cannot be located.".format(d))


def main(args):
    # initialize network module, dataset module
    data_module = importlib.import_module("data.steel")

    # Socket bind
    print("setup socket...")
    zmq_context = zmq.Context()
    zmq_context.setsockopt(zmq.LINGER, 0)
    zmq_socket = zmq_context.socket(zmq.PAIR)
    zmq_socket.bind("tcp://0.0.0.0:%s" % ("6797"))

    # --- Define tensor string names and others ---
    list_scores_str = {"l1": "Testing/Scores/score_l1:0",
                       "l2": "Testing/Scores/score_l2:0"}
    recon_img_str = "generator_model/generator/tanh:0"

    idx = 0
    time_pred_avg = 0

    crop_size = data_module.get_shape_input()[1]

    with tf.Session() as sess:
        # --- get model ---
        with gfile.FastGFile(args.model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())    
        tf.import_graph_def(graph_def, name="", input_map=None,
                            return_elements=None)

        print("waiting for image...")

        while True:
            # read frame
            idx += 1
            score_type = zmq_socket.recv_string()

            fetches_str = [list_scores_str[score_type], recon_img_str]
            input_array = recv_array(zmq_socket)

            # preprocess image
            input_array = Image.fromarray(input_array.astype('uint8'))
            input_array = data_module.process_after_load(input_array)
            input_array = data_module.preprocess_image(input_array)

            # increase dimenstion if it is a gray scale image without 1 channel
            if len(input_array.shape) == 2:
                input_array = np.expand_dims(input_array, axis=2)
            time_pred_start = time.time()

            height, width, _ = input_array.shape
            ncol = width // crop_size
            nrow = height // crop_size

            # cut input image into several 128x128 images
            data_batch = cut_image(input_array)
            data_batch = np.array(data_batch)

            feed_dict = {"input_x:0": data_batch,
                         "is_training_pl:0": False}

            # reconstruct image
            scores, reconst_imgs = sess.run(fetches_str, feed_dict=feed_dict)

            diff_imgs = []
            for input_img, reconst_img in zip(data_batch, reconst_imgs):
                diff_img = (input_img - reconst_img + 2) / 4 * 255
                diff_img = diff_img.astype(np.uint8)
                diff_imgs.append(cv2.applyColorMap(diff_img, cv2.COLORMAP_JET))

            diff_imgs = np.array(diff_imgs)

            # merge all reconstucted images into one image and post process
            return_img = patch_image(diff_imgs, nrow, ncol)

            mean_score = np.mean(scores)
            zmq_socket.send_string(str(mean_score))

            return_img = np.ascontiguousarray(return_img, dtype=np.uint8)
            time_pred_avg = (time_pred_avg*idx +
                             time.time() - time_pred_start) / (idx+1)

            print("average prediction time: {:.0f} msec".
                  format(time_pred_avg * 1000))

            send_array(zmq_socket, return_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference server.')
    parser.add_argument('--model', nargs="?", type=path,
                        help='path of pb file')
    main(parser.parse_args())
