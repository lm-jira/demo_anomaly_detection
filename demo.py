import argparse
import importlib

import cv2
import datetime
import numpy as np
import os
import PIL
from PIL import ImageTk
import tkinter as tk
from tkinter import messagebox
import zmq

from common import send_array, recv_array


IMG_WIDTH, IMG_HEIGHT = 640, 320
TEXT_COLOR = (0, 100, 255)


class Window(tk.Frame):
    def __init__(self, master, server_ip):
        tk.Frame.__init__(self, master)
        self.master = master

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

        # socket manager
        self.zmq_context = zmq.Context()
        self.zmq_context.setsockopt(zmq.LINGER, 0)
        self.zmq_socket = self.zmq_context.socket(zmq.PAIR)
        self.zmq_socket.connect("tcp://%s:%s" % (str(server_ip), "6797"))

        # for center cropping image
        data_module = importlib.import_module("data.steel")
        self.crop_size = data_module.get_shape_input()[1]

        # display score (text)
        self.score_text = tk.StringVar()
        self.score_text.set("")
        # score type (radio button)
        self.score_type = tk.StringVar()
        self.score_type.set("l1")

        self.init_window()
        self.show_image()

    def init_window(self):
        self.master.title("Anomaly Detection Demo")

        camera_label = tk.Label(self.master,
                                text="Input Picture",
                                font="TkDefaultFont 20 bold")
        camera_label.grid(row=0, column=0, padx=10, pady=10)

        # label showing image from camera
        self.camera_image = tk.Label(self.master,
                                     width=IMG_WIDTH,
                                     height=IMG_HEIGHT)
        self.camera_image.grid(row=1, column=0, padx=10, pady=10)

        predict_label = tk.Label(self.master,
                                 text="Prediction result",
                                 font="TkDefaultFont 20 bold")
        predict_label.grid(row=0, column=1, padx=10, pady=10)

        # label showing image from prediction
        self.predict_image = tk.Label(self.master,
                                      width=IMG_WIDTH,
                                      height=IMG_HEIGHT)
        self.predict_image.grid(row=1, column=1, padx=10, pady=10)

        score_frame = tk.Frame(self.master)
        score_frame.grid(row=2, column=1, padx=10, pady=5)

        # score type radio button
        radio_frame = tk.LabelFrame(score_frame, text="Score type")
        radio_frame.grid(row=0, column=0, padx=12, ipady=5)
        tk.Radiobutton(radio_frame, text="l1 score", variable=self.score_type,
                       value="l1", width=10).pack(padx=15)
        tk.Radiobutton(radio_frame, text="l2 score", variable=self.score_type,
                       value="l2", width=10).pack(padx=15)

        score_label1 = tk.Label(score_frame, text="Score: ",
                                font="TkDefaultFont 18 bold")
        score_label1.grid(row=0, column=1)

        score_label2 = tk.Label(score_frame, textvariable=self.score_text,
                                font="TkDefaultFont 18", anchor="e", width=6)
        score_label2.grid(row=0, column=2)

        button_frame = tk.Frame(score_frame)
        button_frame.grid(row=0, column=3, pady=10, padx=10)
        self.capture_button = tk.Button(button_frame, text="Capture",
                                        font="TkDefaultFont 18", width=12,
                                        height=2, command=self.capture)
        self.capture_button.pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Exit", width=12, height=2,
                  command=self.exit, font="TkDefaultFont 18").pack(
            side=tk.RIGHT)

    def show_image(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        frame = self.center_crop(frame)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(cv2image, (IMG_WIDTH, IMG_HEIGHT))
        img = PIL.Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_image.imgtk = imgtk
        self.camera_image.configure(image=imgtk)

        self.zmq_socket.send_string(self.score_type.get())
        self.input_image = frame

        img_tosend = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_tosend = np.ascontiguousarray(img_tosend)
        send_array(self.zmq_socket, img_tosend)

        self.score = self.zmq_socket.recv_string()
        self.score_text.set("{:5.2f}".format(float(self.score)))

        preds = recv_array(self.zmq_socket)

        self.output_image = preds
        preds = cv2.resize(preds, (IMG_WIDTH, IMG_HEIGHT))

        pred_img = PIL.Image.fromarray(preds)
        pred_imgtk = ImageTk.PhotoImage(image=pred_img)
        self.predict_image.imgtk = pred_imgtk
        self.predict_image.configure(image=pred_imgtk)

        self.camera_image.after(10, self.show_image)

    def center_crop(self, image):
        height, width, _ = image.shape

        ncol = width // self.crop_size
        nrow = height // self.crop_size
        new_width = ncol * self.crop_size
        new_height = nrow * self.crop_size

        left = (width - new_width)//2
        top = (height - new_height)//2
        crop_image = image[top:top+new_height, left:left+new_width]

        return crop_image

    def capture(self):
        labeled_image = cv2.putText(self.output_image,
                                    "Score: {}".format(self.score),
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    TEXT_COLOR,
                                    2)
        concat_image = np.concatenate([self.input_image, labeled_image],
                                      axis=1)
        image_name = "result_{}.png".format(datetime.datetime.now().
                                            strftime("%d%m%Y_%H%M%S"))

        if not os.path.exists("capture_image"):
            os.makedirs("capture_image")

        cv2.imwrite(os.path.join("capture_image", image_name), concat_image)
        messagebox.showinfo("Message",
                            "The picture is saved at capture_image/{}"
                            .format(image_name))

    def exit(self):
        exit()


def onKeyPress(event):
    event.widget.config(relief=tk.SUNKEN)
    event.widget.after(200, lambda: event.widget.config(relief=tk.RAISED))


def main(args):
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    root.bind('<Button-1>', onKeyPress)
    Window(root, args.server_ip)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='lmsvr25',
                        help='inference server ip')
    main(parser.parse_args())
