import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter.font as font
import skimage.io as ski_io
from PIL import Image, ImageFilter

class Paint(object):
	DEFAULT_PEN_SIZE = 5.0
	DEFAULT_COLOR = 'black'	
	def __init__(self):
		self.root = Tk()
		self.root.title("RON Digit Recognizer")
		self.choose_size_button = 35
		self.DEFAULT_COLOR = 'black'
		self.h = Text(self.root, height=1, width=45, font=("Helvetica", 16))
		self.h.grid(row = 0, columnspan = 3)
		self.h.insert(END, "Please draw digit to full scale to area for better accuracy")
		self.h.config(state=DISABLED)
		self.c = Canvas(self.root, bg='white', width=400, height=400)
		self.c.grid(row=1, columnspan = 2)
		self.t = Text(self.root, height=1, width=8, font=("Helvetica", 60))
		self.t.grid(row = 1, column = 2)
		self.t.insert(END, "")
		self.reset_button = Button(self.root, text='Reset', command=self.clear)
		self.reset_button.grid(row=2, column=0)
		self.scan_button = Button(self.root, text='Scan', command=self.scan)
		self.scan_button.grid(row=2, column=1)
		self.setup()
		self.root.mainloop()
	def setup(self):
		self.old_x = None
		self.old_y = None
		self.line_width = self.choose_size_button
		self.color = self.DEFAULT_COLOR
		self.eraser_on = False
		self.c.bind('<B1-Motion>', self.paint)
		self.c.bind('<ButtonRelease-1>', self.reset)
	def paint(self, event):
		self.line_width = self.choose_size_button
		paint_color = self.color
		if self.old_x and self.old_y:
			self.c.create_line(self.old_x, self.old_y, event.x, event.y,
				width=self.line_width, fill=paint_color,
				capstyle=ROUND, smooth=TRUE, splinesteps=36)
		self.old_x = event.x
		self.old_y = event.y
	def reset(self, event):
		self.old_x, self.old_y = None, None
	def clear(self) :
		self.c.delete('all')
		self.t.delete(1.0,"end")
		self.t.insert(1.0, "")
	def scan(self) :
		self.c.postscript(file="tmp_canvas.eps", colormode="color", width=400, height=400, pagewidth=399, pageheight=399)
		data = ski_io.imread("tmp_canvas.eps")
		ski_io.imsave("canvas_image.png", data)
		argv = "canvas_image.png"
		im = Image.open(argv).convert('L')
		width = float(im.size[0])
		height = float(im.size[1])
		newImage = Image.new('L', (28, 28), (255))
		if width > height :
			nheight = int(round((20.0 / width * height), 0))
			if (nheight == 0):
				nheight = 1
			img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wtop = int(round(((28 - nheight) / 2), 0))
			newImage.paste(img, (4, wtop))
		else:
			nwidth = int(round((20.0 / height * width), 0))
			if (nwidth == 0):
				nwidth = 1
			img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
			wleft = int(round(((28 - nwidth) / 2), 0))
			newImage.paste(img, (wleft, 4))
		tv = list(newImage.getdata())
		tva = [(255 - x) * 1.0 / 255.0 for x in tv]
		img = np.array(tva).reshape(28, 28)

		# plt.imshow(np.array(tva).reshape(28, 28))
		# plt.colorbar()
		# plt.show()
		model = tf.keras.models.load_model("./mnist_model_file_final/mnist_model")
		probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
		predictions = probability_model.predict(img.reshape(1, 28, 28))
		maxp = np.amax(predictions)
		if maxp > 0.8 :
			self.n = np.argmax(predictions)
			self.t.delete(1.0,"end")
			self.t.insert(1.0, self.n)
		else :
			self.t.delete(1.0,"end")
			self.t.insert(1.0, "Not Sure")

if __name__ == '__main__':
    Paint()

    # print(predictions)
    # print(np.argmax(predictions))
    # plt.imshow(x_test[1])
    # plt.show()


