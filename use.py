print("Loading TK app...")
from tkinter import *
app = Tk()
app.geometry("448x448")
app.resizable(False, False)

print("Loading ML model...")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.models
mnist_model = keras.models.load_model("models/mnist")
print("Initial loading complete. Finishing...")


# Drawing logic
canvas = Canvas(app, bg='black')
canvas.pack(anchor='nw', fill='both', expand=1)


def reset_canvas():
    canvas.delete("all")


def update_mouse_coords(event):
    global pre_x, pre_y
    pre_x, pre_y = event.x, event.y


def draw_delta_motion(event):
    global pre_x, pre_y
    canvas.create_line((pre_x, pre_y, event.x, event.y),
                       fill='white',
                       width=14)
    update_mouse_coords(event)


canvas.bind("<Button-1>", update_mouse_coords)
canvas.bind("<B1-Motion>", draw_delta_motion)


# Predict logic
predict_text = StringVar()
predict_text.set("Click on Submit, and I'll guess!")
predict_label = Label(app, textvariable=predict_text)

from PIL import ImageGrab
import numpy as np


def predict_canvas_digit():
    canvas_1_x = app.winfo_rootx() + canvas.winfo_x()
    canvas_1_y = app.winfo_rooty() + canvas.winfo_y()
    canvas_2_x = canvas_1_x + canvas.winfo_width()
    canvas_2_y = canvas_1_y + canvas.winfo_height()

    canvas_input = ImageGrab.grab(
        (canvas_1_x, canvas_1_y, canvas_2_x, canvas_2_y)
    ).convert("L").resize((30, 30)).crop((1, 1, 29, 29))

    canvas_input = np.asarray(canvas_input.getdata(0)).reshape((28, 28, 1)).astype("float32") / 255
    canvas_input = canvas_input[np.newaxis, :, :, :]

    prediction = mnist_model.predict(canvas_input)[0].argmax()
    predict_text.set("My prediction: " + str(prediction))


predict_button = Button(app, text="Predict!", command=predict_canvas_digit)
predict_button.pack(side="bottom")

predict_label.pack(side="bottom")

# Reset logic
reset_button = Button(app, text="Reset canvas", command=reset_canvas)
reset_button.pack(side="bottom")

app.mainloop()
