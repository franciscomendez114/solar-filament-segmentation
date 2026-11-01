import matplotlib.pyplot as plt
import numpy as np


def show_crop(inputs, target=None, prediction=None):
    images = [inputs[0], inputs[1], inputs[2]]
    titles = ["Original", "CLAHE", "Box prompt"]
    if target is not None:
        images.append(target[0])
        titles.append("Target")
    if prediction is not None:
        images.append(prediction)
        titles.append("Prediction")
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for axis, image, title in zip(np.atleast_1d(axes), images, titles):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    return fig


def show_instances(image, masks, title="Predicted instances"):
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image, cmap="gray")
    if masks:
        rng = np.random.default_rng(42)
        overlay = np.zeros((*image.shape, 4), dtype=np.float32)
        for mask in masks:
            overlay[mask.astype(bool)] = [*rng.random(3), 0.45]
        axis.imshow(overlay)
    axis.set_title(title)
    axis.axis("off")
    return fig

