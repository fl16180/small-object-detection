import argparse
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from constants import *
from utils import *


def detect(original_image, model, min_score, max_overlap, top_k, suppress=None):
    """ Visualize model results on an original VOC image.

        Inputs:
            original_image: image, a PIL Image
            min_score: minimum threshold for a detected box to be considered a match for a certain class
            max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
            top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
            suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    """
    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                     std=IMAGENET_STD)
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(DEVICE)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [VOC_DECODING[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['__background__']:
        # Just return original image
        return original_image

    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                       '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(VOC_ENCODING.keys())}

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


def main(args):

    image = args.image
    image = './data/VOCdevkit/VOC2007/JPEGImages/000017.jpg'
    raw_image = Image.open(image, mode='r')
    raw_image = raw_image.convert('RGB')

    # Load model checkpoint
    checkpoint = args.model
    checkpoint = './checkpoint-epoch98.pth'
    checkpoint = torch.load(checkpoint, map_location='cpu')

    model = checkpoint['model']
    model = model.to(DEVICE)
    model.eval()

    detect(raw_image, model, min_score=0.2, max_overlap=0.5, top_k=200).show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw bbox')
    parser.add_argument('-m', '--model', default=None, type=str,
                        help='path to checkpoint to use')
    parser.add_argument('-i', '--image', default=None, type=str,
                        help='path to image to detect')
    args = parser.parse_args()
    main(args)
