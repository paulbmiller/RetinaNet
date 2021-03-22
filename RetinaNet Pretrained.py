import numpy as np
import torchvision
from torchvision import transforms
import torch
import cv2


CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def show_output(torch_image, model_output, threshold):
    np_img = np.array(torch_image.permute(1, 2, 0) * 255, dtype=np.uint8)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    for i in range(len(output['boxes'])):
        if output['scores'][i] < threshold:
            break
        x0, y0 = int(model_output['boxes'][i][0].item()), int(model_output['boxes'][i][1].item())
        x1, y1 = int(model_output['boxes'][i][2].item()), int(model_output['boxes'][i][3].item())
        cv2.rectangle(np_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
        cv2.putText(np_img, CLASS_NAMES[model_output['labels'][i].item()], (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 0, 0), 1)

    cv2.imshow('Image', np_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # print('{} classes to predict.'.format(len(CLASS_NAMES)))

    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True,
                                                                trainable_backbone_layers=0)

    print('Model loaded successfully.')

    test_data = torchvision.datasets.ImageFolder('sample_images', transform=transforms.Compose(
        [transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    model.eval()

    with torch.no_grad():
        for image, _ in test_loader:
            output = model(image)[0]
            show_output(image[0], output, 0.5)
