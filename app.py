import cv2, json, uuid, os
import numpy as np
from flask import Flask, request
from SimpleHRNet import SimpleHRNet

app = Flask(__name__)
app.config.from_object('config')
os.system('./download_weights.sh')
model = SimpleHRNet(48, 17, 'hrnet_weights.pth')


def predict(image):
    joints = model.predict(image)
    # keep bodies with head confidence > 4e-1
    joints = joints[joints[:, 0, 2] > 4e-1]
    heads = joints[:, 0, :2].astype(int).tolist()
    # keep hands with confidence > 1e-1 and position above head
    lefts = joints[np.all([
        joints[:, 9, 2] > 1e-1,
        joints[:, 9, 0] < joints[:, 0, 0]
    ], axis=0)][:, 9, :2]
    rights = joints[np.all([
        joints[:, 10, 2] > 1e-1,
        joints[:, 10, 0] < joints[:, 0, 0],
        joints[:, 9, 0] >= joints[:, 0, 0]
    ], axis=0)][:, 10, :2]
    hands = np.concatenate([lefts, rights]).astype(int).tolist()
    return heads, hands


@app.route('/', methods=['POST'])
def webhook():
    array = np.fromfile(request.files['image'], np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    request.files['image'].save('/app/storage/{}.jpg'.format(uuid.uuid4().hex))
    return json.dumps(predict(image))


if __name__ == '__main__':
    image = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
    print(predict(image))
