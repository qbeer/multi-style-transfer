import cv2
import torch
import torchvision.transforms as T
from models import FastStylizationNetwork
from PIL import Image

net = FastStylizationNetwork(n_styles=8)
net.load_state_dict(torch.load('weights/model'))
net = net.to('cuda')
net.eval()

style = 0

cap = cv2.VideoCapture(0)

ret, _ = cap.read()

while ret:
    ret, frame = cap.read()

    # Trained network
    frame = T.Resize(size=(9 * 64, 16 * 64))(Image.fromarray(frame))
    frame = T.ToTensor()(frame)
    frame = torch.unsqueeze(frame, 0)  # expand dims basically, batch of 1

    torch_style = torch.tensor([style], dtype=torch.int64)

    frame = frame.to('cuda')
    torch_style = torch_style.to('cuda')

    styled_frame = net(frame, torch_style)

    cv2.imshow('styled_frame',
               styled_frame[0].cpu().detach().permute(1, 2, 0).numpy())
    del frame, styled_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) == ord('c'):
        print('style change')
        style += 1
        style %= 8

cap.release()
cv2.destroyAllWindows()