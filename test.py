import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn


device="cuda" if torch.cuda.is_available() else "cpu"


model=models.resnet50()

model.fc=nn.Sequential(
    nn.Linear(2048,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,2)
)


model.load_state_dict(
    torch.load(
        "ai_detector_resnet50.pth",
        map_location=device
    )
)


model.to(device)

model.eval()


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


img=Image.open("test.JPG")


img=transform(img).unsqueeze(0)


with torch.no_grad():

    output=model(
        img.to(device)
    )


prediction=torch.argmax(
    output
).item()


if prediction==0:
    print("AI Generated")
else:
    print("Real Image")