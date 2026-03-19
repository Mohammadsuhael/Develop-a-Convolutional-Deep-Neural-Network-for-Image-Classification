# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.## Neural Network Model
<img width="1024" height="751" alt="image" src="https://github.com/user-attachments/assets/93bdbae8-0b65-4055-b259-4f5163e8cdfb" />

## DESIGN STEPS
## STEP1'
.Load and Preprocess Data<br>
## STEP2.
Get the shape of the first image in the training dataset<br>
## STEP3.
Get the shape of the first image in the test dataset<br>
## STEP4.
Train the Model<br>
## STEP5.
Test the Model<br>
## STEP6.
Predict on a Single Image<br>
## STEP7.
Display the image

## PROGRAM

### Name: Mohammad Suhael

### Register Number:212224230164

```python

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # conv2 should take 32 channels as input from conv1's output after pooling
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        # conv3 should take 64 channels as input from conv2's output after pooling
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # Calculate input size for fc1: 128 channels * 3x3 feature map after 3 pooling layers
        # (28 -> 14 -> 7 -> 3) based on kernel_size=2, stride=2
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: Mohammad Suhael')
print('Register Number:212224230164')
summary(model, input_size=(1, 28, 28))

# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

     print('Name: Mohammad Suhael')
     print('Register Number:212224230164')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')




```

### OUTPUT

## Training Loss per Epoch

<img width="760" height="525" alt="image" src="https://github.com/user-attachments/assets/d60875b2-3315-4de0-a7cc-8414f0c1fabb" />
<img width="365" height="91" alt="image" src="https://github.com/user-attachments/assets/52a89717-9f12-4916-9a95-ec6b57a8c3ec" />


## Confusion Matrix

<img width="852" height="789" alt="image" src="https://github.com/user-attachments/assets/a1c69560-bc99-4bd9-a62a-a9eec79c4596" />

## Classification Report
<img width="585" height="412" alt="image" src="https://github.com/user-attachments/assets/2ed3f6dc-6b6b-41af-ad54-1bf4719743f8" />

### New Sample Data Prediction
<img width="561" height="569" alt="image" src="https://github.com/user-attachments/assets/f8022966-5ecd-43ca-a6c3-01e6fbfca0a6" />

## RESULT
Thus , a convolutional deep neural network (CNN) for image classification and to verify the response for new images is successfully developed.
