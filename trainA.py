import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import matplotlib.pyplot as plt
import cv2
import imageio.v2
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------------------------
def get_config():
# Parse all command-line arguments and set up device
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="DL_ASS2")
    parser.add_argument("-we", "--wandb_entity", default="3628-pavitrakhare-indian-institute-of-technology-madras")
    parser.add_argument("-key", "--wandb_key", default="bad0d13cb33ad3ab10579145135ecdce4cd371f0")
    parser.add_argument("-dpTrain", "--dpTrain", default="/kaggle/input/my-dataset/inaturalist_12K/train")
    parser.add_argument("-dpTest", "--dpTest", default="/kaggle/input/my-dataset/inaturalist_12K/val")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_dense", type=int, default=256)
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--filter_org", nargs='+', type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--batch_norm", choices=['Yes', 'No'], default='Yes')
    parser.add_argument("--activation", choices=['relu', 'gelu', 'tanh', 'sigmoid'], default='gelu')
    parser.add_argument("--data_aug", choices=['Yes', 'No'], default='No')
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=args.wandb_key)

    #sweep configurations from the command line
    sweep_config_bestParam = {
    'method': 'bayes',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'
    },
    'parameters': {
    'dropout': {
            'values': [args.dropout]
        },
    'num_dense':{
            'values': [args.num_dense]
        },

        'kernel_size':{
            'values': [args.kernel_size]
        },
        'filter_org':{
            'values': [args.filter_org]
        },
        'batch_norm':{
            'values': [args.batch_norm]
        },

        'activation': {
            'values': [ args.activation]
        },

        'data_aug': {
            'values': [args.data_aug]
        },
        'epoch':{
            'values':[args.epoch]
        }
                }
        }

    # creating a sweep id for sweeps
    sweep_id = wandb.sweep(sweep_config_bestParam, project=args.wandb_project,entity=args.wandb_entity)
    
    return sweep_id,args, device



# ---------------------------------------------------------------------------------------

# Loads training and validation data with or without augmentation
def load_train_val_data(train_data_directory, use_data_augmentation):
    basic_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    augment_steps = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(20)
    ]
    # Choose transforms based on augmentation flag
    transform_pipeline = transforms.Compose(augment_steps + basic_transforms[1:]
                                            if use_data_augmentation == 'Yes'
                                            else basic_transforms)
    # Load dataset with chosen transforms
    full_dataset = ImageFolder(root=train_data_directory, transform=transform_pipeline)
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    # Create samplers for train and validation splits
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=SubsetRandomSampler(train_idx),
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=SubsetRandomSampler(val_idx),
                           num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------------------

# Loads test data with or without augmentation
def load_test_data(test_data_directory, apply_data_augmentation):
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    augmented_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(20)
    ] + base_transforms[1:]
    # Select transforms based on augmentation flag
    transform_pipeline = transforms.Compose(augmented_transforms if apply_data_augmentation == 'Yes'
                                            else base_transforms)
    # Load test dataset
    test_dataset = ImageFolder(root=test_data_directory, transform=transform_pipeline)
    return DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)



# ---------------------------------------------------------------------------------------

# Custom CNN model with 5 conv layers and flexible options
class ConvNetworkModel(nn.Module):
    
    #A Convolutional Neural Network with 5 convolutional layers, followed by
    #fully connected layers for image classification.
    
    def __init__(self, in_channels=3, num_filters=[32, 64, 128, 256, 512],
                 filter_size=[3, 3, 5, 5, 7], activation=nn.ReLU(),
                 stride=1, padding=1, pool_size=(2, 2), fc_size=512,
                 num_classes=10, dropout=0, batch_norm='Yes'):
        
        # Initializes the CNN model with configurable conv and FC layers
        # in_channels: Number of input image channels (e.g., 3 for RGB)
        # num_filters: List specifying filters for each conv layer
        # filter_size: Kernel sizes for corresponding conv layers
        # activation: Activation function (e.g., ReLU)
        # stride: Stride value for convolutions
        # padding: Padding for convolutions
        # pool_size: Size of max pooling window
        # fc_size: Number of neurons in the fully connected layer
        # num_classes: Total number of output classes
        # dropout: Dropout rate to prevent overfitting
        # batch_norm: Whether to apply batch normalization ('Yes' or 'No')

        super(ConvNetworkModel, self).__init__()

        # Store configuration parameters
        self.config = {
            'channels': in_channels,
            'num_filters': num_filters,
            'filter_size': filter_size,
            'activation': activation,
            'stride': stride,
            'padding': padding,
            'pool_size': pool_size,
            'fc_size': fc_size,
            'num_classes': num_classes,
            'dropout': dropout,
            'batch_norm': batch_norm
        }

        # Create convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels_list = [in_channels] + num_filters[:-1]

        for i in range(5):
            block = nn.Sequential()
            # Convolutional layer
            block.add_module(f'conv{i+1}',
                nn.Conv2d(in_channels_list[i], num_filters[i],
                          filter_size[i], stride=stride, padding=padding))

            # Optional batch normalization
            if batch_norm == 'Yes':
                block.add_module(f'bn{i+1}', nn.BatchNorm2d(num_filters[i]))

            # Activation function
            block.add_module(f'act{i+1}', activation)

            # Max pooling
            block.add_module(f'pool{i+1}', nn.MaxPool2d(pool_size, stride=2))

            # Dropout
            block.add_module(f'drop{i+1}', nn.Dropout2d(dropout))

            self.conv_blocks.append(block)

        # Calculate output dimensions after convolutions and pooling
        input_size = 224
        for i in range(5):
            input_size = self.findOutputSize(
                input_size, filter_size[i], padding, stride)

        # Fully connected layers
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc',
            nn.Linear(num_filters[4] * (int(input_size) ** 2), fc_size))

        if batch_norm == 'Yes':
            self.classifier.add_module('fc_bn', nn.BatchNorm1d(fc_size))

        self.classifier.add_module('fc_act', activation)
        self.classifier.add_module('fc_drop', nn.Dropout(dropout))
        self.classifier.add_module('output', nn.Linear(fc_size, num_classes))

    def findOutputSize(self, img_size, kernel_size, padding, stride):
        """Calculate the output size after convolution and pooling."""
        # After convolution
        conv_output = (img_size - kernel_size + 2*padding) / stride + 1
        # After pooling (with stride=2)
        return conv_output / 2    
    
    def forward(self, x):
        """Forward pass through the network."""
        # Pass through all convolutional blocks
        for block in self.conv_blocks:
            x = block(x)

        # Flatten for fully connected layers
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        # Pass through classifier
        x = self.classifier(x)

        return x


# ---------------------------------------------------------------------------------------



# Train the model for one epoch and return metrics
def trainDataTraining(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    samples = 0
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    correct = 0
    # Go through all batches in the training loader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        samples += targets.size(0)
    # Return model, average loss, and accuracy
    return model, epoch_loss/(batch_idx+1), 100*correct/samples



# ---------------------------------------------------------------------------------------

# Validate the model on test/validation data
def validDataTesting(model, test_data, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Loop through all batches in test loader
        for inputs, targets in test_data:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    # Return accuracy as a percentage
    return 100 * correct / total



# ---------------------------------------------------------------------------------------

# Train the model across epochs and perform early stopping
def trainCnnModelVal(model, train_data, val_data, epochs, device):
    best_acc = 0.0
    patience_counter = 0
    max_patience = 2
    best_weights = model.state_dict().copy()
    current_epoch = 0
    # Loop over epochs
    while current_epoch < epochs:
        model, train_loss, train_acc = trainDataTraining(model, train_data, device)
        val_acc = validDataTesting(model, val_data, device)
        # Log metrics to wandb
        wandb.log({
            'training_loss': train_loss,
            'training_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'completed_epochs': current_epoch+1
        })
        print(f"Epoch {current_epoch+1}/{epochs} | "
              f"Loss: {train_loss:.3f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")
        # Save best model by validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {current_epoch+1}")
                break
        current_epoch += 1

    # reload the best model
    model.load_state_dict(best_weights)
    print(f"Restored model with best validation accuracy: {best_acc:.2f}%")
    return model



# ---------------------------------------------------------------------------------------

# Visualize predictions for random test images
def plotImage(model, args, device):
    # Predict and visualize random test images
    categories = ['Amphibia','Animalia','Arachnida','Aves','Fungi',
                  'Insecta','Mammalia','Mollusca','Plantae','Reptilia']

    test_dir = args.dpTest
    img_batch, true_labels = [], []

    #iterate over the test data and store the image
    for idx, category in enumerate(categories):
        image_names = random.sample(os.listdir(os.path.join(test_dir, category)), 3)
        for img_file in image_names:
            img_path = os.path.join(test_dir, category, img_file)
            img = imageio.v2.imread(img_path)
            if img.ndim == 3:
                resized_img = cv2.resize(img, (224, 224))
                img_batch.append(resized_img)
                true_labels.append(category)

    np_imgs = np.array(img_batch).astype('float32') / 255.0
    img_tensor = torch.tensor(np.transpose(np_imgs, (0, 3, 1, 2))).to(device)

    # predict the image on the model
    predictions = model(img_tensor)
    predicted_indices = torch.argmax(predictions, dim=1)
    label_map = dict(enumerate(categories))


    # creating a plot of these images
    fig, axes = plt.subplots(10, 3, figsize=(12, 20))
    for row in range(10):
        for col in range(3):
            idx = row * 3 + col
            if idx >= len(img_batch):
                continue

            axes[row, col].imshow(img_batch[idx])
            predicted_label = label_map[predicted_indices[idx].item()]
            is_correct = predicted_label == true_labels[idx]
            title_color = 'green' if is_correct else 'red'
            axes[row, col].set_title(f'True: {true_labels[idx]}, Predicted: {predicted_label}', color=title_color)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('pictures.png', bbox_inches='tight')
    image = Image.open('pictures.png')

    # Convert figure to PIL image for W&B
    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    wandb.log({"images": [wandb.Image(pil_image, caption="Test image prediction")]})


# ---------------------------------------------------------------------------------------

# Main function to run everything
def main():
    # Parse arguments and set up device
    sweep_id,args, device = get_config()
    # Choose activation function
    activation_map = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }
    activation = activation_map[args.activation]


                  # ---------------------------------------------------------------------------------------   
    # Run the sweep agent for 1 trial

    def sweep_train():
        with wandb.init() as run:
            # Build model with sweep parameters
            model = ConvNetworkModel(
                in_channels=3,
                num_filters=wandb.config.filter_org,
                filter_size=wandb.config.kernel_size,
                activation=activation_map[wandb.config.activation],
                fc_size=wandb.config.num_dense,
                num_classes=10,
                dropout=wandb.config.dropout,
                batch_norm=wandb.config.batch_norm
            ).to(device)
            
            # Training logic remains same
            train_loader, val_loader = load_train_val_data(args.dpTrain, wandb.config.data_aug)
            finalModel = trainCnnModelVal(model, train_loader, val_loader, wandb.config.epoch, device)
            test_loader = load_test_data(args.dpTest, apply_data_augmentation=args.data_aug)
            test_acc = validDataTesting(finalModel, test_loader, device)
            # Visualize predictions
            plotImage(finalModel, args, device)
            print(f"Test Accuracy: {test_acc:.2f}%")
            wandb.log({'test_accuracy': test_acc})

                  # ---------------------------------------------------------------------------------------
    
    wandb.agent(sweep_id, function=sweep_train, count=1)



# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
