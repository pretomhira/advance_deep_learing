# Exercise 1.1: Basic structure of a PyTorch project

The task was to get familiar with the provided PyTorch image classification code and improve it in two ways: add data augmentation and save the model with the highest validation accuracy. We used ResNet18 to classify CIFAR10 images and kept the implementation in PyTorch and torchvision, as requested in the assignment.

## What and why

1. **Set up a local dataset folder.** We replaced the placeholder download path with `data/cifar10`, relative to `ex1_main.py`. This keeps the data inside the project and works even when the script is started from a different directory.

2. **Separated training and validation data.** We split the original 50,000 training images into 45,000 for training and 5,000 for validation. The separate 10,000 test images were kept for the final evaluation. We used a fixed seed for the split so the same images are assigned to each subset when running again.

3. **Added training augmentation.** We used a random 32 × 32 crop with 4 pixels of padding and a random horizontal flip. These give the model slightly different versions of the images during training, helping it learn to recognise objects when their position or orientation changes. We also kept tensor conversion and the supplied ImageNet normalization values.

4. **Kept validation and test preprocessing consistent.** These images only use tensor conversion and normalization. We created separate dataset objects for training and validation so random augmentation would not affect validation results. Their image indices do not overlap.

5. **Adjusted ResNet18 for CIFAR10.** We started with random weights and replaced the final classification layer with a layer that outputs 10 scores, one for each class. We also moved the ViT and CrossViT imports into their own model branches so the unfinished models would not stop ResNet18 from running.

6. **Trained and validated the model.** We kept the supplied cross-entropy loss and SGD optimizer, with a learning rate of `0.003` and momentum of `0.9`. We trained for five epochs with batches of 64 images and checked validation accuracy after every epoch. Validation runs without updating the model's weights.

7. **Saved the best checkpoint.** We changed the evaluation function to return accuracy. After each epoch, the training loop compares validation accuracy with the best value so far and saves the model parameters to `checkpoints/r18_best.pt` if the score improves. Saving is automatic. This keeps the best model even if a later epoch performs worse.

8. **Tested the saved model.** After training, We reloaded the best checkpoint before evaluating the test set. This means the reported test accuracy belongs to the model selected using validation accuracy.

## Main functions

- **`parse_args()` reads the settings from the command line.** For example, `--epochs 5` tells the script to train for five epochs. If we leave an option out, it uses the default value defined in this function. The function returns these settings as `args`.

- **`train()` teaches the model for one epoch.** It goes through the training images in batches, makes predictions, and calculates the loss, which measures prediction error. It then calculates gradients with `loss.backward()` and updates the weights with `optimizer.step()`. Before each batch, `optimizer.zero_grad()` clears the previous gradients so they do not accumulate.

- **`test()` checks how well the model performs.** We use this function for both validation and testing. It switches the model to evaluation mode and makes predictions without calculating gradients or updating weights. It prints the average loss and accuracy, then returns accuracy so we can compare epochs and save the best model.

- **`run(args)` brings everything together.** It prepares the data and transforms, creates the model, and sets up the loss function and optimizer. It then calls `train()` and `test()` for each epoch, saves improved checkpoints, and reloads the best model for the final test.

When we run the file directly, the `if __name__ == '__main__':` block calls `parse_args()` first and passes the settings to `run(args)`. The basic flow is:

```text
parse_args() → run(args) → train and validate each epoch
                        → reload the best checkpoint → test
```

## Running on CIP

From the project folder, using the course environment's Python:

```bash
/proj/aimi-adl/envs/adl23_2/bin/python ex1_main.py --model r18 --epochs 5
```

## Results

The best checkpoint was saved at epoch 5.

| Metric | Validation | Test |
| --- | --- | --- |
| Accuracy | 66.08% (3304/5000) | 64.80% (6480/10000) |
| Average cross-entropy loss | 1.0007 | 1.0128 |

The validation and test scores were close in this run. These are the results after five epochs; we have not compared longer training runs or other settings yet.

Assignment: [Sheet 1, Exercise 1.1](exercise-01.pdf). Implementation: [ex1_main.py](ex1_main.py).
