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

8. **Tested the saved model.** After training, we reloaded the best checkpoint before evaluating the test set. This means the reported test accuracy belongs to the model selected using validation accuracy.

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

# Exercise 1.2: VisionTransformer (ViT): Self-attention for image classification

The task was to replace ResNet18 with a Vision Transformer and implement the missing attention mechanism in the provided skeleton. We kept the training pipeline from Exercise 1.1 so we could train and evaluate ViT on the same CIFAR10 split.

## What and why

1. **Used the supplied patch embedding.** The ViT splits each 32 × 32 image into sixteen 8 × 8 patches. Each patch contains 192 pixel values, which a linear layer converts into 64 features. This turns an image into a sequence of tokens that the transformer can process.

2. **Kept the class token and positional embeddings.** The skeleton adds a learned class token before the patch tokens, giving 17 tokens per image. The class token gathers information through attention and is used for classification. Positional embeddings give the model information about where the patches belong in the image.

3. **Implemented query, key, and value projections.** In the `Attention` class, we added learned linear layers to produce queries, keys, and values. A query describes what a token is looking for, a key helps determine how relevant another token is, and a value contains the information that token contributes. In self-attention, all three come from the same input sequence.

4. **Split attention into eight heads.** Each head can learn different relationships between tokens. The configuration uses 64 features per head, so the combined internal dimension is 512. After attention, we combine the heads and project back to the original 64 features per token.

5. **Calculated attention weights.** We compare queries with keys using dot products and divide the scores by the square root of the head dimension. This keeps the scores from becoming too large. Softmax converts the scores into weights over the key tokens. We apply dropout to these weights during training, then use them to combine the values.

6. **Connected ViT to the training script.** We changed the import in `ex1_main.py` to load `ViT` from `my_models_skeleton.py`. Selecting `--model vit` now uses this model instead of ResNet18. The supplied ViT has two transformer blocks and a final layer that produces ten class scores.

7. **Trained with the existing pipeline.** We used the same augmentation, data split, batch size of 64, cross-entropy loss, and SGD settings as in Exercise 1.1. We trained for five epochs, saved improvements in validation accuracy to `checkpoints/vit_best.pt`, and reloaded the best checkpoint for the final test.

## Main functions

- **`Attention.__init__()` sets up the attention layers.** It creates the query, key, and value projections, the scaling factor, softmax, dropout, and the output projection. These layers contain the parameters learned during training.

- **`Attention.forward()` lets tokens exchange information.** It creates queries, keys, and values, splits them into heads, calculates attention weights, and combines the values. Its output has the same shape as its input for self-attention: `[batch, tokens, features]`. The optional `context` argument is also available for cross-attention in the next exercise.

- **`Transformer.forward()` updates the tokens through each block.** Each block applies attention and then a feedforward network, with layer normalization before each operation. Residual connections add the previous token features back to each result, helping information and gradients pass through the network.

- **`ViT.forward()` turns images into predictions.** It embeds the patches, adds the class token and position information, and passes the tokens through the transformer. With the default class-token pooling, it selects the updated class token and sends it through the classification head to produce ten scores per image.

- **`parse_args()`, `train()`, `test()`, and `run(args)` keep the roles described in Exercise 1.1.** We reuse them rather than writing a separate training loop for ViT.

For our configuration, the main tensor shapes are:

```text
Images:            [batch, 3, 32, 32]
Flattened patches: [batch, 16, 192]
Patch embeddings:  [batch, 16, 64]
With class token:  [batch, 17, 64]
Transformer output:[batch, 17, 64]
Class token:       [batch, 64]
Class scores:      [batch, 10]
```

## Running on CIP

From the project folder, using the course environment's Python:

```bash
nice -n 10 /proj/aimi-adl/envs/adl23_2/bin/python ex1_main.py --model vit --epochs 5
```

The `nice` setting gives the process lower CPU priority, following the assignment's guidance for shared CIP machines.

## Results

The best checkpoint was saved at epoch 5.

| Metric | Validation | Test |
| --- | --- | --- |
| Accuracy | 38.34% (1917/5000) | 39.31% (3931/10000) |
| Average cross-entropy loss | 1.7046 | 1.6989 |

ViT performed worse than ResNet18 in our five-epoch runs. Its test accuracy was 39.31%, compared with 64.80% for ResNet18, a difference of 25.49 percentage points. The validation and test scores were close for ViT. These runs compare the two models under the settings we used; they do not show how either model would perform after further tuning or longer training.

Assignment: [Sheet 1, Exercise 1.2](exercise-01.pdf). Implementation: [my_models_skeleton.py](my_models_skeleton.py) and [ex1_main.py](ex1_main.py).
