# Exercise 1 study guide

This guide explains the ideas behind our code in simple words. Read it alongside `ex1_main.py` and `my_models_skeleton.py`. Exercises 1.1–1.3 are implemented. Exercise 1.4 is optional and has not been implemented.

## 1. What are we trying to do?

We want a program to recognise the category of an image. CIFAR10 contains 32 × 32 colour images from ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

An image is a collection of numbers. Each pixel has red, green, and blue values. A neural network transforms these numbers into ten scores. The category with the highest score is its prediction.

We compare three networks:

| Exercise | Model | Main idea |
| --- | --- | --- |
| 1.1 | ResNet18 | Learn visual patterns using convolutions |
| 1.2 | ViT | Split images into patches and connect them using attention |
| 1.3 | CrossViT | Process two patch sizes and exchange information between them |

The model does not receive written rules describing a cat or car. It learns numerical parameters, called weights, from labelled examples.

## 2. Libraries and Python tools

| Import | Why we use it |
| --- | --- |
| `argparse` | Read command-line options such as `--epochs 5` |
| `pathlib.Path` | Construct file paths and create directories |
| `torch` | Store tensors, calculate gradients, and run computations on CPU or GPU |
| `torch.nn` | Build layers, models, and loss functions |
| `torch.optim` | Update model weights during training |
| `torchvision.datasets` | Download and load CIFAR10 |
| `torchvision.transforms` | Prepare and augment images |
| `torchvision.models` | Create the supplied ResNet18 architecture |
| `DataLoader` | Deliver images and labels in batches |
| `einops.rearrange` / `Rearrange` | Reshape tensors using readable dimension names |
| `einops.repeat` | Repeat the class token across a batch |
| `torch.einsum` | Express attention matrix calculations with named dimensions |

`torch.nn.functional` is imported in the skeleton, but our implementation does not need to use it directly. An import alone does not mean a function is being used.

A **tensor** is a numerical array. Its **shape** tells us how many entries it has along each dimension. A batch of 64 CIFAR10 images has shape `[64, 3, 32, 32]`: images, colour channels, height, width.

## 3. Understanding the project files

`ex1_main.py` handles the experiment: settings, data, model selection, training, validation, saving, and final testing.

`my_models_skeleton.py` defines ViT, CrossViT, and their building blocks. Despite its name, we have filled in the required model components.

`data/cifar10` contains downloaded data. `checkpoints` contains saved model parameters. Saving parameters is different from saving the entire Python program: we still need the matching model definition to load them.

## 4. Exercise 1.1: preparing the data

### Separate learning from evaluation

We use 45,000 images for training, 5,000 for validation, and the separate 10,000-image test set for final evaluation.

Training images update the weights. Validation images help us select a checkpoint. Test images measure the selected model's performance. We do not select a model based on its test score.

`torch.randperm()` creates a shuffled list of image indices. We take the first 90% for training and the rest for validation. `Subset` selects those indices. A seeded `torch.Generator` makes this split repeatable. It does not make every part of training repeatable: our script does not seed all sources of randomness.

### Augmentation and preprocessing

Our training transforms run in this order:

1. `RandomCrop(32, padding=4)` adds a border and chooses a random 32 × 32 area. This simulates small shifts.
2. `RandomHorizontalFlip()` mirrors an image with probability 0.5. A mirrored car is still a car.
3. `ToTensor()` converts these images into floating-point tensors, scaling pixel values to the range 0–1.
4. `Normalize()` subtracts a mean and divides by a standard deviation for each colour channel.

Normalization is preprocessing, not random augmentation. We keep the supplied ImageNet values: mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. These are not statistics we calculated from CIFAR10.

Validation and test images only use the last two operations. Randomly changing evaluation images would make comparisons noisier.

We use separate dataset objects for training and validation because subsets of a single dataset share its transform. The objects read the same underlying collection, but their selected indices do not overlap.

### Batches and epochs

A batch is a group of images processed together. An epoch is one pass through the training set. With 45,000 images and batches of 64, an epoch contains 704 batches, including a smaller final batch.

The training loader shuffles examples so batches change order between epochs. Our current loaders use a hard-coded batch size of 64, even though the argument parser exposes `--batch-size`.

## 5. ResNet18 and classification

Convolutional layers learn filters that detect local image patterns. Deeper layers combine simpler patterns into more complex ones.

ResNet uses residual connections. Instead of replacing an input entirely, a block adds a learned update to it: `output = input + update`. These connections help information and gradients travel through the network.

We create ResNet18 with `weights=None`, so it starts without pretrained knowledge. We replace `model.fc` with `nn.Linear(model.fc.in_features, 10)` because CIFAR10 has ten classes.

A linear layer computes a learned weighted combination of its inputs, usually with a bias. Its output scores are called **logits**. They are not percentages and do not need to sum to one.

## 6. Loss, gradients, and learning

**Accuracy** counts correct predictions. **Loss** measures prediction quality in a way the optimizer can use. Two models can have equal accuracy but different loss because their confidence differs.

`CrossEntropyLoss` is suitable when each image belongs to one of several classes. It accepts logits and integer class labels. We do not apply softmax before passing logits to this loss.

Our loss uses `reduction='sum'`. During training we divide by the number of images in the batch to get an average. During evaluation we add the batch losses and divide by the total number of images.

One training step is:

```python
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target) / len(output)
loss.backward()
optimizer.step()
```

- `zero_grad()` clears previous gradients. PyTorch normally accumulates them.
- `model(data)` runs the forward pass and produces predictions.
- The loss compares predictions with the correct labels.
- `backward()` calculates how the loss changes with each parameter. These derivatives are gradients.
- `step()` uses the gradients to adjust parameters.

SGD stands for stochastic gradient descent. Our learning rate is `0.003`; it controls update size. Momentum is `0.9`; it uses previous updates to smooth the direction of learning. A very large learning rate can make training unstable, while a very small one can make progress slow.

## 7. The main script's functions

### `parse_args()`

This defines command-line options and returns them in `args`. For example, `args.model` is `vit` when we pass `--model vit`. Defaults apply when an option is omitted.

### `train()`

This trains for one epoch. It calls `model.train()`, loops over batches, moves data to the selected device, and performs the five learning steps above.

### `test()`

Despite its name, this function handles both validation and testing. `model.eval()` changes the behaviour of layers such as dropout and batch normalization. `torch.no_grad()` disables gradient tracking. These are different operations, and we use both.

`output.argmax(dim=1)` selects the highest-scoring class for each image. Comparing predictions with labels gives the number correct. Accuracy is `correct / total`; multiplying by 100 displays a percentage.

### `run(args)`

This connects everything: create paths, prepare datasets and loaders, select the model, create the optimizer, train and validate for each epoch, save improvements, reload the best checkpoint, and test it.

The final `if __name__ == '__main__':` block starts the experiment when the file is run directly. Importing the file does not execute that block.

### Saving the best model

We start `best_val_accuracy` at `-1.0`, ensuring the first epoch is saved. A later epoch replaces the checkpoint only if validation accuracy is higher. Ties keep the earlier checkpoint.

`torch.save(model.state_dict(), path)` saves parameters and persistent buffers. `load_state_dict(torch.load(...))` restores them into a model with the matching architecture. `map_location=device` controls where loaded tensors are placed.

The saved file does not include optimizer state or training progress, so it is not a full training-resumption checkpoint. A new run with the same model name overwrites its checkpoint when it saves an improvement in that run.

## 8. Exercise 1.2: from images to tokens

ViT treats an image as a sequence of patch representations.

For 8 × 8 patches, a 32 × 32 image gives 16 patches. Each patch has `8 × 8 × 3 = 192` values. `Rearrange` groups the pixels into patches, and a linear layer converts each patch to 64 features. This representation is an **embedding**.

A learned **class token** is placed at the start of the sequence. It gathers information through attention and supplies the final classification representation. We also add learned **positional embeddings** so the model can distinguish patch locations.

```text
Images                 [batch, 3, 32, 32]
Flattened patches      [batch, 16, 192]
Embedded patches       [batch, 16, 64]
Including class token  [batch, 17, 64]
After transformer      [batch, 17, 64]
Selected class token   [batch, 64]
Class scores           [batch, 10]
```

`nn.Parameter` registers a tensor as a trainable parameter. We use it for the class token and position embeddings. `repeat()` supplies a copy of the learned token to every image; these are not separately learned parameters for each image.

## 9. How attention works

Each token produces three vectors using learned linear layers:

- **Query (Q):** what information is this token looking for?
- **Key (K):** what information can this token match?
- **Value (V):** what information can this token contribute?

These descriptions are helpful analogies. The vectors themselves are learned numbers, not words or explicit questions.

Attention computes:

```text
scores = Q × transpose(K) / sqrt(head dimension)
weights = softmax(scores)
output = weights × V
```

A larger query-key score gives that value more weight. Softmax is applied over the key positions, making the weights sum to one before dropout.

For a simple example, weights `[0.7, 0.2, 0.1]` produce `0.7 × value1 + 0.2 × value2 + 0.1 × value3`. Each query gets its own set of weights.

We divide by the square root of the head dimension to control score magnitude. Dropout randomly removes some attention contributions during training; it is disabled during evaluation.

### Multiple attention heads

Our ViT has eight heads, each with 64 features. The internal projection therefore has 512 features, even though each input token has only 64 features. Do not assume the head size must be `64 / 8`: the skeleton explicitly uses `dim_head=64`.

```text
Q, K, V after projection  [batch, 17, 512]
After splitting heads    [batch, 8, 17, 64]
Attention scores         [batch, 8, 17, 17]
Weighted values          [batch, 8, 17, 64]
Combined heads           [batch, 17, 512]
Final output projection  [batch, 17, 64]
```

`chunk(2, dim=-1)` splits the combined key/value projection into two tensors. `einsum('b h i d, b h j d -> b h i j', q, k)` computes every query-key dot product: `b` is batch, `h` is head, `i` is query position, `j` is key position, and `d` is feature position.

### Transformer building blocks

`PreNorm` applies `LayerNorm` before another layer. Layer normalization normalizes features within each token; it differs from image normalization in the dataset transforms.

`FeedForward` applies a linear layer, GELU activation, dropout, a second linear layer, and dropout. GELU adds a nonlinear transformation, allowing the network to learn more than a sequence of linear mappings. This block processes each token separately.

`Transformer.forward()` applies attention and the feedforward block with residual connections:

```python
x = attn(x) + x
x = ff(x) + x
```

`nn.ModuleList` registers a collection of layers so PyTorch tracks their parameters. `nn.Sequential` both registers layers and applies them in order when called.

`ViT.forward()` connects patch embedding, class and position tokens, transformer blocks, and classification. Our default selects `x[:, 0]`, the class token. The skeleton also supports mean pooling, but we did not use it in these runs. `nn.Identity()` simply returns its input unchanged.

## 10. Exercise 1.3: two scales with CrossViT

CrossViT sends the same image through two branches:

| Branch | Patch size | Patch count | Features per token | Tokens including CLS |
| --- | --- | --- | --- | --- |
| Small patches | 8 × 8 | 16 | 64 | 17 |
| Large patches | 16 × 16 | 4 | 128 | 5 |

Small patches describe smaller regions. Large patches describe larger regions. Each branch first learns relationships within its own sequence using self-attention.

### Self-attention versus cross-attention

Self-attention gets Q, K, and V from the same sequence. Cross-attention gets Q from one source and K/V from another.

In our CrossViT, only a branch's class token queries the other branch's patches. The small class token reads large patches; the large class token reads small patches. We include the projected querying class token itself among the keys and values using `kv_include_self=True`.

### `ProjectInOut`

The branches have different embedding sizes. This adapter projects a class token to the required size, applies the wrapped attention layer, and projects back:

```text
Small CLS: 64 → 128 → attention to large patches → 64
Large CLS: 128 → 64 → attention to small patches → 128
```

`*args` and `**kwargs` forward extra positional and named arguments to the wrapped layer, including `context`.

### `CrossTransformer`

This separates each class token from its patch tokens, updates the class token through cross-attention, adds a residual connection, and rejoins the sequence. Patch tokens are unchanged by this block. There is no feedforward block here.

### `MultiScaleEncoder`

Each stage runs the small transformer, the large transformer, and then cross-attention. Our configuration repeats this stage three times. Each branch transformer has two blocks, and each cross-attention stage has two layers.

The dictionaries `sm_enc_params` and `lg_enc_params` hold branch settings. Writing `**sm_enc_params` passes their entries as named arguments to `Transformer`.

### `ImageEmbedder` and `CrossViT`

`ImageEmbedder` groups patch conversion, learned class and position tokens, and dropout into a reusable component. Each branch has its own instance.

`CrossViT.forward()` embeds the image twice, runs the multi-scale encoder, and sends each updated class token through its own classification head. Both heads produce `[batch, 10]`. Their logits are added before cross-entropy loss is calculated.

## 11. Running on CIP

The project files must be copied to CIP before running. Editing files on the Mac does not automatically update the CIP copy.

```bash
cd ~/Desktop/ADL
nice -n 10 /proj/aimi-adl/envs/adl23_2/bin/python ex1_main.py --model r18 --epochs 5
nice -n 10 /proj/aimi-adl/envs/adl23_2/bin/python ex1_main.py --model vit --epochs 5
nice -n 10 /proj/aimi-adl/envs/adl23_2/bin/python ex1_main.py --model cvit --epochs 5
```

Run the commands one at a time. `nice` lowers CPU scheduling priority; it does not reserve or limit GPU use. Using the full interpreter path avoids relying on Conda activation.

A quick forward-pass check is available with:

```bash
/proj/aimi-adl/envs/adl23_2/bin/python my_models_skeleton.py
```

It should print two `[16, 10]` shapes, one for ViT and one for CrossViT. Correct shapes show that the forward passes run, but do not prove that all gradients or learning behaviour are correct.

### Common problems

- `No module named torch`: check that the command uses the course interpreter, not system Python.
- Missing skeleton import: copy both Python files to the same project folder.
- Pytest opens instead of training: run the script as a Python program, not the function named `test` as a pytest test.
- A terminal shows `>` after a pasted heredoc: finish the block with its closing marker, such as `PY`, on its own line.
- A Mac cannot find `/proj/aimi-adl/...`: connect to CIP first; that path belongs to the Linux course machines.

Our script automatically uses CUDA when available, otherwise CPU. The existing `--no-cuda` argument is currently not checked in that choice. Likewise, `--dry-run` stops training after one batch but still runs evaluation, so it is not a complete fast-run mode.

## 12. Understanding our results

| Model | Reported validation accuracy | Test accuracy | Test loss |
| --- | --- | --- | --- |
| ResNet18 | 66.08%, best at epoch 5 | 64.80% | 1.0128 |
| ViT | 38.34%, best at epoch 5 | 39.31% | 1.6989 |
| CrossViT | 39.90%, final epoch shown | 40.43% | 1.6587 |

The CrossViT output did not include the last saving message, so its best validation score and selected epoch are not recorded. The test score uses the checkpoint selected by the script.

ResNet18 performed best in these runs. CrossViT exceeded ViT by 1.12 percentage points. These are individual five-epoch runs with the settings we used, not proof that one architecture is always better.

Convolutional networks build in assumptions about local image structure. Transformers learn relationships through attention and can need different training settings. We have not run controlled experiments to establish the cause of the differences here.

**Overfitting** means fitting training examples well while performing worse on unseen examples. Similar validation and test scores alone do not prove that overfitting is absent; training performance and learning curves also matter.

**Underfitting** means the model has not learned the task well enough. More training may help, but improvement is not guaranteed. Repeatedly choosing settings using test results would turn the test set into another validation set.

## 13. Optional Exercise 1.4: viewing attention

The assignment asks us to inspect attention from the class token to patch tokens in the last ViT self-attention block. This is not implemented yet.

A possible implementation would capture the attention weights after softmax in the last block, run a trained model in evaluation mode, select query position zero (the class token), and select key positions 1–16 (the patches). Averaging across heads gives sixteen values that can be arranged into a 4 × 4 grid and displayed over the image.

The highest value identifies the patch with the largest attention weight under that aggregation. Attention is useful for inspection, but the highest attention weight is not automatically proof that a patch caused the prediction. Value vectors, other heads, residual connections, and later layers also affect the output.

## 14. Questions to check your understanding

1. Why do we choose checkpoints using validation accuracy rather than test accuracy?
2. Why do training and validation need separate transforms?
3. Why does CIFAR10 need ten output scores?
4. What is the difference between an epoch and a batch?
5. Why do we clear gradients before each update?
6. Why does ViT have 17 tokens for our patch size?
7. What do Q, K, and V do in attention?
8. Why must the attention output return to the original token dimension before a residual addition?
9. Which tokens change inside the cross-attention block?
10. Why do CrossViT's classification outputs have matching shapes even though its token dimensions differ?

Answers: validation guides selection while test data is reserved for final evaluation; evaluation should avoid random augmentation; there are ten categories; an epoch covers the full training set while a batch is one group; gradients otherwise accumulate; sixteen patches plus one class token; queries and keys determine weights applied to values; residual addition needs matching shapes; only the class tokens; each classification head maps its branch representation to the same ten classes.

Assignment: [exercise-01.pdf](../exercise-01.pdf). Code: [ex1_main.py](../ex1_main.py) and [my_models_skeleton.py](../my_models_skeleton.py). Experiment summary: [README.md](../README.md).
