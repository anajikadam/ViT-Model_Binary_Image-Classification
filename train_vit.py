import torch
# from tqdm.notebook import tqdm
from tqdm import tqdm

from torch.utils.data import DataLoader

from transformers import ConvNextFeatureExtractor
from transformers import AutoModelForImageClassification
from datasets import load_metric
metric = load_metric("accuracy")
model_name = "facebook/convnext-tiny-224"

from dataset_load_train import load_my_dataset
labels, id2label, label2id, processed_dataset = load_my_dataset()

device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name,
                                                        num_labels=len(labels),
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        ignore_mismatched_sizes=True)

model.to(device)


def collate_fn(examples):
  pixel_values = torch.stack([example["pixel_values"] for example in examples])
  labels = torch.tensor([example["label"] for example in examples])
  return {"pixel_values": pixel_values, "labels": labels}

dataloader = DataLoader(processed_dataset["train"], collate_fn=collate_fn, batch_size=4, shuffle=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
num_epoch = 10
for epoch in range(num_epoch):
  print("Epoch:", epoch)
  correct = 0
  total = 0
  for idx, batch in enumerate(tqdm(dataloader)):
    # move batch to GPU
    batch = {k:v.to(device) for k,v in batch.items()}

    optimizer.zero_grad()

    # forward pass
    outputs = model(pixel_values=batch["pixel_values"],
                    labels=batch["labels"])

    loss, logits = outputs.loss, outputs.logits
    loss.backward()
    optimizer.step()

    # metrics
    total += batch["labels"].shape[0]
    predicted = logits.argmax(-1)
    correct += (predicted == batch["labels"]).sum().item()

    accuracy = correct/total

    if idx % 100 == 0:
      print(f"Loss after {idx} steps:", loss.item())
      print(f"Accuracy after {idx} steps:", accuracy)

output_dir = "Model"
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)


