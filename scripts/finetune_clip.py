import math
import os,sys
import clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils

def clip_loss(
    image_emb,
    text_emb,
    logit_scale=None,
    temperature=0.07,
    label_smoothing=0.05,
):
    if logit_scale is not None:
        logits = logit_scale * (image_emb @ text_emb.T)
    else:
        logits = image_emb @ text_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
    return (loss_i + loss_t) / 2

def make_path(images_dir, image_id):
    return os.path.normpath(os.path.join(images_dir, str(image_id) + ".jpg"))

def configure_trainable_layers(model, n_visual=1, n_text=1):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.visual, "transformer"):
        for p in model.visual.transformer.resblocks[-n_visual:].parameters():
            p.requires_grad = True
    for p in model.transformer.resblocks[-n_text:].parameters():
        p.requires_grad = True
    model.text_projection.requires_grad = True
    model.visual.proj.requires_grad = True
    model.logit_scale.requires_grad = False

class NewsImagesDataset(Dataset):
    def __init__(self, dataframe, images_dir, preprocess):
        self.preprocess = preprocess
        self.samples = []
        for _, row in dataframe.iterrows():
            path = make_path(images_dir, row["image_id"])
            if not os.path.isfile(path):
                continue
            title = row["article_title"]
            title = str(title) if pd.notna(title) else ""
            self.samples.append((path, title))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {images_dir}. "
                "Place the MediaEval train bundle here or adjust dataset directory."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, title = self.samples[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        return image, title

def collate_batch(batch):
    images = torch.stack([b[0] for b in batch])
    titles = [b[1] for b in batch]
    return images, titles

def save_model(model, path,model_name):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_name": model_name},
        path,
    )

if __name__ == "__main__":
    dataset_dir = os.path.join("..", "dataset", "newsimages_train_26_v1.1")
    images_dir = os.path.join(dataset_dir, "news_images")
    dataset = os.path.join(dataset_dir, "news_articles.csv")
    model_name = "ViT-B/16"

    device = utils.get_device()

    model, preprocess = clip.load(model_name, device=device, jit=False)
    configure_trainable_layers(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", n_trainable)

    df = pd.read_csv(dataset)
    dataset = NewsImagesDataset(df, images_dir, preprocess)
    print("training pairs:", len(dataset))

    batch_size = 32
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    num_epochs = 3
    trainable = [p for p in model.parameters() if p.requires_grad]
    base_lr = 3e-6
    optimizer = torch.optim.AdamW(
        trainable,
        lr=base_lr,
        weight_decay=0.01,
        eps=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 2), eta_min=base_lr * 0.05
    )
    clip_max_norm = 0.5
    loss_temperature = 0.07
    label_smoothing = 0.05
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_steps = 0
        skipped = 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for images, titles in pbar:
            images = images.to(device, non_blocking=True)
            tokens = clip.tokenize(titles, truncate=True).to(device)

            image_emb = model.encode_image(images)
            text_emb = model.encode_text(tokens)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            if not torch.isfinite(image_emb).all() or not torch.isfinite(text_emb).all():
                skipped += 1
                pbar.set_postfix(loss="skip", emb="nan")
                continue

            loss = clip_loss(
                image_emb,
                text_emb,
                logit_scale=None,
                temperature=loss_temperature,
                label_smoothing=label_smoothing,
            )

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="skip", reason="loss")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip_max_norm)
            if not math.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="skip", reason="grad")
                continue

            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        mean = epoch_loss / max(n_steps, 1)
        print(
            f"epoch {epoch + 1} mean loss: {mean:.4f} (optimizer steps: {n_steps}, skipped: {skipped})"
        )

    model.eval()
    print("done.")


    finetuned_path = os.path.join("..", "artifacts", f"{model_name.lower().replace('/','_')}_clip_news_finetuned.pth")
    save_model(model, finetuned_path,model_name)





