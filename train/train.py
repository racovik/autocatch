import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.types import FileLike
from torchvision import datasets, models, transforms


# fodase os fundos, agora é tudo aleatório vai se fuder poketwo pokerealm pokemon pokehq
def process_with_random_background(image) -> Image.Image:
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # random color
    random_color = tuple(np.random.randint(0, 256, size=3))

    bg = Image.new("RGB", image.size, random_color)
    bg.paste(image, (0, 0), image)
    return bg.convert("RGB")


def process_with_white_background(img) -> Image.Image:
    """
    converter a imagem para fundo branco para melhorar a visualização do modelo, já q o fundo do bot é transparente.
    """
    img = img.convert("RGBA")
    white_background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img_final = Image.alpha_composite(white_background, img)

    return img_final.convert("RGB")


class Trainer:
    def __init__(
        self,
        data_dir="./pokerealmdataset",
        batch_size=32,
        epochs=10,
        lr=0.001,
        data_loader_num_workers=0,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"running on cuda: {self.device}")
        self.train_transform = transforms.Compose(
            [
                transforms.Lambda(process_with_random_background),
                transforms.RandomResizedCrop(
                    224, scale=(0.7, 1.0)
                ),  # Aprende que o poke pode estar em qualquer lugar/tamanho
                transforms.RandomHorizontalFlip(),  # Aprende que o poke pode estar virado
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2
                ),  # Ignora pequenas mudanças de cor
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_dataset = datasets.ImageFolder(
            self.data_dir, transform=self.train_transform
        )
        self.num_classes = len(self.train_dataset.classes)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=data_loader_num_workers,
        )
        self.model = self.load_model()

    class TrainedModel:
        def __init__(self, model_state_dict, classes):
            self.model_state_dict = model_state_dict
            self.classes = classes

        def save(self, path):
            torch.save(
                {"model_state_dict": self.model_state_dict, "classes": self.classes},
                path,
            )

    def load_model(self) -> models.MobileNetV2:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)  # type: ignore
        model = model.to(self.device)

        return model

    def train(self) -> TrainedModel:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.classifier[1].parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculamos apenas sobre o set de treino
            epoch_loss = running_loss / len(self.train_dataset)
            epoch_acc = running_corrects.double() / len(self.train_dataset)  # type: ignore
            duration = time.time() - start_time

            print(f"epoch {epoch + 1}/{self.epochs} | Time: {duration:.1f}s")
            print(f"  [Trainer] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print("-" * 35)
        return self.TrainedModel(self.model.state_dict(), self.train_dataset.classes)


class PokemonModel:
    def __init__(self, checkpoint_model_path: FileLike):
        self.MODEL_PATH = checkpoint_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"running on {self.device}")
        self.model, self.class_mapping = self.load_model()
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: process_with_white_background(img)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self) -> tuple[models.MobileNetV2, list[str]]:
        checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(
            num_ftrs,  # type: ignore
            len(checkpoint["classes"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        class_mapping = checkpoint["classes"]
        model.eval()
        return model, class_mapping

    def predict(self, img: Image.Image) -> tuple[str, float]:
        batch = self.transform(img).unsqueeze(0).to(self.device)  # type: ignore
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probabilities, 1)
            predicted_class_idx = int(preds.item())
            predicted_class = self.class_mapping[predicted_class_idx]
            percentage = conf.item() * 100
        return predicted_class, percentage


def load_args():
    argument_parser = argparse.ArgumentParser(description="entrenar el modelo")
    argument_parser.add_argument(
        "--data_dir",
        type=str,
        default="./pokerealmdataset",
        help="directorio del dataset de entrenamiento",
        dest="data_dir",
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="tamaño del batch para entrenamiento",
        dest="batch_size",
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="número de épocas para entrenamiento",
        dest="epochs",
    )
    argument_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="tasa de aprendizaje para el optimizador",
        dest="lr",
    )
    argument_parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=0,
        help="número de workers para DataLoader",
        dest="num_workers",
    )
    argument_parser.add_argument(
        "--path",
        type=str,
        default="pokerealm_model.pth",
        help="ruta para guardar el modelo entrenado",
        dest="path",
    )
    argument_parser.add_argument(
        "-i",
        "--inference",
        action="store_true",
        help="modo de inferencia (no entrena)",
        dest="inference",
    )
    argument_parser.add_argument(
        "--im",
        "--image",
        type=str,
        help="ruta de la imagen para inferencia",
        dest="image",
    )
    return argument_parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    if not args.inference:
        print("loading trainer with the following parameters:")
        print(
            f"  data_dir: {args.data_dir}\n  batch_size: {args.batch_size}\n  epochs: {args.epochs}\n  lr: {args.lr}\n num_workers: {args.num_workers}\n  model_save_path: {args.path}"
        )

        trainer = Trainer(args.data_dir, args.batch_size, args.epochs, args.lr)
        print("starting training...")
        trained_model = trainer.train()
        trained_model.save(args.path)
    else:
        print("loading inference model...")
        pokemon_model = PokemonModel(args.path)
        predicted_poke, percentage = pokemon_model.predict(Image.open(args.image))
        print(f"predicted: {predicted_poke} with {percentage}% confidence")
