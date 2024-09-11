import os
import cv2
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.metrics import f1_score


# Definir Dataset
class OilSpillDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))
        try:
            targets = self.parse_annotation(label_path)
        except ValueError as e:
            print(e)
            return None  # Retornar None se não houver anotações válidas

        if self.transform:
            img = self.transform(img)

        return img, targets

    def parse_annotation(self, label_path):
        with open(label_path, 'r') as file:
            annotations = file.readlines()

        masks = []
        boxes = []
        labels = []

        img_height, img_width = 640, 640  # Assumindo que suas imagens são 640x640

        for annotation in annotations:
            ann = annotation.strip().split()
            class_id = int(ann[0])

            # Ignorar classes que não sejam 0 ou 1
            if class_id not in [0, 1]:
                print(f"Atenção: Classe inválida {class_id} encontrada em {label_path}. Ignorando essa anotação.")
                continue

            coords = list(map(float, ann[1:]))

            # Converter as coordenadas normalizadas para as dimensões da imagem
            poly_coords = np.array(coords).reshape(-1, 2)
            poly_coords[:, 0] *= img_width  # Converter a coordenada x
            poly_coords[:, 1] *= img_height  # Converter a coordenada y

            # Criar uma máscara para o polígono
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(poly_coords)], 1)
            masks.append(mask)

            # Criar uma caixa delimitadora (bounding box)
            x_min, y_min = np.min(poly_coords, axis=0)
            x_max, y_max = np.max(poly_coords, axis=0)
            boxes.append([x_min, y_min, x_max, y_max])

            # Adicionar o rótulo
            labels.append(class_id)

        if not labels:
            raise ValueError(f"Não foram encontradas anotações válidas em {label_path}.")

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)  # Otimizar a conversão
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return {'boxes': boxes, 'labels': labels, 'masks': masks}


# Função collate_fn que remove amostras inválidas (None)
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Remover entradas None
    return tuple(zip(*batch)) if batch else ([], [])  # Lidar com batch vazio

# Configurações
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Transformações para a imagem
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Caminhos
train_dataset = OilSpillDataset('train/images', 'train/labels', transform=data_transform)
val_dataset = OilSpillDataset('val/images', 'val/labels', transform=data_transform)
test_dataset = OilSpillDataset('test/images', None, transform=data_transform)

# Definição dos loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Carregar o modelo Mask R-CNN pré-treinado
model = maskrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, num_classes=2)  # Classe 0 = fundo, Classe 1 = derramamento

# Otimizador e critério de perda
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# model = model.cuda()

# Treinamento
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        if not images:  # Verificar se o batch está vazio
            continue

        images = [img.cpu() for img in images]  # Em vez de img.cuda()
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]  # Em vez de v.cuda()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

# Validação
model.eval()
all_preds = []
all_gts = []

with torch.no_grad():
    for images, targets in val_loader:
        if not images:
            continue

        images = [img.cpu() for img in images]
        outputs = model(images)

        preds = outputs[0]['labels'].cpu().numpy()
        gts = targets[0]['labels'].cpu().numpy()

        all_preds.extend(preds)
        all_gts.extend(gts)

f1 = f1_score(all_gts, all_preds, average='macro')
print(f'F1 Score (Validation): {f1}')


# Teste
def test_model(test_loader, model):
    model.eval()
    all_test_preds = []

    with torch.no_grad():
        for images in test_loader:
            if not images:
                continue

            images = [img.cpu() for img in images]
            outputs = model(images)

            pred_classes = outputs[0]['labels'].cpu().numpy()
            pred_masks = outputs[0]['masks'].cpu().numpy()

            all_test_preds.append((pred_classes, pred_masks))

    return all_test_preds


test_predictions = test_model(test_loader, model)
