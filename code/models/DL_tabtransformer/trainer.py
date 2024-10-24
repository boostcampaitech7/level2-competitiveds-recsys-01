# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
from tqdm import tqdm

class TabTransformerTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        """TabTransformerTrainer 초기화."""
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        # 손실 및 MAE 추적 리스트 초기화
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_mae_list = []
        self.valid_mae_list = []

    @staticmethod
    def mean_absolute_error(predictions, targets):
        """MAE 계산 함수."""
        return torch.mean(torch.abs(predictions - targets))

    def train_epoch(self, X_train, y_train, batch_size):
        """하나의 에폭에 대해 학습 진행."""
        self.model.train()
        train_loss, train_mae = 0.0, 0.0

        for i in tqdm(range(0, len(X_train), batch_size), desc="Training"):
            X_batch = X_train[i:i + batch_size].to(self.device)
            y_batch = y_train[i:i + batch_size].to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)

            # 손실 및 MAE 계산
            loss = self.loss_fn(y_pred, y_batch)
            mae = self.mean_absolute_error(y_pred, y_batch)

            loss.backward()
            self.optimizer.step()

            # 배치별 손실과 MAE 누적
            train_loss += loss.item() * X_batch.size(0)
            train_mae += mae.item() * X_batch.size(0)

        # 평균 손실 및 MAE 계산
        train_loss /= len(X_train)
        train_mae /= len(X_train)

        # 결과 저장
        self.train_loss_list.append(train_loss)
        self.train_mae_list.append(train_mae)

        return train_loss, train_mae

    def validate_epoch(self, X_valid, y_valid, batch_size):
        """하나의 에폭에 대해 검증 진행."""
        self.model.eval()
        val_loss, val_mae = 0.0, 0.0

        with torch.no_grad():
            for i in range(0, len(X_valid), batch_size):
                X_batch = X_valid[i:i + batch_size].to(self.device)
                y_batch = y_valid[i:i + batch_size].to(self.device)

                y_pred = self.model(X_batch)

                # 손실 및 MAE 계산
                loss = self.loss_fn(y_pred, y_batch)
                mae = self.mean_absolute_error(y_pred, y_batch)

                # 배치별 손실과 MAE 누적
                val_loss += loss.item() * X_batch.size(0)
                val_mae += mae.item() * X_batch.size(0)

        # 평균 손실 및 MAE 계산
        val_loss /= len(X_valid)
        val_mae /= len(X_valid)

        # 결과 저장
        self.valid_loss_list.append(val_loss)
        self.valid_mae_list.append(val_mae)

        return val_loss, val_mae

    def train(self, X_train, y_train, X_valid, y_valid, batch_size, num_epochs, print_every=10):
        """전체 학습 루프."""
        for epoch in range(num_epochs):
            train_loss, train_mae = self.train_epoch(X_train, y_train, batch_size)
            val_loss, val_mae = self.validate_epoch(X_valid, y_valid, batch_size)

            # 주기적으로 출력
            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    def inference(self, test_loader):
        """테스트 데이터에 대한 추론."""
        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu())
                targets.append(y_batch.cpu())

        # 예측 및 실제값 텐서를 하나로 합침
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        mae = self.mean_absolute_error(predictions, targets)
        print(f"Test MAE: {mae.item():.4f}")

        return predictions, targets