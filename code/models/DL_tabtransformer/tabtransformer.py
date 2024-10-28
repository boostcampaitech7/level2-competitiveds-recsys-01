import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    def __init__(self, num_features, cat_emb_dim, n_heads=4, n_layers=2, output_dim=1, dropout_rate=0.5):
        super(TabTransformer, self).__init__()
        
        # Transformer Encoder for Categorical Variables
        transformer_layer = nn.TransformerEncoderLayer(d_model=cat_emb_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)

        # Fully Connected Layers for Numerical Variables with Batch Normalization and Dropout
        self.num_fc = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.BatchNorm1d(32),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 드롭아웃 추가
            nn.Linear(32, 16)
        )

        # Final Fully Connected Layer for Prediction with Dropout
        self.fc = nn.Sequential(
            nn.Linear(cat_emb_dim + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 드롭아웃 추가
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 드롭아웃 추가
            nn.Linear(32, output_dim)  # 최종 출력
        )

    def forward(self, x_num, x_cat):
        # Transformer on categorical data
        x_cat_transformed = self.transformer(x_cat.unsqueeze(1)).squeeze(1)  # (batch_size, cat_emb_dim)

        # Fully connected layers on numerical data
        x_num_processed = self.num_fc(x_num)  # (batch_size, 16)

        # Concatenate numerical and categorical features
        x = torch.cat([x_num_processed, x_cat_transformed], dim=1)  # (batch_size, 16 + cat_emb_dim)

        return self.fc(x)
