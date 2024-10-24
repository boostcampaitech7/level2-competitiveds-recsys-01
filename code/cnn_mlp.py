from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from models.CombinedModel import *
from handler.datasets import *
from utils.common_utils import *

def train(model, train_loader, loss_fn, optimizer, epochs, device, losses):
    # train 모드로 설정
    model.train()

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch in tqdm(train_loader):
            X_cnn,X_mlp, y = batch
            X_cnn, X_mlp, y = X_cnn.to(device), X_mlp.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X_cnn, X_mlp)
            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

def validate(model, valid_loader, device) -> int:
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            X_cnn, X_mlp, y = batch
            X_cnn, X_mlp, y = X_cnn.to(device), X_mlp.to(device), y.to(device)
            
            output = model(X_cnn, X_mlp)
            
            # 예측값을 numpy로 변환하고 (batch_size, 1) 형태로 조정 후 리스트에 저장
            all_preds.append(output.cpu().numpy().reshape(-1, 1))
            all_targets.append(y.cpu().numpy().reshape(-1, 1))  # 실제값도 같은 형태로 저장

    # 전체 데이터를 (len(valid), 1) 형태의 numpy 배열로 변환
    all_preds = np.vstack(all_preds)  # 세로로 쌓아서 (len(valid), 1) 배열 생성
    all_targets = np.vstack(all_targets)  # 실제값도 같은 방식으로 변환
    # MAE 계산
    mae_score = mean_absolute_error(all_targets, all_preds)
    print(f'\nMAE Score: {mae_score:.4f}')
    return mae_score

def inference_test(model, test_loader, device) -> np.ndarray:
    model.to(device)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            X_cnn, X_mlp = batch
            X_cnn, X_mlp = X_cnn.to(device), X_mlp.to(device)
            y_pred = model(X_cnn, X_mlp)
            all_preds.append(y_pred.cpu().numpy().reshape(-1,1))
            
        output = np.array(np.vstack(all_preds))

    return output

if __name__=='__main__':
    # valid MAE 따로 구하고 싶은 경우 주석 해제
    train_set = CombinedDataset(mode='train')
    valid_set = CombinedDataset(mode='valid')
    test_set = CombinedDataset(mode='test')

    train_loader = DataLoader(dataset=train_set, batch_size = 64, shuffle=False)
    valid_loader = DataLoader(dataset=valid_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

    mlp_input = len(train_set.mlp[0][0][0])
    device='cuda'
    model = CombinedModel(mlp_input).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    losses = []

    ### train 시작
    train(model = model, train_loader= train_loader, 
        loss_fn = criterion, optimizer = optimizer, epochs = epochs, 
        device = device, losses = losses)
    
    ### validation
    validate(model = model, valid_loader = valid_loader,device = device)

    ### inference with test set
    exp_title = 'CNN+MLPv4'
    submission = inference_test(model=model, test_loader=test_loader, device=device)
    submission = pd.DataFrame(submission)
    submission_to_csv(submission, exp_title)