import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
from typing import Optional
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from ChessEngine import chess_engine,chess_state
import random
import concurrent.futures
from functools import partial

class ChessCNN(nn.Module):
    """
    CNN архитектура для оценки шахматных позиций
    """
    def __init__(self, input_channels=12):
        super(ChessCNN, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Skip connection
        self.skip_conv = nn.Conv2d(256, 128, kernel_size=1) if 256 != 128 else nn.Identity()
        
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        
        # Используем LeakyReLU вместо ReLU
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        # Первые два слоя
        # Первый слой
        x1 = self.activation(self.bn1(self.conv1(x)))
        
        # Второй слой
        x2 = self.activation(self.bn2(self.conv2(x1)))
        
        # Третий слой с skip connection от x2
        residual = self.skip_conv(x2)  # преобразуем 256->128 каналов
        x3 = self.conv3(x2)  # 256->128 каналов
        x3 = self.activation(self.bn3(x3) + residual)
        
        x = self.global_pool(x3)
        x = x.view(x.size(0), -1)
        
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))
        
        return x
    
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Упрощенное представление - только фигуры
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)  # Только 12 каналов для фигур
    
    piece_to_channel = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = 7 - square // 8, square % 8
            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = channel_offset + piece_to_channel[piece.piece_type]
            tensor[channel, row, col] = 1
    
    return torch.from_numpy(tensor)

def load_model(model_path: Optional[str] = None) -> ChessCNN:
    """
    Загружает модель из файла или создает новую
    """
    model = ChessCNN()
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
            print(f"Модель загружена из {model_path}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}, создана новая модель")
    else:
        print("Создана новая модель")
    
    return model

def evaluate_position(model: ChessCNN, tensor) -> float:
    """
    Оценивает шахматную позицию с помощью модели
    
    Возвращает:
        Оценку от -1 (полный перевес черных) до 1 (полный перевес белых)
    """
    # Добавляем batch dimension
    tensor = tensor.unsqueeze(0)  # [1, 18, 8, 8]
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Вычисляем оценку
    with torch.no_grad():
        evaluation = model(tensor)
    
    return evaluation.item()

def print_evaluation(evaluation: float):
    """
    Красиво выводит оценку позиции
    """
    if evaluation > 0.2:
        print(f"Сильный перевес белых: +{evaluation:.3f}")
    elif evaluation > 0:
        print(f"Небольшой перевес белых: +{evaluation:.3f}")
    elif evaluation < -0.2:
        print(f"Сильный перевес черных: {evaluation:.3f}")
    elif evaluation < 0:
        print(f"Небольшой перевес черных: {evaluation:.3f}")
    else:
        print("Равная позиция")

def create_training_data_from_pgn(pgn_path: str, max_games: int = 100) -> list:
    """
    Создает обучающие данные из PGN файла
    """
    training_data = []
    white_wins = 0
    black_wins = 0
    draw = 0
    game_counter = 0
    
    with open(pgn_path) as pgn_file:
        while game_counter < max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            headers = game.headers
            white_elo = int(headers.get("WhiteElo", "N/A"))
            black_elo = int(headers.get("BlackElo", "N/A"))

            if (black_elo + white_elo) / 2 < 2000:
                continue

            # Определяем победителя
            result = game.headers.get('Result', '')
            if result == "1/2-1/2":
                target = 0.0  # Ничья
                draw +=1
            elif result == "1-0":
                target = 1.0  # Победа белых
                white_wins +=1
            elif result == "0-1":
                target = -1.0  # Победа черных
                black_wins+=1
            else:
                continue  # Пропускаем игры с неизвестным результатом
            
            # Добавляем все позиции из этой партии в обучающие данные
            board = game.board()
            for move in game.mainline_moves():
                # Добавляем текущую позицию
                board_tensor = board_to_tensor(board)
                training_data.append((board_tensor, target))
                
                # Делаем ход
                board.push(move)
            game_counter += 1
            if game_counter % 100 == 0:
                print(game_counter)
    print("Победы белых: ",white_wins/(white_wins+black_wins+draw))
    print("Победы чёрных: ",black_wins/(white_wins+black_wins+draw))
    print("Ничьи: ",draw/(white_wins+black_wins+draw))
    return training_data

def train_model_parallel(model: ChessCNN, train_data: list, device, batch_size=32, epochs=10, num_workers=4):
    tensors = [item[0] for item in train_data]
    targets = [item[1] for item in train_data]
    
    tensor_stack = torch.stack(tensors)
    target_stack = torch.tensor(targets, dtype=torch.float32)
    
    dataset = TensorDataset(tensor_stack, target_stack)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch_tensors, batch_targets) in enumerate(dataloader):
            batch_tensors = batch_tensors.to(device)
            batch_targets = batch_targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(batch_tensors)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % max(1, len(dataloader) // 10) == 0:  # 10 раз за эпоху
                progress = batch_idx / len(dataloader) * 100
                print(f'Эпоха {epoch+1}, Прогресс: {progress:.1f}%, Loss: {loss.item():.4f}')
                torch.save(model.state_dict(), "trained_chess_model.pth")

        avg_loss = total_loss / len(dataloader)
        print(f'Эпоха {epoch+1} завершена, Средний Loss: {avg_loss:.4f}')

def state_to_tensor(state: chess_state) -> torch.Tensor:
    """
    Упрощенное представление - только фигуры
    Конвертирует состояние chess_state в тензор размерности (12, 8, 8)
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)  # 12 каналов для фигур
    
    # Соответствие типов фигур каналам (аналогично оригинальной функции)
    piece_to_channel = {
        1: 0,   # пешка
        2: 1,   # конь
        3: 2,   # слон
        4: 3,   # ладья
        5: 4,   # ферзь
        6: 5    # король
    }
    
    # Проходим по всем фигурам в словаре figures
    for piece_type, positions in state.figures.items():
        if not positions:  # пропускаем пустые списки
            continue
            
        # Определяем смещение канала в зависимости от цвета фигуры
        channel_offset = 0 if piece_type > 0 else 6
        # Получаем тип фигуры без учета знака (цвета)
        abs_piece_type = abs(piece_type)
        
        # Получаем соответствующий канал
        channel = channel_offset + piece_to_channel[abs_piece_type]
        
        # Заполняем тензор для каждой позиции фигуры
        for pos in positions:
            row, col = pos
            # Преобразуем координаты: в шахматах обычно белые снизу, черные сверху
            # В вашем классе доска ориентирована как в numpy: [0,0] - верхний левый угол
            # Поэтому row уже соответствует строке (0-7), col - столбцу (0-7)
            tensor[channel, row, col] = 1
    
    return torch.from_numpy(tensor)

def play_game_and_get_data(model_state_dict, game_id):
    """
    Играет одну игру и возвращает данные для обучения
    - model_state_dict: state_dict модели (чтобы каждый процесс имел копию)
    """
    # Локально создаём модель в процессе
    model = ChessCNN()
    model.load_state_dict(model_state_dict)
    model.eval()

    all_states = []
    current_state = chess_state.default_state()
    all_states.append(state_to_tensor(current_state))

    while current_state.game_state == 'game_is_on':
        all_moves = chess_engine.all_possible_moves(current_state)
        if not all_moves:
            break

        # --- батчевый рассчёт всех ходов ---
        next_states = [chess_engine.get_state_after_move(current_state, mv) for mv in all_moves]
        batch_tensors = torch.stack([state_to_tensor(st) for st in next_states])

        with torch.no_grad():
            ratings = model(batch_tensors).squeeze(1).numpy()

        moves_ratings = dict(zip(all_moves, ratings))

        # Берём топ-N случайным образом
        moves_threshold = 3 if len(moves_ratings) > 2 else len(moves_ratings)
        top_moves = sorted(moves_ratings, key=lambda k: moves_ratings[k], reverse=True)[:moves_threshold]
        selected_move = random.choice(top_moves)

        current_state = chess_engine.get_state_after_move(current_state, selected_move)
        all_states.append(state_to_tensor(current_state))

    # --- награды ---
    if current_state.game_state in ['draw_50', 'draw_repeat', 'white_stalemate', 'black_stalemate']:
        white_reward = black_reward = -0.1
    elif current_state.game_state == 'white_lose':
        white_reward, black_reward = -5, 5
    else:  # black_lose
        white_reward, black_reward = 5, -5

    training_data = []
    for i, state_vec in enumerate(all_states):
        is_white_move = (i % 2 == 0)
        target = white_reward if is_white_move else black_reward
        training_data.append((state_vec, target))

    return training_data

def train_on_parallel_games(model, num_games=100, num_workers=None):
    if num_workers is None:
        num_workers = min(num_games, os.cpu_count() or 2)

    print(f"Количество игр параллельно: {num_workers}")

    # Сохраняем state_dict, чтобы раздать процессам (модель нельзя напрямую)
    model_state_dict = model.state_dict()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            partial(play_game_and_get_data, model_state_dict),
            range(num_games)
        ))

    all_training_data = []
    for game_data in results:
        all_training_data.extend(game_data)

    if all_training_data:
        train_model_parallel(
            model,
            all_training_data,
            torch.device('cpu'),
            batch_size=64,
            epochs=1,
            num_workers=num_workers
        )

# Финальная версия main
def main():
    model = load_model('chess_model_after_RL.pth')
    
    # Используем оптимальные настройки для Core Ultra 9
    optimal_workers = 6  # Начните с этого значения и ajust по результатам
    
    for epoch in range(50):  # Больше эпох, но меньше игр за эпоху
        print(f"Эпоха {epoch+1}")
        train_on_parallel_games(model, num_games=20, num_workers=optimal_workers)
        
        # Сохраняем модель каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"chess_model_after_RL.pth")


if __name__ == "__main__":
    main()