import ChessEngine
import torch
import random
import numpy as np
import ChessAI

def board_to_tensor(state: ChessEngine.chess_state) -> torch.Tensor:
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


model = ChessAI.load_model('trained_chess_model.pth')
random_wins = 0
ai_wins = 0

for i in range(100): # 100 игр ИИ vs Random
    chess_state = ChessEngine.chess_state.default_state()
    is_ai_turn = bool(random.randint(0,1)) 
    color = 'ai за белых' if is_ai_turn else 'ai за чёрных'

    while chess_state.game_state == 'game_is_on':
        possible_moves = ChessEngine.chess_engine.all_possible_moves(chess_state)
        if is_ai_turn:
            moves_ratings = {}
            for move in possible_moves:
                tensor_state = board_to_tensor(ChessEngine.chess_engine.get_state_after_move(chess_state,move))
                rating = ChessAI.evaluate_position(model,tensor_state)
                moves_ratings[move] = rating
            move = max(moves_ratings, key=lambda k: moves_ratings[k])
        else:
            move = random.choice(possible_moves)

        chess_state = ChessEngine.chess_engine.get_state_after_move(chess_state,move)
        is_ai_turn = not is_ai_turn

    if is_ai_turn and (chess_state.game_state == 'white_lose' or chess_state.game_state == 'black_lose'):
            random_wins +=1
            print(i, "Победа Random")
    elif not is_ai_turn and (chess_state.game_state == 'white_lose' or chess_state.game_state == 'black_lose'):
            ai_wins += 1
            print(i, "Победа AI")
    else:
        print(i, "Ничья")

print("Победы AI: ",ai_wins/100)
print("Победы Random: ", random_wins/100)
print("Ничьи: ", (100-ai_wins-random_wins)/100)
        