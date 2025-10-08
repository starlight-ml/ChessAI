import numpy as np
import random

class ZobristHasher:
    def __init__(self):
        # Инициализируем случайные числа для каждой фигуры на каждой позиции
        self.piece_keys = np.zeros((8, 8, 13), dtype=np.uint64)
        self.castling_keys = {}
        self.en_passant_keys = {}
        self.side_key = None
        
        random.seed(42)  # Для воспроизводимости
        self._initialize_keys()
    
    def _initialize_keys(self):
        # Ключи для фигур на каждой позиции
        for y in range(8):
            for x in range(8):
                for piece_type in range(13):  # 12 типов фигур + пустая клетка
                    self.piece_keys[y, x, piece_type] = random.getrandbits(64)
        
        # Ключи для прав на рокировку
        castling_rights = ['white_long', 'white_short', 'black_long', 'black_short']
        for right in castling_rights:
            self.castling_keys[right] = random.getrandbits(64)
        
        # Ключи для взятия на проходе (для каждой колонки)
        for x in range(8):
            self.en_passant_keys[x] = random.getrandbits(64)
        
        # Ключ для стороны, которая ходит
        self.side_key = random.getrandbits(64)
    
    def compute_hash(self, state):
        """Вычисляет Zobrist hash для текущей позиции"""
        h = 0
        
        # Добавляем фигуры на доске
        for y in range(8):
            for x in range(8):
                piece = state.board[y, x]
                if piece != 0:
                    # Преобразуем номер фигуры в индекс (от -6 до 6 -> от 0 до 12)
                    piece_index = int(piece) + 6
                    h ^= self.piece_keys[y, x, piece_index]
        
        # Добавляем права на рокировку
        for right in state.castlings_rights:
            h ^= self.castling_keys[right]
        
        # Добавляем возможность взятия на проходе
        last_move = state.last_move
        if last_move: #type: ignore
            if len(last_move) > 2 and last_move[3] == 'double_step':
                y, x = last_move[1]  # Позиция, куда пошла пешка
                h ^= self.en_passant_keys[x]
        
        # Добавляем сторону, которая ходит
        if state.is_white_move:
            h ^= self.side_key #type: ignore
        
        return h

class chess_state():
    def __init__(self, board, is_white_move, castlings_rights, game_state, last_move, moves_50_draw_counter, material_adventage,figures = None, position_history = None):
        self.board = board
        if figures == None:
            figures = {1:[],2:[],3:[],4:[],5:[],6:[],-1:[],-2:[],-3:[],-4:[],-5:[],-6:[]}
            for y in range(board.shape[0]):
                for x in range(board.shape[1]):
                    if board[y,x] == 0: continue
                    if board[y,x] > 0:
                        figures[board[y,x]].append((y,x))
                    else:
                        figures[board[y,x]].append((y,x))
        self.figures = figures
        self.is_white_move = is_white_move
        self.castlings_rights = castlings_rights
        self.game_state = game_state
        self.last_move = last_move
        self.moves_50_draw_counter = moves_50_draw_counter
        self.material_adventage = material_adventage
        # История позиций для обнаружения повторений
        if position_history is None:
            self.position_history = {}
        else:
            self.position_history = position_history.copy()
        
    @classmethod
    def default_state(cls):
        # 1 - пешка, 2 - конь, 3 - слон, 4 - ладья, 5 - ферзь, 6 - король
        # чёрные также, только с минусом (-1,-2...)
        board = np.zeros((8,8))
        # Расставляем пешки
        board[1,:] = np.ones(8) * (-1) # чёрные
        board[6,:] = np.ones(8)
        # Остальные фигуры
        board[0,:] = np.array([4,2,3,5,6,3,2,4]) * (-1) # чёрные
        board[7,:] = np.array([4,2,3,5,6,3,2,4])

        # Добавляем фигуры в словарь figures['цвет фигур']['id фигуры'] - список позиций (y,x), на которых находятся эти фигуры
        figures = {1:[],2:[],3:[],4:[],5:[],6:[],-1:[],-2:[],-3:[],-4:[],-5:[],-6:[]}
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if board[y,x] == 0: continue
                if board[y,x] > 0:
                    figures[board[y,x]].append((y,x))
                else:
                    figures[board[y,x]].append((y,x))

        castlings_rights = ['white_long','white_short','black_long','black_short']
        last_move = None

        is_white_move = True
        game_state = 'game_is_on'
        moves_50_draw_counter = 0
        material_adventage = 0
        position_history = {}

        return cls(board, is_white_move, castlings_rights, game_state, last_move, moves_50_draw_counter,material_adventage, figures, position_history)
    
    @classmethod
    def light_state(cls):
        board = np.zeros((8,8))
        board[0,7] = -6
        board[2,4] = 4
        board[3,2] = 4
        board[4,1] = 4
        board[0,0] = 6

        # Добавляем фигуры в словарь figures['цвет фигур']['id фигуры'] - список позиций (y,x), на которых находятся эти фигуры
        figures = {1:[],2:[],3:[],4:[],5:[],6:[],-1:[],-2:[],-3:[],-4:[],-5:[],-6:[]}
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if board[y,x] == 0: continue
                if board[y,x] > 0:
                    figures[board[y,x]].append((y,x))
                else:
                    figures[board[y,x]].append((y,x))

        castlings_rights = []
        last_move = None

        is_white_move = True
        game_state = 'game_is_on'
        moves_50_draw_counter = 0
        material_adventage = 9
        position_history = {}

        return cls(board, is_white_move, castlings_rights, game_state, last_move, moves_50_draw_counter,material_adventage, figures, position_history)

    def copy(self):
        board_copy = np.copy(self.board)
        figures_copy = {}
        for key, value in self.figures.items():
            figures_copy[key] = value.copy()
        castlings_rights_copy = self.castlings_rights.copy()
        game_state = self.game_state + ''
        last_move = self.last_move[:] if self.last_move else None
        position_history_copy = self.position_history.copy()
        return chess_state(board_copy, self.is_white_move, castlings_rights_copy, game_state, last_move,self.moves_50_draw_counter, self.material_adventage, figures_copy,position_history_copy)

class chess_engine:

    zobrist_hasher = ZobristHasher()

    @classmethod
    def get_state_after_move(cls, state: chess_state, move: tuple, check_end: bool = True):
        # создаём тестовую доску, подобную основной
        from_pos, to_pos, move_type, extra_data = move
        state = state.copy()
        if state.game_state != 'game_is_on': return state
        go_piece = int(state.board[from_pos[0],from_pos[1]])
        end_piece = int(state.board[to_pos[0],to_pos[1]])

        # делаем возможный ход 
        if move_type == 'normal':
            state.board[to_pos[0],to_pos[1]] = go_piece
            state.board[from_pos[0],from_pos[1]] = 0

            state.figures[go_piece].remove((from_pos[0],from_pos[1]))
            state.figures[go_piece].append((to_pos[0],to_pos[1]))
            if end_piece != 0:
                state.figures[end_piece].remove((to_pos[0],to_pos[1]))
        # Рокировки
        elif move_type == 'castling_wl':
            state.board[7,2] = 6
            state.board[7,3] = 4
            state.board[7,0] = 0
            state.board[7,4] = 0
            state.figures[6].remove((7,4))
            state.figures[6].append((7,2))
            state.figures[4].remove((7,0))
            state.figures[4].append((7,3))
        elif move_type == 'castling_ws':
            state.board[7,6] = 6
            state.board[7,5] = 4
            state.board[7,7] = 0
            state.board[7,4] = 0
            state.figures[6].remove((7,4))
            state.figures[6].append((7,6))
            state.figures[4].remove((7,7))
            state.figures[4].append((7,5))
        elif move_type == 'castling_bl':
            state.board[0,2] = -6
            state.board[0,3] = -4
            state.board[0,0] = 0
            state.board[0,4] = 0
            state.figures[-6].remove((0,4))
            state.figures[-6].append((0,2))
            state.figures[-4].remove((0,0))
            state.figures[-4].append((0,3))
        elif move_type == 'castling_bs':
            state.board[0,6] = -6
            state.board[0,5] = -4
            state.board[0,7] = 0
            state.board[0,4] = 0
            state.figures[-6].remove((0,4))
            state.figures[-6].append((0,6))
            state.figures[-4].remove((0,7))
            state.figures[-4].append((0,5))
        # ход с превращением пешки
        elif move_type == 'promotion':
            promotion_piece = extra_data
            state.board[to_pos[0],to_pos[1]] = promotion_piece
            state.board[from_pos[0],from_pos[1]] = 0

            # key_error: 0 при превращении в коня, при установке состояния после хода в chessgame
            if go_piece == 0: print(move, end_piece)
            state.figures[go_piece].remove((from_pos[0],from_pos[1]))
            state.figures[promotion_piece].append((to_pos[0],to_pos[1]))
            if end_piece != 0:
                state.figures[end_piece].remove((to_pos[0],to_pos[1]))
        # взятие на проходе
        elif move_type == 'en_passant':
            attacked_pawn = extra_data
            state.board[to_pos[0],to_pos[1]] = go_piece
            state.board[from_pos[0],from_pos[1]] = 0
            state.board[attacked_pawn[0], attacked_pawn[1]] = 0

            state.figures[go_piece].remove((from_pos[0],from_pos[1]))
            state.figures[go_piece].append((to_pos[0],to_pos[1]))
            state.figures[-go_piece].remove(attacked_pawn)


        # пересмотр прав на рокировки
        if go_piece == 6:
            if 'white_long' in state.castlings_rights:
                state.castlings_rights.remove('white_long')
            if 'white_short' in state.castlings_rights:
                state.castlings_rights.remove('white_short')
        elif go_piece == -6:
            if 'black_long' in state.castlings_rights:
                state.castlings_rights.remove('black_long')
            if 'black_short' in state.castlings_rights:
                state.castlings_rights.remove('black_short')
        elif go_piece == 4 and from_pos == (7,7):
            if 'white_short' in state.castlings_rights:
                state.castlings_rights.remove('white_short')
        elif go_piece == 4 and from_pos == (7,0):
            if 'white_long' in state.castlings_rights:
                state.castlings_rights.remove('white_long')
        elif go_piece == -4 and from_pos == (0,0):
            if 'black_long' in state.castlings_rights:
                state.castlings_rights.remove('black_long')
        elif go_piece == -4 and from_pos == (0,7):
            if 'black_short' in state.castlings_rights:
                state.castlings_rights.remove('black_short')

        state.is_white_move = not state.is_white_move

        # Храним в списке последний ход
        state.last_move = move

        # Проверяем на 50 ходов без взятия и ходов пешками
        if abs(go_piece) == 1 or end_piece != 0:
            state.moves_50_draw_counter = 0
        else:
            state.moves_50_draw_counter += 1
        # Оцениваем материальное преимущество после хода
        if end_piece != 0:
            if abs(end_piece) == 1:
                state.material_adventage -= end_piece
            elif abs(end_piece) == 2 or abs(end_piece) == 3:
                state.material_adventage -= end_piece//abs(end_piece) * 3
            elif abs(end_piece) == 4:
                state.material_adventage -= end_piece//abs(end_piece) * 5
            elif abs(end_piece) == 5:
                state.material_adventage -= end_piece//abs(end_piece) * 9

        if check_end:
            current_hash = cls.zobrist_hasher.compute_hash(state)
            state.position_history[current_hash] = state.position_history.get(current_hash, 0) + 1
            state.game_state = cls.game_state(state)

        return state
    
    @classmethod
    def is_check(cls, state: chess_state, is_checking_white: bool) -> bool:
        state = state.copy()
        is_check = False
        king = 6 if is_checking_white else -6
        king_y, king_x = state.figures[king][0]

        # Проверка пешек
        if is_checking_white:
            if king_y - 1 >= 0 and king_x + 1 < state.board.shape[1]:
                if state.board[king_y - 1, king_x + 1] == -1:
                    is_check = True
            if king_y - 1 >= 0 and king_x - 1 >= 0:
                if state.board[king_y - 1, king_x - 1] == -1:
                    is_check = True
        else:
            if king_y + 1 < state.board.shape[0] and king_x + 1 < state.board.shape[1]:
                if state.board[king_y + 1, king_x + 1] == 1:
                    is_check = True
            if king_y + 1 < state.board.shape[0] and king_x - 1 >= 0:
                if state.board[king_y + 1, king_x - 1] == 1:
                    is_check = True

        if is_check:
            return True

        # Проверка коней
        for i in [-2, -1, 1, 2]:
            for j in [-2, -1, 1, 2]:
                if abs(i) == abs(j):
                    continue
                y, x = king_y + i, king_x + j
                if 0 <= y < state.board.shape[0] and 0 <= x < state.board.shape[1]:
                    piece = state.board[y, x]
                    if abs(piece) == 2 and piece * king < 0:
                        return True

        # Проверка слонов, ладей, ферзей и королей
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for i, j in directions:
            distance = 1
            while True:
                y = king_y + i * distance
                x = king_x + j * distance
                if not (0 <= y < state.board.shape[0] and 0 <= x < state.board.shape[1]):
                    break

                piece = state.board[y, x]
                if piece != 0:
                    if piece * king < 0:  # Вражеская фигура
                        # Проверяем тип фигуры в зависимости от направления
                        if abs(i) == abs(j):  # Диагональ
                            if abs(piece) in [3, 5, 6]:  # Слон, ферзь или король
                                if abs(piece) == 6 and distance == 1:
                                    return True
                                elif abs(piece) in [3, 5]:
                                    return True
                        else:  # Горизонталь/вертикаль
                            if abs(piece) in [4, 5, 6]:  # Ладья, ферзь или король
                                if abs(piece) == 6 and distance == 1:
                                    return True
                                elif abs(piece) in [4, 5]:
                                    return True
                    break  # Любая фигура прерывает луч
                distance += 1

        return False

    @classmethod
    def check_figure_possible_moves(cls, position, state : chess_state):
        state = state.copy()
        if state.game_state != 'game_is_on': return[]
        # Возвращает массив позиций, на которые можно переместить фигуру, находящуюся на position
        # position - кортеж (y,x)
        y,x = position
        figure = int(state.board[y,x])

        # если выбрано поле без фигуры, возвращаем пустой список
        if figure == 0:
            return []

        # проверяем, точно ли выбрана фигура того же цвета, чей ход
        if (figure < 0 and state.is_white_move) or (figure > 0 and  not state.is_white_move):
            # возвращаем пустой список, в котором нет возможных ходов
            return []

        possible_moves = []

        if abs(figure) == 1: # пешки
            # определяем направление хода и стартовую линию пешек, в зависимости от цвета
            if state.is_white_move: 
                step = -1
                start_line = 6
            else: 
                step = 1
                start_line = 1

            # вообще наверно нет смысла в проверке не стоит ли пешка на конечном поле, так как она обязана тогда превращаться

            # если нет препятствий, то можно на одну клетку вперёд
            if state.board[y + step,x] == 0:
                if (y + step != 0 and y + step != 7):
                    possible_moves.append((position,(y + step,x),'normal',None))
                else:
                    for promo_piece in [2, 3, 4, 5]:  # Варианты превращения
                        possible_moves.append((position,(y + step, x),'promotion', promo_piece * -step))

            # если пешка на стартовом поле, то даём возможность пойти на две клетки вперёд
            if y == start_line and state.board[y + 2*step,x] == 0 and state.board[y + 1*step,x] == 0:
                possible_moves.append((position,(y + 2*step,x),'normal','double_step'))
            
            # теперь проверяем, может ли пешка съесть фигуры
            if x < state.board.shape[1] - 1: # не стоит ли она на правом краю доски
                right_cell = state.board[y + step,x+1]
                # проверяем есть ли фигура врага на поле
                if right_cell * figure < 0:
                    if y != start_line + 5*step:
                        possible_moves.append((position,(y + step,x+1),'normal',None))
                    else:
                        for promo_piece in [2, 3, 4, 5]:  # Варианты превращения
                            possible_moves.append((position,(y + step,x+1),'promotion',promo_piece * -step))
                # Рассматриваем взятие на проходе
                if state.last_move and state.last_move[1] == (y,x+1) and state.last_move[3] == 'double_step':
                    possible_moves.append((position,(y + step,x+1),'en_passant',(y,x+1))) 
            if x > 0: # не стоит ли она на левом краю доски
                left_cell = state.board[y + step,x-1]
                # проверяем есть ли фигура врага на поле
                if left_cell * figure < 0:
                    if y != start_line + 5*step:
                        possible_moves.append((position,(y + step,x-1),'normal',None))
                    else:
                        for promo_piece in [2, 3, 4, 5]:  # Варианты превращения
                            possible_moves.append((position,(y + step,x-1),'promotion',promo_piece * -step))
                # Рассматриваем взятие на проходе
                if state.last_move and state.last_move[1] == (y,x-1) and state.last_move[3] == 'double_step':
                    possible_moves.append((position,(y + step,x-1),'en_passant',(y,x-1))) 

        elif abs(figure) == 2: # кони
            for i in [-2,-1,1,2]:
                for j in [-2,-1,1,2]:
                    # у коня нет ходов (+1,+1),(-2,2)...
                    if abs(i) == abs(j): continue
                    # проверяем не выходит ли возможный ход за пределы поля
                    if y + i >= state.board.shape[0] or x + j >= state.board.shape[1]: continue
                    if y + i < 0 or x + j < 0: continue
                    # проверяем не блокируется ли ход союзной фигурой
                    if state.board[y + i, x + j] * figure > 0: continue
                    # если все проверки пройдены, то можно добавлять ход
                    possible_moves.append((position,(y + i,x + j),'normal',None))
        
        elif abs(figure) == 3: # слоны
            # создаём цикл для расчёта движений в каждую из четырёх сторон
            for i,j in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                distance = 1
                # запускаем цикл движений слона в одну сторону, пока он не уткнется в край поля
                while (y + i*distance < state.board.shape[0] and 
                       y + i*distance >= 0 and
                       x + j*distance < state.board.shape[1] and
                       x + j*distance >= 0
                    ):
                    # завершаем цикл преждевременно, если на поле для хода союзная фигура
                    if state.board[y + i*distance, x + j*distance] * figure > 0: break
                    possible_moves.append((position,(y+i*distance,x+j*distance),'normal',None))
                    # завершаем цикл преждевременно, если на поле для хода вражеская фигура, после добавления хода в возможные
                    if state.board[y + i*distance, x + j*distance] * figure < 0: break
                    distance += 1

        elif abs(figure) == 4: # ладьи
            # создаём цикл для расчёта движений в каждую из четырёх сторон
            for i,j in [(1,0),(-1,0),(0,1),(0,-1)]:
                distance = 1
                # запускаем цикл движений ладьи в одну сторону, пока она не уткнется в край поля
                while (y + i*distance < state.board.shape[0] and 
                       y + i*distance >= 0 and
                       x + j*distance < state.board.shape[1] and
                       x + j*distance >= 0
                    ):
                    # завершаем цикл преждевременно, если на поле для хода союзная фигура
                    if state.board[y + i*distance, x + j*distance] * figure > 0: break
                    possible_moves.append((position,(y+i*distance,x+j*distance),'normal',None))
                    # завершаем цикл преждевременно, если на поле для хода вражеская фигура, после добавления хода в возможные
                    if state.board[y + i*distance, x + j*distance] * figure < 0: break
                    distance += 1

        elif abs(figure) == 5: # королевы
        # создаём цикл для расчёта движений в каждую из восьми сторон
            for i,j in [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]:
                distance = 1
                # запускаем цикл движений королевы в одну сторону, пока она не уткнется в край поля
                while (y + i*distance < state.board.shape[0] and 
                    y + i*distance >= 0 and
                    x + j*distance < state.board.shape[1] and
                    x + j*distance >= 0
                    ):
                    # завершаем цикл преждевременно, если на поле для хода союзная фигура
                    if state.board[y + i*distance, x + j*distance] * figure > 0: break
                    possible_moves.append((position,(y+i*distance,x+j*distance),'normal',None))
                    # завершаем цикл преждевременно, если на поле для хода вражеская фигура, после добавления хода в возможные
                    if state.board[y + i*distance, x + j*distance] * figure < 0: break
                    distance += 1

        elif abs(figure) == 6: # короли
            # проверяем каждое поле вокруг фигуры
            for i,j in [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]:
                if (y + i < state.board.shape[0] and 
                    y + i >= 0 and
                    x + j < state.board.shape[1] and
                    x + j >= 0
                    ):
                    # переходим к следующему ходу, если на этом поле союзная фигура
                    if state.board[y + i, x + j] * figure > 0: continue
                    # добавляем ход в список возможных
                    possible_moves.append((position,(y+i,x+j),'normal',None))

            # проверяем возможность рокировки
            if figure == 6:
                if ('white_long' in state.castlings_rights
                    and state.board[7,0] == 4
                    and state.board[7,1] == 0
                    and state.board[7,2] == 0
                    and state.board[7,3] == 0
                    and not cls.is_check(state,True)
                    and not cls.is_check(cls.get_state_after_move(state,((7,4),(7,3),'castling_wl',None)),True)
                    ):
                    possible_moves.append((position,(7,2),'castling_wl',None))
                if ('white_short' in state.castlings_rights
                    and state.board[7,7] == 4
                    and state.board[7,6] == 0
                    and state.board[7,5] == 0
                    and not cls.is_check(state,True)
                    and not cls.is_check(cls.get_state_after_move(state,((7,4),(7,5),'castling_ws', None)),True)
                    ):
                    possible_moves.append((position,(7,6),'castling_ws',None))
            elif figure == -6:
                if ('black_long' in state.castlings_rights
                    and state.board[0,0] == -4
                    and state.board[0,1] == 0
                    and state.board[0,2] == 0
                    and state.board[0,3] == 0
                    and not cls.is_check(state,False)
                    and not cls.is_check(cls.get_state_after_move(state,((0,4),(0,3),'castling_bl',None)),False)
                    ):
                    possible_moves.append((position,(0,2),'castling_bl',None))
                if ('black_short' in state.castlings_rights
                    and state.board[0,7] == -4
                    and state.board[0,6] == 0
                    and state.board[0,5] == 0
                    and not cls.is_check(state,False)
                    and not cls.is_check(cls.get_state_after_move(state,((0,4),(0,5),'castling_bs',None)),False)
                    ):
                    possible_moves.append((position,(0,6),'castling_bs',None))
        legal_moves = []
        # После всего этого проверяем нет ли check после исполнения ходов (и другие ограничения)
        for move in possible_moves:
            # создаём тестовую доску, подобную основной
            state_after_move = cls.get_state_after_move(state,move,False)
            # проверяем не появляется ли check после возможного хода
            is_check_after_move = cls.is_check(state_after_move, state.is_white_move)
            # если ход не влечёт цвет собственному королю, то добавляем в легальные
            if not is_check_after_move:
                legal_moves.append(move)
                
        return legal_moves
    
    @classmethod
    def all_possible_moves(cls, state):
        state = state.copy()
        if state.game_state != 'game_is_on': return []
        # словарь[стартовая позиция] = список конечных позиций
        possible_moves = []
        for positions in state.figures.values():
            for pos in positions:
                possible_moves.extend(cls.check_figure_possible_moves(pos, state))
        return possible_moves
    
    @classmethod
    def game_state(cls, state: chess_state):
        if state.game_state != 'game_is_on': return state.game_state
        # Ничья при 50 ходах без взятия фигур и ходов пешками
        if state.moves_50_draw_counter == 50: return 'draw_50'
        # Проверка на троекратное повторение позиции
        current_hash = cls.zobrist_hasher.compute_hash(state)
        if state.position_history.get(current_hash, 0) >= 3:
            return 'draw_repeat'
        possible_moves = cls.all_possible_moves(state)
        # Если находим хоть один ход, то "игра идёт"
        if len(possible_moves) > 0: return 'game_is_on'
        if state.is_white_move:
            if cls.is_check(state, True): return 'white_lose'
            else: return 'white_stalemate' # - ничья при пате
        else:
            if cls.is_check(state,False): return 'black_lose'
            else: return 'black_stalemate' # - ничья при пате