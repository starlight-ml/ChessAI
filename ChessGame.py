import pygame
import ChessEngine
import os

class ChessField():
        def __init__(self,
            screen,
            game_width,
            game_height,
            color_white = (250, 208, 160),
            color_black = (211,166,112),
            ):
            self.color_white = color_white
            self.color_black = color_black
            self.screen = screen
            # Расчёт поля для игры
            self.chessfield_w = (game_width // 2) * 0.85
            self.chessfield_start_x = game_width // 3.5 - 0.5*self.chessfield_w
            self.chessfield_h = self.chessfield_w
            self.chessfield_start_y = game_height//2 - 0.5*self.chessfield_h
            self.chessfield_cell_size = self.chessfield_h // 8
            # Словарь с изображениями фигур для игры
            self.pieces = self.load_piece_images()
            # Словарь соотношений (число : фигура)
            self.piece_type = self.get_pieces_types()
            # состояние доски
            self.state = ChessEngine.chess_state.default_state()
            self.selected_piece = None

        def set_state(self, state: ChessEngine.chess_state = ChessEngine.chess_state.default_state()):
            self.state = state
            self.draw()

        def draw(self):
            font_size = 20
            font = pygame.font.SysFont('Times New Roman',font_size)
            start_x = self.chessfield_start_x
            start_y = self.chessfield_start_y
            field_h = self.chessfield_h
            cell_size = self.chessfield_cell_size
            board = self.state.board

            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0: current_cell_color = self.color_white
                    else: current_cell_color = self.color_black
                    pygame.draw.rect(self.screen, current_cell_color,( start_x + i * cell_size, start_y + j* cell_size, cell_size,cell_size))
                    if board[j,i] != 0:
                        figure_num = int(board[j,i])
                        figure_name = self.piece_type[figure_num]
                        figure_img = self.pieces[figure_name]
                        self.screen.blit(figure_img,(start_x + i * cell_size,start_y + j* cell_size))

            for i, let in enumerate(['a','b','c','d','e','f','g','h']):
                text = font.render(let,True,(0,0,0))
                self.screen.blit(text,(start_x - font_size//1.5, start_y + cell_size * i + cell_size//2 - font_size//2))
                text = font.render(str(i+1), True, (0,0,0))
                self.screen.blit(text,(start_x + cell_size * i + cell_size//2 - font_size//3, start_y + field_h))

            pygame.display.flip()

        def load_piece_images(self):
            pieces = {}
            for color in ['white_','black_']:
                for piece_type in ['pawn','horse','bishop','rook','queen','king']:
                    key = f'{color}{piece_type}'
                    try:
                        img = pygame.image.load(fr'шахматные фигуры\{key}.png')
                        img = pygame.transform.scale(img, (self.chessfield_cell_size, self.chessfield_cell_size))
                        pieces[key] = img
                    except:
                        print(f"Ошибка загрузки изображения для {key}")
            return pieces
    
        def get_pieces_types(self):
            return {
                1:'white_pawn',
                2:'white_horse',
                3:'white_bishop',
                4:'white_rook',
                5:'white_queen',
                6:'white_king',
                -1:'black_pawn',
                -2:'black_horse',
                -3:'black_bishop',
                -4:'black_rook',
                -5:'black_queen',
                -6:'black_king',
        }

        def is_pos_in_click_area(self,click_pos_x,click_pos_y):
            if (click_pos_x > self.chessfield_start_x 
                and click_pos_x < self.chessfield_start_x + self.chessfield_w 
                and click_pos_y > self.chessfield_start_y 
                and click_pos_y < self.chessfield_start_y + self.chessfield_h
                ):
                return True
            return False

        def try_click(self,click_pos):
            click_pos_x,click_pos_y = click_pos[0],click_pos[1]
            # если клик вне доски, ты return
            if not self.is_pos_in_click_area(click_pos_x,click_pos_y): return

            # получаем номер клетки
            cell_y = int((click_pos_y - self.chessfield_start_y)//self.chessfield_cell_size)
            cell_x = int((click_pos_x - self.chessfield_start_x)//self.chessfield_cell_size)
            
            if self.selected_piece == None:
                # раскрасить в новый цвет
                self.selected_piece = (cell_y,cell_x)
                legal_moves =  ChessEngine.chess_engine.check_figure_possible_moves((cell_y,cell_x),self.state)
                print(legal_moves)
            else:
                # Если нажали на уже выбранную фигуру, то ничего не происходит
                if (cell_y,cell_x) == self.selected_piece: return
                # Если нажали на свою же фигуру, то перевыбираем фигуру
                if self.state.board[cell_y,cell_x] * self.state.board[self.selected_piece] > 0:
                    self.selected_piece = (cell_y,cell_x)
                    legal_moves =  ChessEngine.chess_engine.check_figure_possible_moves((cell_y,cell_x),self.state)
                    print(legal_moves)
                    return
                # Иначе рассматриваем вариант с возможностью хода
                legal_moves = ChessEngine.chess_engine.check_figure_possible_moves(position=self.selected_piece, state = self.state)
                for move in legal_moves:
                    if (cell_y,cell_x) == move[1]:
                        self.set_state(ChessEngine.chess_engine.get_state_after_move(self.state,move))
                        break
                self.selected_piece = None

class GameInterface():

    def __init__(self,
        game_width = 1000,
        game_height = 550,
        background_color = (244, 241, 232)
        ):
        pygame.display.set_caption("fay-chess")
        self.screen = pygame.display.set_mode((game_width,game_height))
        self.game_width = game_width
        self.game_height = game_height
        self.background_color = background_color
        
    def create_game_interface(self):
        self.elements = []
        # Создаём шаматное поле
        chess_field = ChessField(screen = self.screen, game_width=self.game_width,game_height=self.game_height)
        self.elements.append(chess_field)
        self.draw_chess_interface()

    def click(self, click_pos):
        for element in self.elements:
            element.try_click(click_pos)

    def draw_chess_interface(self):
        self.screen.fill(self.background_color)
        # отрисовка шахматного поля
        for element in self.elements:
            element.draw()

        pygame.display.flip()

pygame.init()

interface = GameInterface(game_width = 1000,game_height = 550)
clock = pygame.time.Clock()
interface.create_game_interface()

game_mode = {
    '1vs1',
    '1vsAI'
}

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            interface.click(mouse_pos)
            pass