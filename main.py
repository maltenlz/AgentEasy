import pygame

from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board
FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")

def main():
    run = True
    i = 1
    clock = pygame.time.Clock()
    board = Board(WIN)
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                #board.action_by_mouse(event.pos)
                board.action_by_id((3,3))
            if event.type == pygame.MOUSEBUTTONUP:
                pass
            if event.type == pygame.QUIT:
                run = False
        
        board.draw_board()
        pygame.display.update()
        i += 1
    pygame.quit()

if __name__ == "__main__":
    main()