import pygame

from take_it_easy.board import Board
from take_it_easy.constants import HEIGHT, WIDTH

FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")


# idx = number_moves() + 1
def main():
    run = True
    i = 1
    clock = pygame.time.Clock()
    board = Board(WIN)
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                # save previous state
                action, valid = board.action_by_mouse(event.pos)
            if event.type == pygame.MOUSEBUTTONUP:
                pass
            if event.type == pygame.QUIT:
                run = False

        board.render_board()
        pygame.display.update()
        i += 1
    pygame.quit()


if __name__ == "__main__":
    main()
