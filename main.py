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
                # save previous state
                state_t = transform_state(board)
                score_t = board.smooth_value
                action, valid = board.action_by_mouse(event.pos)
                if valid:
                    state_t1 = transform_state(board)
                    score_t1 = board.smooth_value
                    reward = score_t1 - score_t
                    # save transition
                #board.action_by_id((3,3))
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