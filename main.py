import pygame
import json
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board
from take_it_easy.utils import transform_state
FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")

def save_transition_to_json(filename, data_dict):
    """ saves the transition to a json file """
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
            existing_data.update(data_dict)  # Update if the file exists
        with open(filename, 'w') as f:
            json.dump(existing_data, f)
    except FileNotFoundError:
        with open(filename, 'w') as f:  # Write if the file doesn't exist
            json.dump(data_dict, f)

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
                    transition = {
                        i: {
                            "state_t": state_t,
                            "action": action,
                            "state_t1": state_t1,
                            "reward": reward,
                            "finished": board.finished
                        }
                    }
                    save_transition_to_json('molt_plays.json',transition)
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