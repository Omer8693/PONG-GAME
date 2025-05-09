import pygame
import sys
sys.path.append("c:\\Users\\Admin\\Desktop\\Pong Game")
import Constants

class Paddle(pygame.sprite.Sprite):
    def __init__(self, color):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([Constants.PADDLE_WIDTH, Constants.PADDLE_HEIGHT])
        self.image.fill(color)
        self.rect = self.image.get_rect()

    def Move(self, action):
        if action == 0:  # Up
            self.rect.y -= Constants.MOVE_COEF
        elif action == 2:  # Down
            self.rect.y += Constants.MOVE_COEF

        # Sınır kontrolü
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y > Constants.HEIGHT - Constants.PADDLE_HEIGHT:
            self.rect.y = Constants.HEIGHT - Constants.PADDLE_HEIGHT