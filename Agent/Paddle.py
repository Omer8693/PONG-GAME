import pygame
import sys

# Add the project directory to sys.path (adjust this path if needed)
sys.path.append("c:\\Users\\Admin\\Desktop\\Pong Game")

import Constants

class Paddle(pygame.sprite.Sprite):
    """
    Represents a paddle (agent or bot) in the Pong game using Pygame's sprite system.
    """

    def __init__(self, color):
        super().__init__()
        # Create a rectangular paddle surface and apply color
        self.image = pygame.Surface([Constants.PADDLE_WIDTH, Constants.PADDLE_HEIGHT])
        self.image.fill(color)
        self.rect = self.image.get_rect()  # Gets the rectangle object for positioning

    def Move(self, action):
        """
        Moves the paddle vertically based on the action:
        0 = up, 1 = stay, 2 = down
        """
        if action == 0:  # Move up
            self.rect.y -= Constants.MOVE_COEF
        elif action == 2:  # Move down
            self.rect.y += Constants.MOVE_COEF

        # Keep paddle within screen bounds
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y > Constants.HEIGHT - Constants.PADDLE_HEIGHT:
            self.rect.y = Constants.HEIGHT - Constants.PADDLE_HEIGHT
