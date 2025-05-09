import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import numpy as np
import Constants
from Agent.Paddle import Paddle

# Screen dimensions and middle of paddle
WIDTH = Constants.WIDTH
HEIGHT = Constants.HEIGHT
MIDDLE = Constants.PADDLE_HEIGHT * 0.5

# Color definitions
background_color = (230, 220, 200)
ball_color = (200, 60, 60)
agent_color = (0, 150, 255)
bot_color = (255, 120, 120)

# Ball speed from constants
BALL_Xspeed = Constants.BALL_Xspeed
BALL_Yspeed = Constants.BALL_Yspeed

class PongEnv(gym.Env):
    """
    A custom Pong environment built with pygame and gymnasium.
    """

    class Ball(pygame.sprite.Sprite):
        """
        Represents the ball in the Pong game using Pygame's Sprite system.
        """
        def __init__(self):
            super().__init__()
            radius = int(WIDTH * 0.008)
            self.image = pygame.Surface([radius * 2, radius * 2], pygame.SRCALPHA)
            pygame.draw.circle(self.image, ball_color, (radius, radius), radius)
            self.rect = self.image.get_rect()
            self.Xspeed = BALL_Xspeed
            self.Yspeed = BALL_Yspeed

        def clip(self):
            # Bounce off top and bottom walls
            if self.rect.y > HEIGHT - self.rect.height or self.rect.y < 0:
                self.Yspeed *= -1
            # Return True if the ball goes off left or right (game over)
            if self.rect.x <= 0 or self.rect.x >= WIDTH - self.rect.width:
                return True
            return False

    def __init__(self, complex_mode=False):
        """
        Initialize the Pong environment.
        """
        super(PongEnv, self).__init__()
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.action_space = spaces.Discrete(3)  # 0: up, 1: stay, 2: down
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.complex_mode = complex_mode
        self.paddle_height = Constants.PADDLE_HEIGHT
        self.timestep_count = 0

        self.Agent = Paddle(agent_color)
        self.Bot = Paddle(bot_color)
        self.BALL = self.Ball()

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.Agent, self.Bot, self.BALL)

        self.reward = 0
        self.current_distance = self.Distance_Between_Paddle_Ball()
        self.info = {}
        self.reset()

    def step(self, action):
        """
        Executes one environment step based on the agent's action.
        """
        self.render()
        self.reward = 0
        self.timestep_count += 1
        terminated = False
        truncated = False

        # Update agent and ball positions
        self.Agent.Move(action)
        self.BALL.rect.x += self.BALL.Xspeed
        self.BALL.rect.y += self.BALL.Yspeed
        self.Bot.rect.y = self.BALL.rect.y - WIDTH * 0.02  # Bot follows ball

        # Ball collides with bot
        if self.Bot.rect.colliderect(self.BALL.rect):
            self.BALL.Xspeed = -abs(self.BALL.Xspeed)
            self.BALL.rect.x = self.Bot.rect.left - self.BALL.rect.width - 1
            speed_change = random.uniform(0.9, 1.1)
            self.BALL.Xspeed *= speed_change
            self.BALL.Yspeed *= speed_change
            if self.complex_mode:
                self.BALL.Xspeed *= 1.1
                self.BALL.Yspeed *= 1.1

        # Ball collides with agent
        if self.Agent.rect.colliderect(self.BALL.rect):
            self.BALL.Xspeed = abs(self.BALL.Xspeed)
            self.BALL.rect.x = self.Agent.rect.right + 1
            speed_change = random.uniform(0.9, 1.1)
            self.BALL.Xspeed *= speed_change
            self.BALL.Yspeed *= speed_change
            self.reward += 30
            if self.complex_mode:
                self.BALL.Xspeed *= 1.1
                self.BALL.Yspeed *= 1.1
            if abs(self.BALL.Xspeed) < 1:
                self.BALL.Xspeed = 2 * np.sign(self.BALL.Xspeed)

        # In complex mode, reduce paddle size every 100 timesteps
        if self.complex_mode and self.timestep_count % 100 == 0:
            self.paddle_height = max(self.paddle_height * 0.95, 20)
            self.Agent.image = pygame.transform.scale(self.Agent.image, (Constants.PADDLE_WIDTH, int(self.paddle_height)))
            self.Agent.rect = self.Agent.image.get_rect(topleft=(self.Agent.rect.x, self.Agent.rect.y))
            self.Bot.image = pygame.transform.scale(self.Bot.image, (Constants.PADDLE_WIDTH, int(self.paddle_height)))
            self.Bot.rect = self.Bot.image.get_rect(topleft=(self.Bot.rect.x, self.Bot.rect.y))

        # Check for out-of-bounds
        if self.BALL.clip():
            terminated = True
            self.reward -= 30
            return self.get_observation(), self.reward, terminated, truncated, self.info

        # Reward shaping
        if self.Distance_Between_Paddle_Ball() < self.current_distance:
            self.reward += 0.1
        if self.Agent.rect.bottom > self.BALL.rect.y > self.Agent.rect.y:
            self.reward += 1

        self.current_distance = self.Distance_Between_Paddle_Ball()
        return self.get_observation(), self.reward, terminated, truncated, self.info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new initial state.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reward = 0
        self.timestep_count = 0
        self.paddle_height = Constants.PADDLE_HEIGHT

        # Reset positions
        self.Agent.rect.x = int(WIDTH * 0.02)
        self.Agent.rect.y = random.randint(0, int(HEIGHT / 10)) * 10
        self.Bot.rect.x = int(WIDTH * 0.95)
        self.Bot.rect.y = random.randint(0, int(HEIGHT / 10)) * 10
        self.BALL.rect.x = int(WIDTH * 0.5)
        self.BALL.rect.y = int(HEIGHT * 0.25) + random.randint(0, int(HEIGHT * 0.4))

        # Random initial ball direction
        self.BALL.Xspeed = random.choice([-BALL_Xspeed, BALL_Xspeed])
        self.BALL.Yspeed = random.choice([-BALL_Yspeed, BALL_Yspeed])
        self.current_distance = self.Distance_Between_Paddle_Ball()

        # Resize paddles if complex mode is on
        if self.complex_mode:
            self.Agent.image = pygame.Surface([Constants.PADDLE_WIDTH, self.paddle_height])
            self.Agent.image.fill(agent_color)
            self.Bot.image = pygame.Surface([Constants.PADDLE_WIDTH, self.paddle_height])
            self.Bot.image.fill(bot_color)

        return self.get_observation(), {}

    def render(self):
        """
        Renders the environment to the screen using Pygame.
        """
        self.screen.fill(background_color)
        self.all_sprites.draw(self.screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        pygame.display.update()

    def get_observation(self):
        """
        Constructs the current state representation.
        """
        return np.array([
            self.current_distance / WIDTH,
            self.Agent.rect.y / HEIGHT,
            self.BALL.rect.x / WIDTH,
            self.BALL.rect.y / HEIGHT,
            self.BALL.Xspeed / BALL_Xspeed,
            self.BALL.Yspeed / BALL_Yspeed
        ], dtype=np.float32)

    def Distance_Between_Paddle_Ball(self):
        """
        Calculates Euclidean distance between paddle center and ball.
        """
        return np.linalg.norm(
            np.array([self.Agent.rect.x, self.Agent.rect.y + MIDDLE]) -
            np.array([self.BALL.rect.x, self.BALL.rect.y])
        )

    def close(self):
        """
        Closes the environment.
        """
        if self.screen is not None:
            pygame.quit()
