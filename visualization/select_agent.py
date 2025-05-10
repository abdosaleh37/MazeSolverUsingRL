import pygame
import sys

class AgentSelector:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Select Agent Type")
        
        # Load and scale background image
        self.background = pygame.image.load("assets/background.jpg")
        self.background = pygame.transform.scale(self.background, (self.width, self.height))
        
        # Colors
        self.colors = {
            'button': (128, 0, 128),
            'button_hover': (75, 0, 75),
            'text': (255, 255, 255)
        }
        
        # Fonts
        self.label_font = pygame.font.Font(None, 50)
        self.button_font = pygame.font.Font(None, 36)
        
        # Button dimensions and positions
        self.button_width, self.button_height = 300, 80
        self.button_space = 20
        y_buttons = 480
        self.q_learning_button = pygame.Rect(
            80, y_buttons, self.button_width, self.button_height
        )
        self.policy_gradient_button = pygame.Rect(
            120 + self.button_width + self.button_space, y_buttons, self.button_width, self.button_height
        )

    def draw_button(self, rect, text, hover=False):
        color = self.colors['button_hover'] if hover else self.colors['button']
        pygame.draw.rect(self.screen, color, rect, border_radius=15)
        self.draw_maze_game_style_text(
            self.screen, text, rect.center, self.button_font,
            self.colors['text'], (180, 120, 255), (0, 0, 0), spacing=2
        )

    def draw_maze_game_style_text(self, surface, text, center, font, main_color, glow_color, shadow_color=None, spacing=4):
        text = text.upper()
        x, y = center
        text_height = font.size("A")[1]
        y = y - text_height // 2
        total_width = sum(font.size(c)[0] + spacing for c in text) - spacing
        start_x = x - total_width // 2

        # Glow
        for dx, dy in [(-4, 0), (4, 0), (0, -4), (0, 4), (-2, -2), (2, 2), (-2, 2), (2, -2)]:
            letter_x = start_x + dx
            for c in text:
                letter_surface = font.render(c, True, glow_color)
                surface.blit(letter_surface, (letter_x, y + dy))
                letter_x += font.size(c)[0] + spacing

        # Shadow
        if shadow_color:
            letter_x = start_x + 2
            for c in text:
                letter_surface = font.render(c, True, shadow_color)
                surface.blit(letter_surface, (letter_x, y + 2))
                letter_x += font.size(c)[0] + spacing

        # Main text
        letter_x = start_x
        for c in text:
            letter_surface = font.render(c, True, main_color)
            surface.blit(letter_surface, (letter_x, y))
            letter_x += font.size(c)[0] + spacing

    def run(self):
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.policy_gradient_button.collidepoint(mouse_pos):
                        pygame.quit()
                        return "pg"
                    elif self.q_learning_button.collidepoint(mouse_pos):
                        pygame.quit()
                        return "q"
            self.screen.blit(self.background, (0, 0))
            # Label background
            label_bg_rect = pygame.Rect(self.width // 2 - 260, 380, 520, 80)
            s = pygame.Surface((label_bg_rect.width, label_bg_rect.height), pygame.SRCALPHA)
            s.fill((40, 0, 80, 180))
            pygame.draw.rect(s, (40, 0, 80, 180), s.get_rect(), border_radius=30)
            self.screen.blit(s, label_bg_rect.topleft)
            # Label
            self.draw_maze_game_style_text(
                self.screen, "Select Agent Type",
                (self.width // 2, 430),
                self.label_font, self.colors['text'],
                (180, 120, 255), (0, 0, 0)
            )
            # Buttons
            self.draw_button(
                self.policy_gradient_button, "Policy Gradient",
                self.policy_gradient_button.collidepoint(mouse_pos)
            )
            self.draw_button(
                self.q_learning_button, "Q-Learning",
                self.q_learning_button.collidepoint(mouse_pos)
            )
            pygame.display.flip()

def select_agent():
    return AgentSelector().run() 