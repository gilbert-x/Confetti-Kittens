import pygame
from pygame.math import Vector2  # for 2D vectors
import sys
import os  # for checking save file existence
import random
import math
from collections import deque
import json  # for saving game state

# Constants
WIDTH, HEIGHT = 800, 600       # Original Size
FPS = 60
SQUARE_SIZE = 16                # Size of one tile unit
CORRIDOR_SCALE = 3              # Corridor width/height factor (tiles per cell)
TILE_SIZE = SQUARE_SIZE * CORRIDOR_SCALE  # Pixel size of one maze cell (48px)
CREATURE_BASE = SQUARE_SIZE // 2  # Base size for creatures (8px)
PLAYER_SPEED = 5                # pixels per frame
NUM_EGGS = 10                   # Number of eggs to spawn in maze
NUM_GEMS = 8                    # Number of hidden gems to spawn
GEM_BURST_DURATION = 500        # ms duration of burst animation
GEM_TIMER_REDUCTION = 15000     # ms to subtract from each egg timer
EGG_ZONE_HEIGHT = TILE_SIZE     # Height of bottom UI bar for collected eggs (48px)
FOLLOW_DELAY = 15               # frames of delay between each follower

# Bright color palette for creatures
BRIGHT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (255, 0, 128), (128, 0, 255),
    (0, 128, 255), (128, 255, 0), (0, 255, 128)
]

# Initialize Pygame and set up the window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Confetti Kittens")
clock = pygame.time.Clock()

# --- Sound setup ---
try:
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
except Exception as _e:
    pass  # mixer may already be initialized

def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None

SOUNDS = {
    'meow': load_sound('meow.wav'),
    'gem': load_sound('gem.wav'),
    'egg': load_sound('egg.wav'),
    'hatch': load_sound('hatch.wav'),
    'level_up': load_sound('level_up.wav'),
}

def play_sound(name, volume=1.0, pan=0.5):
    """Play a sound by name with optional volume and stereo pan (0.0=left, 1.0=right)."""
    snd = SOUNDS.get(name)
    if not snd:
        return
    ch = pygame.mixer.find_channel(True)
    # Clamp
    pan = max(0.0, min(1.0, pan))
    volume = max(0.0, min(1.0, volume))
    left = volume * (1.0 - pan)
    right = volume * pan
    try:
        ch.set_volume(left, right)
    except TypeError:
        ch.set_volume(volume)
    ch.play(snd)

# --- Save helper ---
def save_game_to_file(filename, level, gem_score, player_pos, followers, eggs, gems, maze):
    """Serialize core game state to JSON."""
    now_save = pygame.time.get_ticks()
    save_data = {
        'level': level,
        'gem_score': gem_score,
        'player': {'x': float(player_pos.x), 'y': float(player_pos.y)},
        'followers': [
            {'x': float(f.pos.x), 'y': float(f.pos.y), 'tail_color': list(f.tail_color)}
            for f in followers
        ],
        'eggs': [
            {
                'col': e.col,
                'row': e.row,
                'state': e.state,
                'remaining_hatch': (
                    max(0, e.hatch_time - (now_save - (e.collected_time or now_save)))
                    if e.state == INVENTORY else e.hatch_time
                )
            }
            for e in eggs
        ],
        'gems': [
            {'col': g.col, 'row': g.row, 'state': g.state}
            for g in gems
        ],
        'maze': maze,
    }
    with open(filename, 'w') as f:
        json.dump(save_data, f)
    print(f'Saved to {filename}')

# States
UNCOLLECTED = 0
INVENTORY = 1
HATCHED = 2
GEM_HIDDEN = 0
GEM_DISCOVERED = 1
GEM_COLLECTED = 2
CREATURE_SPAWNED = 0
CREATURE_FOLLOW = 1

# Font for timers
font = pygame.font.SysFont(None, 24)

# Path history and kitten train
path_history = deque(maxlen=5000)  # stores player positions
kitten_train = []            # creatures in follow train


def generate_player_sprite(size=SQUARE_SIZE, direction=Vector2(0, 1)):
    """
    Create a placeholder player sprite with dynamic blue eyes.
    Inputs:
      size      - base tile size (px)
      direction - Vector2 indicating look direction
    Output:
      Surface of size (size, size*2) with white head (with eyes) and blue body
    """
    surf = pygame.Surface((size, size * 2), pygame.SRCALPHA)
    # Head (white)
    head_rect = pygame.Rect(0, 0, size, size)
    pygame.draw.rect(surf, (255, 255, 255), head_rect)
    # Compute eye offsets from direction
    if direction.length() != 0:
        dir_norm = direction.normalize()
    else:
        dir_norm = Vector2(0, 1)
    # Use a smaller offset so eyes remain visible on head
    max_offset = size // 8
    eye_offset_x = int(dir_norm.x * max_offset)
    eye_offset_y = int(dir_norm.y * max_offset)
    # Eye positions relative to head
    eye_r = 2
    left_eye = (size // 4 + eye_offset_x, size // 4 + eye_offset_y)
    right_eye = (3 * size // 4 + eye_offset_x, size // 4 + eye_offset_y)
    pygame.draw.circle(surf, (0, 0, 255), left_eye, eye_r)
    pygame.draw.circle(surf, (0, 0, 255), right_eye, eye_r)
    # Body (blue)
    body_rect = pygame.Rect(0, size, size, size)
    pygame.draw.rect(surf, (0, 0, 255), body_rect)
    return surf


def generate_egg_sprite(size=SQUARE_SIZE):
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    shell = tuple(random.randint(180, 255) for _ in range(3))
    pattern = tuple(random.randint(0, 100) for _ in range(3))
    cx, cy = size // 2, size // 2
    r = size // 2 - 2
    pygame.draw.circle(surf, shell, (cx, cy), r)
    typ = random.choice(['vertical', 'horizontal', 'speckles'])
    if typ == 'vertical':
        w = size // 4
        pygame.draw.rect(surf, pattern, (cx - w//2, 2, w, size - 4))
    elif typ == 'horizontal':
        h = size // 4
        pygame.draw.rect(surf, pattern, (2, cy - h//2, size - 4, h))
    else:
        for _ in range(8):
            x = random.randint(cx - r + 2, cx + r - 2)
            y = random.randint(cy - r + 2, cy + r - 2)
            pygame.draw.circle(surf, pattern, (x, y), 2)
    return surf


def generate_gem_sprite(color, size=SQUARE_SIZE):
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pts = [(size//2, 0), (size, size//2), (size//2, size), (0, size//2)]
    pygame.draw.polygon(surf, color, pts)
    return surf


def generate_creature_sprite(base=CREATURE_BASE, face_col=None, body_col=None):
    """
    Draw a cute kitten sprite half the player size, with big eyes, nose, whiskers, and fur patterns.
    Inputs:
      base - half-player size
      face_col - RGB tuple for face; if None, random from BRIGHT_COLORS
      body_col - RGB tuple for body; if None, random distinct from face_col
    Output: Surface with kitten graphic
    """
    size = base * 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    # Choose colors
    if face_col is None:
        face_col = random.choice(BRIGHT_COLORS)
    if body_col is None:
        pool = [c for c in BRIGHT_COLORS if c != face_col]
        body_col = random.choice(pool)
    eye_col = (0, 0, 255)
    glint_col = (255, 255, 255)
    nose_col = (255, 150, 150)
    whisker_col = (0, 0, 0)
    # Head
    cx, cy = size // 2, size // 2 - base//4
    pygame.draw.circle(surf, face_col, (cx, cy), base)
    # Ears
    ear_h = base//2
    left_ear = [(cx - base, cy), (cx - base//2, cy - ear_h), (cx - base//4, cy)]
    right_ear = [(cx + base, cy), (cx + base//2, cy - ear_h), (cx + base//4, cy)]
    pygame.draw.polygon(surf, face_col, left_ear)
    pygame.draw.polygon(surf, face_col, right_ear)
    # Fur patterns
    for i in range(2):
        offset = (-base//3 + i * (base//3))
        start = (cx + offset, cy - base//2)
        end = (cx + offset, cy)
        pygame.draw.line(surf, whisker_col, start, end, 1)
    # Eyes
    eye_r = base // 3
    pupil_r = eye_r // 2
    left_eye = (cx - base//2, cy)
    right_eye = (cx + base//2, cy)
    pygame.draw.circle(surf, glint_col, (left_eye[0] - pupil_r//2, left_eye[1] - pupil_r//2), pupil_r//2)
    pygame.draw.circle(surf, glint_col, (right_eye[0] - pupil_r//2, right_eye[1] - pupil_r//2), pupil_r//2)
    pygame.draw.circle(surf, eye_col, left_eye, eye_r)
    pygame.draw.circle(surf, eye_col, right_eye, eye_r)
    pygame.draw.circle(surf, (0, 0, 0), left_eye, pupil_r)
    pygame.draw.circle(surf, (0, 0, 0), right_eye, pupil_r)
    # Nose
    nose = [(cx, cy + base//4), (cx - base//8, cy + base//2), (cx + base//8, cy + base//2)]
    pygame.draw.polygon(surf, nose_col, nose)
    # Whiskers
    for side in (-1, 1):
        y0 = cy + base//3
        for i in range(3):
            y = y0 + (i - 1) * 2
            start = (cx, y)
            end = (cx + side * base, y)
            pygame.draw.line(surf, whisker_col, start, end, 1)
    # Body
    body_rect = pygame.Rect(size//4, cy + base//2, base, base)
    pygame.draw.ellipse(surf, body_col, body_rect)
    return surf, body_col

class Egg:
    def __init__(self, col, row):
        self.col, self.row = col, row
        self.x = col * TILE_SIZE + TILE_SIZE//2
        self.y = row * TILE_SIZE + TILE_SIZE//2
        self.sprite = generate_egg_sprite()
        self.rect = self.sprite.get_rect(center=(self.x, self.y))
        self.state = UNCOLLECTED
        self.hatch_time = random.randint(60000, 180000)
        self.collected_time = None

    def collect(self, now):
        if self.state == UNCOLLECTED:
            self.state = INVENTORY
            self.collected_time = now

    def update(self, now):
        if self.state == INVENTORY and now - self.collected_time >= self.hatch_time:
            self.state = HATCHED

    def draw(self, surface):
        if self.state == UNCOLLECTED:
            surface.blit(self.sprite, self.rect)

class Gem:
    def __init__(self, col, row):
        self.col, self.row = col, row
        self.x = col * TILE_SIZE + TILE_SIZE//2
        self.y = row * TILE_SIZE + TILE_SIZE//2
        self.rect = pygame.Rect(self.x - SQUARE_SIZE//2, self.y - SQUARE_SIZE//2,
                                 SQUARE_SIZE, SQUARE_SIZE)
        self.state = GEM_HIDDEN
        self.discovered_time = None
        self.color = tuple(random.randint(150, 255) for _ in range(3))
        self.sprite = generate_gem_sprite(self.color)

    def discover(self, now):
        if self.state == GEM_HIDDEN:
            self.state = GEM_DISCOVERED
            self.discovered_time = now

    def update(self, now):
        if self.state == GEM_DISCOVERED and now - self.discovered_time >= GEM_BURST_DURATION:
            self.state = GEM_COLLECTED

    def draw(self, surface, now):
        if self.state == GEM_DISCOVERED:
            elapsed = now - self.discovered_time
            ratio = elapsed / GEM_BURST_DURATION
            radius = int(TILE_SIZE * (1 + ratio))
            pygame.draw.circle(surface, self.color, (self.x, self.y), radius, 3)
        elif self.state == GEM_COLLECTED:
            surface.blit(self.sprite, self.sprite.get_rect(center=(self.x, self.y)))

class Creature:
    def __init__(self, x, y):
        self.x, self.y = x, y
        # Generate sprite and tail color
        sprite_data = generate_creature_sprite()
        if isinstance(sprite_data, tuple):
            self.sprite, self.tail_color = sprite_data
        else:
            self.sprite = sprite_data
            self.tail_color = (150, 75, 0)
        self.rect = self.sprite.get_rect(center=(self.x, self.y))
        self.state = CREATURE_SPAWNED
        self.pos = Vector2(self.x, self.y)
        # Roaming setup
        self.dir = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if self.dir.length() == 0:
            self.dir = Vector2(1, 0)
        else:
            self.dir = self.dir.normalize()
        self.roam_speed = 1
        self.next_dir_time = 0
        # Tail animation path: record recent positions
        self.tail_path = deque(maxlen=5)  # stores last 10 positions

    def collect(self, now):
        """Switch creature to follow mode when collected."""
        if self.state == CREATURE_SPAWNED:
            self.state = CREATURE_FOLLOW

    def update(self, player_pos, now, walls, leader_pos=None):
        """
        Update creature each frame:
          - Append current pos to tail path
          - Roam or follow logic
        """
        # Record for tail
        self.tail_path.append(self.pos.copy())
        if self.state == CREATURE_SPAWNED:
            # Roaming behavior
            if now >= self.next_dir_time:
                self.dir = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
                if self.dir.length() == 0:
                    self.dir = Vector2(1, 0)
                else:
                    self.dir = self.dir.normalize()
                self.next_dir_time = now + random.randint(1000, 3000)
            candidate = self.pos + self.dir * self.roam_speed
            candidate_rect = self.sprite.get_rect(center=(round(candidate.x), round(candidate.y)))
            if any(candidate_rect.colliderect(w) for w in walls):
                self.dir *= -1
            else:
                self.pos = candidate
        elif self.state == CREATURE_FOLLOW and leader_pos is not None:
            # Follow leader using recorded path externally
            pass
        self.rect.center = (round(self.pos.x), round(self.pos.y))

    def draw(self, surface):
        """
        Draw a wagging tail based on tail_path, then the kitten sprite.
        """
        if len(self.tail_path) >= 2:
            pts = list(self.tail_path)
            tail_pts = []
            t = pygame.time.get_ticks()
            for idx, p in enumerate(pts):
                nxt = pts[min(idx+1, len(pts)-1)]
                direction = (nxt - p)
                if direction.length() != 0:
                    normal = direction.normalize().rotate(90)
                else:
                    normal = Vector2(0, 1)
                phase = t/200 + idx * 0.5
                offset = normal * (math.sin(phase) * 4)
                tail_pts.append((p.x + offset.x, p.y + offset.y))
            pygame.draw.lines(surface, self.tail_color, False, tail_pts, 3)
        # Draw kitten on top
        surface.blit(self.sprite, self.rect)# Maze generation & drawing functions

def handle_player_movement(pos, sprite, walls):
    keys = pygame.key.get_pressed()
    dx = dy = 0
    if keys[pygame.K_LEFT]: dx = -PLAYER_SPEED
    if keys[pygame.K_RIGHT]: dx = PLAYER_SPEED
    if keys[pygame.K_UP]: dy = -PLAYER_SPEED
    if keys[pygame.K_DOWN]: dy = PLAYER_SPEED
    pos.x += dx; rect = sprite.get_rect(center=pos)
    if any(rect.colliderect(w) for w in walls): pos.x -= dx
    pos.y += dy; rect = sprite.get_rect(center=pos)
    if any(rect.colliderect(w) for w in walls): pos.y -= dy
    pos.x = max(0, min(WIDTH, pos.x)); pos.y = max(0, min(HEIGHT - EGG_ZONE_HEIGHT, pos.y))

def generate_maze():
    cols = math.ceil(WIDTH / TILE_SIZE)
    if cols % 2 == 0: cols += 1
    rows = math.ceil((HEIGHT - EGG_ZONE_HEIGHT) / TILE_SIZE)
    if rows % 2 == 0: rows += 1
    maze = [[1]*cols for _ in range(rows)]
    def carve(cx, cy):
        dirs = [(0,-1),(1,0),(0,1),(-1,0)]; random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<cols//2 and 0<=ny<rows//2:
                x1,y1 = cx*2+1, cy*2+1; x2,y2 = nx*2+1, ny*2+1
                if maze[y2][x2]==1:
                    maze[(y1+y2)//2][(x1+x2)//2]=0; maze[y2][x2]=0; carve(nx,ny)
    maze[1][1]=0; carve(0,0)
    cx, cy = cols//2, rows//2
    for oy in (-1,0,1):
        for ox in (-1,0,1): maze[cy+oy][cx+ox]=0
    return maze

def draw_maze(maze):
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell==1:
                pygame.draw.rect(screen, (0,0,0), (x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE))
# Main game loop

def main(load_data=None):
    # INITIAL SETUP
    maze = generate_maze()
    cols, rows = len(maze[0]), len(maze)
    # Precompute wall rects
    walls = [pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
             for y,row in enumerate(maze) for x,cell in enumerate(row) if cell]
    # Free cells for spawning
    free = [(x,y) for y,row in enumerate(maze) for x,cell in enumerate(row) if cell==0]
    # Spawn eggs and gems
    eggs = [Egg(c, r) for c,r in random.sample(free, min(NUM_EGGS,len(free)))]
    initial_gem_cells = [c for c in free if c not in [(e.col,e.row) for e in eggs]]
    gems = [Gem(c, r) for c,r in random.sample(initial_gem_cells, min(NUM_GEMS,len(initial_gem_cells)))]
    creatures = []
    # Player start at center
    center_x, center_y = cols//2, rows//2
    player = generate_player_sprite()
    player_pos = pygame.Vector2(center_x*TILE_SIZE+TILE_SIZE//2,
                                center_y*TILE_SIZE+TILE_SIZE//2)
    # Clear path & followers
    path_history.clear(); kitten_train.clear()
    # Initialize score tracking
    level = 1
    gem_score = 0
    # Track count of followers at start of level for cutscene trigger
    level_start_count = len(kitten_train)
    # Cutscene flags
    cutscene_active = False
    cutscene_start = 0
    cutscene_next_meow_time = 0  # schedule for silly meows during cutscene
    cutscene_gem_particles = []  # falling gems visuals during cutscene

    # ----------------------
    # LOAD SAVED STATE (if provided)
    # ----------------------
    if load_data:
        now_load = pygame.time.get_ticks()
        # Restore basic scores/state
        level = load_data.get('level', level)
        gem_score = load_data.get('gem_score', gem_score)
        # Restore maze
        if 'maze' in load_data and load_data['maze']:
            maze = load_data['maze']
            cols, rows = len(maze[0]), len(maze)
            walls = [pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
                     for y, row in enumerate(maze) for x, cell in enumerate(row) if cell]
            free = [(x, y) for y, row in enumerate(maze) for x, cell in enumerate(row) if cell == 0]
        # Restore player
        if 'player' in load_data:
            player_pos.x = load_data['player'].get('x', player_pos.x)
            player_pos.y = load_data['player'].get('y', player_pos.y)
        # Restore eggs
        eggs = []
        for ed in load_data.get('eggs', []):
            e = Egg(ed.get('col', 0), ed.get('row', 0))
            e.state = ed.get('state', UNCOLLECTED)
            if e.state == INVENTORY:
                remaining = ed.get('remaining_hatch', 60000)
                e.hatch_time = remaining
                e.collected_time = now_load  # start countdown from load time
            elif e.state == HATCHED:
                e.hatch_time = 0
                e.collected_time = now_load
            eggs.append(e)
        # Restore gems
        gems = []
        for gd in load_data.get('gems', []):
            g = Gem(gd.get('col', 0), gd.get('row', 0))
            g.state = gd.get('state', GEM_HIDDEN)
            if g.state == GEM_DISCOVERED:
                g.discovered_time = now_load
            gems.append(g)
        # Restore followers/kittens
        creatures = []
        kitten_train.clear()
        for fd in load_data.get('followers', []):
            fx, fy = fd.get('x', player_pos.x), fd.get('y', player_pos.y)
            cr = Creature(fx, fy)
            cr.state = CREATURE_FOLLOW
            cr.pos = Vector2(fx, fy)
            cr.rect.center = (round(fx), round(fy))
            if 'tail_color' in fd:
                cr.tail_color = tuple(fd['tail_color'])
            # Seed a small tail history so lines render immediately
            cr.tail_path.clear()
            for _ in range(5):
                cr.tail_path.append(cr.pos.copy())
            creatures.append(cr)
            kitten_train.append(cr)
        # Seed path history so the train starts following right away
        path_history.clear()
        for _ in range(200):
            path_history.appendleft((player_pos.x, player_pos.y))
        # Set baseline for this level so the next cutscene triggers correctly
        level_start_count = len(kitten_train)

    # GAME LOOP
    running = True
    # Initialize facing direction (down)
    player_direction = pygame.Vector2(0, 1)
    while running:
        now = pygame.time.get_ticks()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_s:
                save_game_to_file('savegame.json', level, gem_score, player_pos,
                                   kitten_train, eggs, gems, maze)
                # --- End save logic ---
                # Handle cutscene if active (before game updates)
        if cutscene_active:
            # Play silly meows at random stereo positions & volumes during cutscene
            if now >= cutscene_next_meow_time:
                cutscene_next_meow_time = now + random.randint(150, 400)
                play_sound('meow', volume=random.uniform(0.5, 1.0), pan=random.random())
            screen.fill((0, 0, 0))
            # --- Gem rain: more gems collected => more particles ---
            target_particles = min(150, 5 + gem_score * 3)
            # Spawn until we reach target count
            while len(cutscene_gem_particles) < target_particles:
                color = random.choice(BRIGHT_COLORS)
                sprite = generate_gem_sprite(color)
                particle = {
                    'x': random.randint(0, WIDTH),
                    'y': random.randint(-HEIGHT, 0),
                    'vy': random.uniform(2.0, 6.0),
                    'sprite': sprite
                }
                cutscene_gem_particles.append(particle)
            # Update & draw particles
            alive = []
            for p in cutscene_gem_particles:
                p['y'] += p['vy']
                rect = p['sprite'].get_rect(center=(p['x'], p['y']))
                screen.blit(p['sprite'], rect)
                if p['y'] < HEIGHT + 24:
                    alive.append(p)
            cutscene_gem_particles = alive
            t = now - cutscene_start
            x_offset = math.sin(t / 500.0) * 100
            y_offset = math.cos(t / 300.0) * 50
            total = len(kitten_train) + 1
            x_start = WIDTH // 2 - total * 40 // 2
            dancers = [player] + [cr.sprite for cr in kitten_train]
            for idx, sprite in enumerate(dancers):
                phase = t + idx * 200
                dx = x_offset * math.cos(phase / 300.0)
                dy = y_offset * math.sin(phase / 200.0)
                base_x = x_start + idx * 40
                base_y = HEIGHT // 2
                rect = sprite.get_rect(center=(base_x + dx, base_y + dy))
                screen.blit(sprite, rect)
            if now - cutscene_start >= 10000:
                # reset for new level
                maze = generate_maze()
                walls = [pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
                         for y,row in enumerate(maze) for x,cell in enumerate(row) if cell]
                free = [(x,y) for y,row in enumerate(maze) for x,cell in enumerate(row) if cell==0]
                eggs = [Egg(c, r) for c,r in random.sample(free, min(NUM_EGGS,len(free)))]
                initial_gem_cells = [c for c in free if c not in [(e.col,e.row) for e in eggs]]
                gems = [Gem(c, r) for c,r in random.sample(initial_gem_cells, min(NUM_GEMS,len(initial_gem_cells)))]
                path_history.clear()
                center_x = len(maze[0]) // 2
                center_y = len(maze) // 2
                player_pos.x = center_x * TILE_SIZE + TILE_SIZE // 2
                player_pos.y = center_y * TILE_SIZE + TILE_SIZE // 2
                for follower in kitten_train:
                    follower.pos = pygame.Vector2(player_pos.x, player_pos.y)
                    follower.rect.center = (round(player_pos.x), round(player_pos.y))
                level_start_count = len(kitten_train)
                cutscene_gem_particles.clear()
                # Auto-save new level state
                save_game_to_file('savegame.json', level, gem_score, player_pos,
                                   kitten_train, eggs, gems, maze)
                # Auto-save new level state
                save_game_to_file('savegame.json', level, gem_score, player_pos,
                                   kitten_train, eggs, gems, maze)
                cutscene_active = False
            pygame.display.flip()
            clock.tick(FPS)
            continue

        # Determine movement direction from input
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * PLAYER_SPEED
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * PLAYER_SPEED
        # Update facing if moving
        if dx != 0 or dy != 0:
            player_direction = pygame.Vector2(dx, dy)
        # Move player
        handle_player_movement(player_pos, player, walls)
        # Regenerate sprite with eye direction
        player = generate_player_sprite(SQUARE_SIZE, player_direction)
        path_history.appendleft((player_pos.x, player_pos.y))
        player_rect = player.get_rect(center=player_pos)

        # Eggs: collection and hatching -> spawn creatures
        for egg in eggs:
            if egg.state==UNCOLLECTED and player_rect.colliderect(egg.rect):
                egg.collect(now)
                play_sound('egg', volume=0.8)
            egg.update(now)
            if egg.state==HATCHED:
                creatures.append(Creature(egg.x, egg.y))
                play_sound('hatch', volume=0.9)
                # happy meow when a kitten appears
                play_sound('meow', volume=0.8, pan=random.random())
                egg.state = HATCHED + 1

        # Gems: discovery and respawn
        for gem in list(gems):
            if gem.state==GEM_HIDDEN and player_rect.colliderect(gem.rect):
                gem.discover(now)
                play_sound('gem', volume=0.8)
                # Increment gem score on discovery
                gem_score += 1
                for egg in eggs:
                    if egg.state==INVENTORY:
                        egg.hatch_time = max(0, egg.hatch_time - GEM_TIMER_REDUCTION)
                occupied = {(g.col,g.row) for g in gems} | {(e.col,e.row) for e in eggs}
                available = [c for c in free if c not in occupied]
                if available:
                    c, r = random.choice(available)
                    gems.append(Gem(c, r))
            gem.update(now)

        # Creatures: collect and follow
        for cr in creatures:
            if cr.state==CREATURE_SPAWNED and player_rect.colliderect(cr.rect):
                cr.collect(now)
                kitten_train.append(cr)
                play_sound('meow', volume=0.9, pan=random.random())
                # If last kitten collected for this level, start cutscene
                if (len(kitten_train) - level_start_count) == NUM_EGGS:
                    level += 1
                    cutscene_active = True
                    cutscene_start = now
                    cutscene_next_meow_time = now
                    cutscene_gem_particles = []  # reset rain particles
                    play_sound('level_up', volume=1.0)
            cr.update(player_pos, now, walls)

        # Update follower train positions
        for idx, follower in enumerate(kitten_train):
            delay = (idx + 1) * FOLLOW_DELAY
            if len(path_history) > delay:
                pos = path_history[delay]
                follower.pos = pygame.Vector2(pos)
                follower.rect.center = pos

        # Drawing
        screen.fill((50,150,200))
        # Draw score at top
        lvl_surf = font.render(f"Level: {level}", True, (255, 255, 255))
        kittens_surf = font.render(f"Kittens: {len(kitten_train)}", True, (255, 255, 255))
        gems_surf = font.render(f"Gems: {gem_score}", True, (255, 255, 255))
        draw_maze(maze)
        screen.blit(lvl_surf, (50, 10))
        screen.blit(kittens_surf, (400, 10))
        screen.blit(gems_surf, (700, 10))
        for egg in eggs: egg.draw(screen)
        for gem in gems: gem.draw(screen, now)
        for cr in creatures: cr.draw(screen)
        screen.blit(player, player_rect)

        # UI bar
        pygame.draw.rect(screen, (40,40,40), (0,HEIGHT-EGG_ZONE_HEIGHT, WIDTH, EGG_ZONE_HEIGHT))
        inv_e = [e for e in eggs if e.state in (INVENTORY,HATCHED)]
        for i,e in enumerate(inv_e):
            x = 10 + i*(SQUARE_SIZE+8); y = HEIGHT-EGG_ZONE_HEIGHT+EGG_ZONE_HEIGHT//2
            screen.blit(e.sprite, e.sprite.get_rect(center=(x,y)))
            if e.state==INVENTORY:
                elapsed = now - e.collected_time
                ratio = min(1, elapsed / e.hatch_time) if e.hatch_time>0 else 1
                bar_x = x - SQUARE_SIZE//2
                bar_y = y + SQUARE_SIZE//2 + 2
                pygame.draw.rect(screen, (100,100,100), (bar_x, bar_y, SQUARE_SIZE, 4))
                pygame.draw.rect(screen, (0,255,0), (bar_x, bar_y, int(SQUARE_SIZE*ratio), 4))
        inv_g = [g for g in gems if g.state==GEM_COLLECTED]
        for j,g in enumerate(inv_g):
            x = 10+len(inv_e)*(SQUARE_SIZE+8)+j*(SQUARE_SIZE+8)
            y = HEIGHT-EGG_ZONE_HEIGHT+EGG_ZONE_HEIGHT//2
            screen.blit(g.sprite, g.sprite.get_rect(center=(x,y)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

def show_menu():
    """Display menu and return selected option (keyboard or mouse)."""
    menu_font = pygame.font.SysFont(None, 48)
    title_font = pygame.font.SysFont(None, 96)  # Larger, fun title
    title_text = "Confetti Kittens"
    # Initialize per-letter colors (skip spaces)
    letter_indices = [i for i, ch in enumerate(title_text) if ch != ' ']
    title_colors = [random.choice(BRIGHT_COLORS) if ch != ' ' else None for ch in title_text]
    last_shift = pygame.time.get_ticks()
    SHIFT_INTERVAL = 200  # ms between color shifts

    options = ["Start New Game", "Load Saved Game"]
    selected = 0
    while True:
        # Shift title colors left->right over letters (ignore spaces)
        now = pygame.time.get_ticks()
        if now - last_shift >= SHIFT_INTERVAL:
            new_colors = title_colors[:]
            for j in range(len(letter_indices)-1, -1, -1):
                i = letter_indices[j]
                if j == 0:
                    new_colors[i] = random.choice(BRIGHT_COLORS)
                else:
                    new_colors[i] = title_colors[letter_indices[j-1]]
            title_colors = new_colors
            last_shift = now

        save_exists = os.path.exists('savegame.json')
        mouse_pos = pygame.mouse.get_pos()
        # Precompute option rects and hover BEFORE processing events so clicks work
        option_rects = []
        for i, opt in enumerate(options):
            tmp_surf = menu_font.render(opt, True, (255, 255, 255))
            opt_rect = tmp_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + i*50))
            option_rects.append(opt_rect)
        hovered = None
        for i, r in enumerate(option_rects):
            if r.collidepoint(mouse_pos):
                hovered = i
                selected = i

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif ev.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif ev.key == pygame.K_RETURN:
                    # Prevent selecting Load if no save exists
                    if options[selected] == "Load Saved Game" and not save_exists:
                        pass
                    else:
                        return options[selected]
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Click on hovered option
                if hovered is not None:
                    if options[hovered] == "Load Saved Game" and not save_exists:
                        pass
                    else:
                        return options[hovered]

        screen.fill((0, 0, 0))
        # Render rainbow-shifting title by letters
        # First compute total width for centering
        glyph_surfs = []
        total_w = 0
        for ch, col in zip(title_text, title_colors):
            if ch == ' ':
                w, h = title_font.size(' ')
                glyph_surfs.append((None, None, w))
                total_w += w
            else:
                surf = title_font.render(ch, True, col)
                glyph_surfs.append((surf, surf.get_rect(), surf.get_width()))
                total_w += surf.get_width()
        start_x = WIDTH//2 - total_w//2
        x = start_x
        y = HEIGHT//4 - title_font.get_height()//2
        for surf, rect, w in glyph_surfs:
            if surf is not None:
                rect.topleft = (x, y)
                screen.blit(surf, rect)
            x += w

        # Draw options with hover effect (using precomputed rects)
        for i, (opt, r) in enumerate(zip(options, option_rects)):
            # Grey out Load if no save file
            is_disabled = (opt == "Load Saved Game" and not save_exists)
            base_color = (180, 180, 180) if is_disabled else (255, 255, 255)
            highlight = (255, 255, 0) if i == selected else base_color
            # Hover background
            if i == selected and not is_disabled:
                bg = r.inflate(20, 10)
                pygame.draw.rect(screen, (50, 50, 50), bg)
                pygame.draw.rect(screen, (200, 200, 200), bg, 2)
            opt_surf = menu_font.render(opt, True, highlight)
            screen.blit(opt_surf, r)

        pygame.display.flip()
        clock.tick(FPS)

if __name__=="__main__":
    choice = show_menu()
    load_data = None
    if choice == "Load Saved Game":
        try:
            with open('savegame.json', 'r') as f:
                load_data = json.load(f)
        except FileNotFoundError:
            load_data = None
    main(load_data)
