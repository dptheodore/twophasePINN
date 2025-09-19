import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from tables import STATES

# ignore divide by zero errors
np.seterr(divide='ignore', invalid='ignore')

RADIUS = 450 # for circle testing

class Grid:

    def __init__(self, scale, x_count, y_count):
        self.x_count = x_count
        self.y_count = y_count
        self.scale = scale
        self.values = np.zeros((x_count + 1) * (y_count + 1)).reshape(y_count + 1, x_count + 1)

    def set(self, x, y, v):

        self.values[y, x] = v

    def get(self, x, y):

        return self.values[y, x]


def get_state(a,b,c,d,iso):
    
    # Either 0 or 1
    # < iso = 0
    # > iso = 1

    sa = (a - iso) > 0
    sb = (b - iso) > 0
    sc = (c - iso) > 0
    sd = (d - iso) > 0

    # Index for lookup table
    s = round(sa * 8 + sb * 4 + sc * 2 + sd * 1)

    return s

def circle(x, y, cx, cy, radius):
    # distance from center point
    d = np.sqrt((cx-x) ** 2 + (cy-y) ** 2) - radius
    return d

def lerp(a, b, t):
    c = a + t * (b - a)
    return c

def lerp_points(p0, p1, t):

    return [
        lerp(p0[0], p1[0], t),
        lerp(p0[1], p1[1], t)
    ]

def find_lerp_factor(v0, v1, iso_val):
    t = (iso_val - v0) / (v1 - v0)
    return max(min(1, t), 0)

def map(x, y):
    d = circle(x, y, 540, 540, RADIUS)
    return d

def map_grid(grid: Grid):
    
    for i in range(grid.x_count + 1):
        for j in range(grid.y_count + 1):
            x = i * grid.scale
            y = j * grid.scale
            grid.set(i, j, map(x, y))

def march(grid: Grid, iso, interpolated=True):

    graph = []

    for i in range(grid.y_count):
        for j in range(grid.x_count):
            scale = grid.scale

            x = i * scale
            y = j * scale

            ## values at (corner) points

            v0 = grid.get(j    , i    )
            v1 = grid.get(j    , i + 1)
            v2 = grid.get(j + 1, i + 1)
            v3 = grid.get(j + 1, i    )

            ## Interopolation factors
            ta = find_lerp_factor(v0, v1, iso)
            tb = find_lerp_factor(v1, v2, iso)
            tc = find_lerp_factor(v3, v2, iso) # flip due to sign change
            td = find_lerp_factor(v0, v3, iso)

            ## edge point locations (interpolated)
            a = [x + ta * scale , y              ] # 01
            b = [x + scale      , y + tb * scale ] # 12
            c = [x + tc * scale , y + scale      ] # 23
            d = [x              , y + td * scale ] # 30

            ## edge point locations (not interpolated)
            _a = [x + 0.5 * scale , y               ] # 01
            _b = [x + scale       , y + 0.5 * scale ] # 12
            _c = [x + 0.5 * scale , y + scale       ] # 23
            _d = [x               , y + 0.5 * scale ] # 30

            edge_points = [a,b,c,d]

            if interpolated == False:
                edge_points = [_a,_b,_c,_d]

            state = get_state(v0, v1, v2, v3, iso)

            edges = STATES[state]
            
            for line in edges:
                p1 = edge_points[line[0]]
                p2 = edge_points[line[1]]
                graph.append((p1, p2))
    return graph

def values_from_image(grid, img, **kwargs):
    blur = kwargs.get("blur", 0)
    channel = kwargs.get("channel", "all").lower()

    if blur > 0:
        image = img.filter(ImageFilter.BoxBlur(blur))
    else:
        image = img

    w, h = image.size

    for i in range(grid.x_count):
        for j in range(grid.y_count):

            scale = grid.scale
            x = min(j * scale, w - 1)
            y = min(i * scale, h - 1)
            pixel = image.getpixel((x, y))
            brightness = np.mean(np.array(pixel))
            
            if channel == "r":
                v = pixel[0] - brightness
            elif channel == "g":
                v = pixel[1] - brightness
            elif channel == "b":
                v = pixel[2] - brightness
            else:
                v = brightness
            
            grid.set(j, i, v)

def main():

    # Main parameters
    interpolated = True
    iso = 10
    divisions = 75

    # Create a blank image
    x_res = 1080
    y_res = 1080
    img = Image.new('RGB', (x_res, y_res)) 

    # Or read an image
    # img = Image.open('test1080RGB.png')
    # x_res, y_res = img.size
    
    
    draw = ImageDraw.Draw(img)

    def center_ellipse(x,y,r,c):

        draw.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def draw_dots(grid):

        for j in range(grid.y_count):
            for i in range(grid.x_count):
                scale = grid.scale

                x = i * scale
                y = j * scale
                v = round(grid.get(i, j))
                if v < 0:
                    c = max(0, -v)
                    center_ellipse(x, y, 5, f'rgb({0},{c},{c})')
                else:
                    c = max(0, v)
                    center_ellipse(x, y, 5, f'rgb({c},{c},{c})')
                    

    grid_scale = round(x_res / divisions) # pixels
    x_divs = int(np.floor(x_res / grid_scale)) + 1
    y_divs = int(np.floor(y_res / grid_scale)) + 1

    grid = Grid(grid_scale, x_divs, y_divs)

    # set the values
    map_grid(grid)                                    # populate with a function
    # values_from_image(grid, img, channel="all", blur=10) # or read from an image

    draw_dots(grid)

    def draw_edges(grid, img_path):

        edges = march(grid, iso, interpolated)

        for line in edges:
            p1 = line[0]
            p2 = line[1]
            x1 = p1[1] # reversed x & y for consistency
            y1 = p1[0]
            x2 = p2[1]
            y2 = p2[0]
            draw.line([x1, y1, x2, y2], fill=f'rgb({0},{255},{255})',width=4)

        img.save(img_path)

    draw_edges(grid, "marched.png")

if __name__ == "__main__":
    main()
