from PIL import Image, ImageDraw
import numpy as np

def generate_brick_texture(width=512, height=512, brick_width=60, brick_height=30, gap=5, brick_color=(170, 74, 68), mortar_color=(138, 142, 143)):
    """Generates a procedural brick texture."""
    image = Image.new('RGB', (width, height), color=mortar_color)
    draw = ImageDraw.Draw(image)

    for y in range(0, height, brick_height + gap):
        for x in range(0, width, brick_width + gap):
            row = (y // (brick_height + gap)) % 2
            offset = (brick_width // 2) if row == 1 else 0
            x_pos = x - offset

            # Draw full bricks
            draw.rectangle([x_pos, y, x_pos + brick_width, y + brick_height], fill=brick_color)
            # Draw wrapped-around bricks for seamless tiling
            if x_pos < 0:
                draw.rectangle([x_pos + width, y, width, y + brick_height], fill=brick_color)
            if x_pos + brick_width > width:
                 draw.rectangle([0, y, x_pos + brick_width - width, y + brick_height], fill=brick_color)

    return image

def generate_tile_texture(width=512, height=512, tile_size=64, color1=(240, 240, 240), color2=(220, 220, 220)):
    """Generates a procedural tile texture."""
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if (x // tile_size) % 2 == (y // tile_size) % 2:
                pixels[x, y] = color1
            else:
                pixels[x, y] = color2
    return image

def generate_wood_texture(width=512, height=512, frequency=0.02, turbulence=5):
    """Generates a simplified, stylized wood grain texture without a noise library."""
    image = Image.new('RGB', (width, height))
    pixels = np.array(image)

    # Base wood colors
    color1 = np.array([160, 110, 70])  # Darker wood color
    color2 = np.array([190, 140, 90]) # Lighter wood color

    for y in range(height):
        for x in range(width):
            # Simple sine wave pattern for grain, with some turbulence
            base_value = np.sin(y * frequency * (1 + 0.1 * np.sin(x * 0.05)))
            # Add some turbulence to make it less regular
            turbulence_value = np.sin(x * 0.1) * np.cos(y * 0.1) * turbulence
            value = (base_value + turbulence_value) / (1 + turbulence)

            # Interpolate between the two colors based on the value
            color = color1 * (1 - value) + color2 * value
            pixels[y, x] = np.clip(color, 0, 255).astype(np.uint8)

    return Image.fromarray(pixels, 'RGB')
