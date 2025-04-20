from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a white background image
    width, height = 600, 200
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, or use default if not available
    try:
        # Try to use Arial or another system font
        font_path = "arial.ttf"  # This should work on Windows
        title_font = ImageFont.truetype(font_path, 60)
        subtitle_font = ImageFont.truetype(font_path, 20)
    except Exception:
        # Use default font if custom font fails
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    
    # Colors
    primary_color = (25, 47, 107)  # Dark blue
    secondary_color = (59, 130, 246)  # Lighter blue
    
    # Draw the text
    text = "TwinTitles"
    text_width = draw.textlength(text, font=title_font)
    text_x = (width - text_width) / 2
    draw.text((text_x, 50), text, font=title_font, fill=primary_color)
    
    # Draw two document icons next to each other
    # First document
    draw.rectangle((text_x - 80, 60, text_x - 30, 110), outline=primary_color, width=3)
    # Second document - slightly offset
    draw.rectangle((text_x - 65, 45, text_x - 15, 95), outline=secondary_color, width=3)
    
    # Draw subtitle
    subtitle = "Research Title Similarity Analyzer"
    subtitle_width = draw.textlength(subtitle, font=subtitle_font)
    subtitle_x = (width - subtitle_width) / 2
    draw.text((subtitle_x, 130), subtitle, font=subtitle_font, fill=secondary_color)
    
    # Save the image
    logo_path = "static/img/logo.png"
    image.save(logo_path)
    print(f"Logo created and saved to {logo_path}")

if __name__ == "__main__":
    create_logo() 