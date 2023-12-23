import easyocr
reader = easyocr.Reader(['en','ja'], gpu = True) # need to run only once to load model into memory
import cv2
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_BGR2RGB
import requests
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
# from manga_ocr import MangaOcr
# mocr = MangaOcr()
from googletrans import Translator
from io import BytesIO
import random
import string
import re


def imgshow(image):
    # random_text = generate_random_text()
    # cv2.imshow(random_text, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('Processing..')

def img_url(url):
    response = requests.get(url)

    image_data = BytesIO(response.content)

    image_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)

    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Return the decoded image
    return img

def readert(img):
  res = reader.readtext(img,detail =0)
  return res[0]

def pil_to_cv2(pil_image):
    # Convert PIL Image to NumPy array
    image_np = np.array(pil_image)

    # Convert RGB to BGR (OpenCV uses BGR by default)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_cv2

def process_image(input_image, scale_factor=1.0, blur_radius=0):
    try:
        # Scale up the image if a scale factor is provided
        if scale_factor > 1.0:
            new_width = int(input_image.width * scale_factor)
            new_height = int(input_image.height * scale_factor)
            input_image = input_image.resize((new_width, new_height), PILImage.LANCZOS)

        # Apply blur filter if a blur radius is provided
        if blur_radius > 0:
            input_image = input_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Return the processed image
        return input_image

    except Exception as e:
        print(f"Error: {e}")
        return None

def x2_img(img):
  imz = cv2_to_pil(img)
  return pil_to_cv2(process_image(imz,2))

def contains_only_numbers(input_string):
    # Define a regular expression pattern that matches only digits
    # pattern = '^[０-９,\s．]+$'
    pattern = '^-?\d+(\.\d+)?$'

    # Use re.match to check if the entire string matches the pattern
    match = re.match(pattern, input_string)

    # If there is a match, the string contains only numbers
    return match is not None

def enhance_img(image):
  img = image.copy()
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  denoised_image = cv2.fastNlMeansDenoisingColored(rgb, None, 10, 10, 7, 21)
  img2gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
  sharpened_image = cv2.filter2D(img2gray, -1, kernel=kernel)
  thresh = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  return thresh

def get_box(img):
  arr = []
  image = img.copy()
  res = reader.readtext(img)

  for (bbox, text, prob) in res:

    # unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # cv2.rectangle(img1, tl, br, (0, 255, 0), 2)
    x, y, w, h = tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]
    arr.append((x, y, w, h))
  return arr

def show_box(img, arr):
  imgz = img.copy()
  for item in arr:
    x,y,w,h = item[0],item[1],item[2],item[3]
    cropped = img[y:y + h, x:x + w]

    fst, scnd = most_frequent_rgb(cropped)
    # jp_text = text_mgocr(x2_img(cropped))
    try:
      jp_text = readert(x2_img(cropped))
    except:
      continue
    # jp_text = readert(x2_img(cropped))
    imgt = text_to_image(trans_jp(jp_text), w,h,fst,scnd)
    tcheck = contains_only_numbers(jp_text)
    print(jp_text, tcheck)
    if tcheck == False:
      imgshow(imgt)
      # print(x,y,w,h)
      imgz[y:y + h, x:x + w] = imgt
      imgshow(cropped)
      print("")
  cv2.imwrite("./uploads/result.jpg", imgz)

def draw_reg(img, arr):
  for item in arr:
    x,y,w,h = item[0],item[1],item[2],item[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
  imgshow(img)

def show_colored_rectangle(color_tuple):
    # Create a black image
    width, height = 100, 50
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert the color tuple to a NumPy array
    color_np = np.array(color_tuple)

    # Ensure the color values are in the valid range [0, 255]
    color_np = np.clip(color_np, 0, 255)

    # Convert to uint8
    color_np = color_np.astype(np.uint8)

    # Draw a rectangle filled with the specified color
    image = cv2.rectangle(image, (0, 0), (100, 50), color_np.tolist(), thickness=cv2.FILLED)

    # Display the image
    imgshow(image)

def most_frequent_rgb(image):
    """
    Find the most frequent RGB color in an image.

    Args:
        image (numpy.ndarray): The image to analyze.

    Returns:
        tuple: The most frequent RGB color in the image.
    """
    # Flatten the image into a 1D array
    flattened_image = image.reshape(-1, 3)

    # Count the occurrences of each RGB color
    color_counts = np.unique(flattened_image, axis=0, return_counts=True)[1]

    # Find the index of the most frequent color
    most_frequent_color_index = np.argmax(color_counts)
    second_arr = np.delete(color_counts, most_frequent_color_index)
    second_frequent_color_index = np.argmax(second_arr)

    most = tuple(np.unique(flattened_image, axis=0)[most_frequent_color_index])
    second = tuple(np.unique(flattened_image, axis=0)[second_frequent_color_index])
    if (most[2] >220 and most[1] >220 and most[0] >220):
      second = (0,0,0)

    # show_colored_rectangle(most_frequent_color)
    # show_colored_rectangle(second_frequent_color)
    return [(most[2],most[1],most[0]),(second[2],second[1],second[0])]

def calculate_font_size(text, img_w, img_h, font_path):
    font_size = max(img_w, img_h)

    while True:
        temp_image = PILImage.new("RGB", (img_w, img_h), (255, 255, 255))
        temp_draw = ImageDraw.Draw(temp_image)
        temp_font = ImageFont.truetype(font_path, font_size)

        # Use textbbox instead of textsize
        text_bbox = temp_draw.textbbox((0, 0), text, font=temp_font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        if text_w <= img_w and text_h <= img_h*0.5:
            break

        font_size -= 1

    return font_size

def text_to_image(text, img_w, img_h, bg_cl, text_cl, font_path="Arial.ttf"):
    font_size = calculate_font_size(text, img_w, img_h, font_path)

    image = PILImage.new("RGB", (img_w, img_h), bg_cl)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    x = (img_w - (text_bbox[2] - text_bbox[0])) // 2
    y = (img_h - (text_bbox[3] - text_bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=text_cl)

    nimg = np.array(image)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    return ocvim

def cv2_to_pil(cv2_image):
    """
    Convert an OpenCV image (cv2) to a Pillow (PIL) image.

    Parameters:
    cv2_image (numpy.ndarray): The input OpenCV image.

    Returns:
    PIL.Image.Image: The converted PIL image.
    """
    if len(cv2_image.shape) == 2:
        # Grayscale image
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
    elif cv2_image.shape[2] == 4:
        # RGBA image
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2RGB)

    # Convert BGR to RGB
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL image
    pil_image = PILImage.fromarray(cv2_image_rgb)

    return pil_image

# def text_mgocr(imagez):
#     image = cv2_to_pil(imagez)
#     # mocr = MangaOcr()
#     text = mocr(image)
#     return text
def trans_jp(text, lang = 'en'):    # using https://rapidapi.com/falcondsp/api/opentranslator/
    translator = Translator()
    translation = translator.translate(text, src='ja', dest = lang)
    return translation.text

def main():
  img = cv2.imread("./uploads/input.jpg")
  img2 = enhance_img(img)
  imgshow(img2)
  boxes = get_box(img2)
  # print(len(boxes))
  show_box(img, boxes)
  # draw_reg(img, boxes)
  return('result.jpg')