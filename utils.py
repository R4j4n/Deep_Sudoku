import cv2
import math
import numpy as np
from scipy import ndimage
from utils import *
from backtracking import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def thresholding(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.GaussianBlur(gray, (7,7), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    thres = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)
    thres = cv2.resize(thres, (800, 800))
    return thres

def wrap_img(img):
    contours , _ = cv2.findContours(img , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    pts = biggest.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = maxWidth
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped 

def split_Cell(image):
    image_height = image.shape[0]
    image_width = image.shape[1]
    cell_height  = image_height // 9 
    cell_width = image_width // 9 
    rects = []
    cells = []
    for i in range(9):
        for j in range(9):
            p1 = (j*cell_height , i*cell_width)
            p2 = ((j+1)*cell_height , (i+1)*cell_width)
            rects.append((p1, p2))
            cv2.rectangle(image, p1, p2, (255,0,0),3)
    for coords in rects:
        rect = image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]
        cells.append(rect)
    return cells

def crop_tile(img):
    height , width = img.shape[:2]
    start_r , start_c = int(height*0.15) , int(width*0.15)
    end_r , end_c = int(height*0.85) , int(width*0.85)
    cropped = img[start_r:end_r , start_c:end_c]
    return cropped

# EXPLAINED ON : https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
def get_text_center(img):
    gray = cv2.resize(img , (28,28))

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
         gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    return gray

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

            

def beautiful_graphics_version2(grid,old_grid):
    answer_frame = np.ones((450,450,3), np.uint8)
    # answer_frame = np.zeros([450,450,3],dtype=np.uint8)
    # answer_frame.fill(255)
    height = 50
    width = 50
    th = 2
    font_scale = 2
    font_th = 2
    font_letter = cv2.FONT_HERSHEY_PLAIN
    for i in range(9):
        for j in range(9):
            text = str(grid[i][j])
            old_text = str(old_grid[i][j])
            x = width*j
            y = height*i
            cv2.rectangle(answer_frame, (x + th, y + th), (x + width - th, y + height - th), (255,255,255), th)
            text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
            width_text, height_text = text_size[0], text_size[1]
            text_x = int((width - width_text) / 2) + x
            text_y = int((height + height_text) / 2) + y
            if text == old_text:
                cv2.putText(answer_frame, text, (text_x, text_y), font_letter, font_scale, (0,0,255), font_th)
            else:
                cv2.putText(answer_frame, text, (text_x, text_y), font_letter, font_scale, (255, 255, 0), font_th)
    
    return answer_frame 


def find_result(path):
    model = load_model('digit.h5')
    img = cv2.imread(path)
    p_img = thresholding(img)
    wrapped_img = wrap_img(p_img )
    tiles = split_Cell(wrapped_img)
    digits_highlight = []
    predictions = []
    for tile in tiles:

        crop = crop_tile(tile)
        count_white_pixles = cv2.countNonZero(crop)
        
        if count_white_pixles > 150 :
            centered_Text = get_text_center(crop)
            img = cv2.resize(centered_Text,(28,28))
            img = img.reshape((1,28,28,1))
            x = img.astype('float32')/255
            result = model.predict(x)
            index = np.argmax(result) + 1
            predictions.append(index)
            digits_highlight.append(index)
            
        else:
            x = 0
            predictions.append(x)
            digits_highlight.append(x)

    board = np.array(predictions).reshape((9, 9))
    solver = BackTracing(board)
    solver.solve()
    final = solver.bo
    board_h = np.array(digits_highlight).reshape((9, 9))
    final_result = beautiful_graphics_version2(final,board_h)
    return final_result




