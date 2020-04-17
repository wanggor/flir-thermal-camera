import numpy as np
from PIL import Image,ImageFont, ImageDraw

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
            # names[name.strip('\n')] = ID
    return names

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


class Drawing_obj():
    def __init__(self, class_path, font_path ):
        self.classes = read_class_names(class_path)
        self.colorHelper   = MplColorHelper('Paired',0,len(self.classes))
        self.font_path = font_path


    def draw_text(self, frame, text , left, top, right, bottom):

        
        font = ImageFont.truetype(font=self.font_path,size=np.floor(3e-2 * frame.size[1] + 0.5).astype('int32'))
        thickness = (frame.size[0] + frame.size[1]) // 700

        draw = ImageDraw.Draw(frame)

        label_size = draw.textsize(text, font)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, bottom ])
        else:
            text_origin = np.array([left, bottom + 1])

        draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill= (0, 255, 0, 100))

        draw.text(text_origin, text.capitalize(), fill=(0, 0, 0), font=font)

        del draw



    def draw(self, frame, left, top, right, bottom, ind_class, text  = None):
        

        font = ImageFont.truetype(font=self.font_path,size=np.floor(3e-2 * frame.size[1] + 0.5).astype('int32'))
        thickness = (frame.size[0] + frame.size[1]) // 700

        draw = ImageDraw.Draw(frame)

        ind_class = int(ind_class)

        color = self.colorHelper.get_rgb(ind_class)
        color = (int(color[2]*255),int(color[1]*255),int(color[0]*255),int(color[3]*255))

        for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
        
        if text is not None:
            text = str(text)
            text = f"{self.classes[ind_class]} : {text}"
            label_size = draw.textsize(text, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top ])
            else:
                text_origin = np.array([left, top + 1])

            
            draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=color)

            draw.text(text_origin, text.capitalize(), fill=(0, 0, 0), font=font)

        del draw