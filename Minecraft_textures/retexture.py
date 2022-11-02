import json
import os
import pickle
import numpy as np
from PIL import Image
import json


class GenTextures:
    imsize: tuple = (16,16)
    def __init__(self,pwd: str='./',input_dir: str='',output_dir: str=''):
        self.pwd = pwd
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.__load_data()

    def __load_data(self):
        f = open(self.pwd + 'data.json')
        data = json.load(f)
        self.colorPalette = data["textures"]
        self.type_blocks = data["type_blocks"]
        FILES = [x.split('.') for x in os.listdir(self.pwd + self.input_dir)] 
        self.TEXTURES = [x[0] if x [1]=='png' else None for x in FILES]
        self.sorted_blocks = {tb:[] for tb in self.type_blocks}
        self.sorted_blocks, self.not_sorted_blocks= self.__default_blocks()

    def generateTextures(self):
        for tb in self.sorted_blocks:
            for b in self.sorted_blocks[tb]:
                self.__save_texture(b, tuple(self.colorPalette[tb]))
        for b in self.not_sorted_blocks:
            im = Image.open(self.pwd + self.input_dir + b + ".png")
            im = np.array(im)
            self.__save_texture(b, (int(np.mean(im[:][0])),int(np.mean(im[:][1])),int(np.mean(im[:][2]))) )


    def __save_texture(self, b, color:tuple=(0,0,0)):
        Image.new('RGB', self.imsize,color).save(self.pwd + self.output_dir + b + '.png','png')

    def __default_blocks(self):
        sorted_blocks = self.sorted_blocks
        not_used_blocks = self.TEXTURES 
        for tb in self.type_blocks:
            for b in self.TEXTURES:
                for l in self.type_blocks[tb][0]:
                    for notl in self.type_blocks[tb][1]:
                        if (l in b) and (not str(notl) in b) and (b in not_used_blocks):
                            sorted_blocks[tb].append(b)
                            not_used_blocks.remove(b)
        print(sorted_blocks,not_used_blocks)

        return sorted_blocks, not_used_blocks

tx = GenTextures('Minecraft_textures/', 'blocks/', 'output/')
tx.generateTextures()