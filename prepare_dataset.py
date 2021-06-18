import os
import random
import shutil

root_dir = 'COVID-19_Radiography_Dataset'
classes = ['normal', 'viral', 'covid']
class_dirs = ['Normal', 'Viral Pneumonia', 'COVID']

if os.path.isdir(os.path.join(root_dir, class_dirs[1])):
    os.mkdir(os.path.join(root_dir, 'test'))

    for i, d in enumerate(class_dirs):
        os.rename(os.path.join(root_dir, d), os.path.join(root_dir, classes[i]))

    for c in classes:
        os.mkdir(os.path.join(root_dir, 'test', c))

    for c in classes:
        images = [name for name in os.listdir(os.path.join(root_dir, c)) if name.lower().endswith('png')]
        selected_images = random.sample(images, 200)
        for image in selected_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)
