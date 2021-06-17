import os
import random
import shutil

root_dir = 'COVID-19_Radiography_Dataset'
classes = ['normal', 'viral', 'covid']

if os.path.isdir(os.path.join(root_dir, classes[1])):
    for c in classes:
        test_images = [x for x in os.listdir(os.path.join(root_dir, 'test', c)) if x.lower().endswith('png')]
        for image in test_images:
            source_path = os.path.join(root_dir, 'test', c, image)
            target_path = os.path.join(root_dir, c, image)
            shutil.move(source_path, target_path)

        train_images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
        new_test_images = random.sample(train_images, 100)
        for image in new_test_images:
            source_path = os.path.join(root_dir, c, image)
            target_path = os.path.join(root_dir, 'test', c, image)
            shutil.move(source_path, target_path)

