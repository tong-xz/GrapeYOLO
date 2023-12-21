import os

path = 'data/images'

def get_file_names(folder_path):
    file_names_without_extension = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            name, extension = os.path.splitext(file)
            file_names_without_extension.append(name)
    return file_names_without_extension

# 使用示例
image_names = get_file_names(path)
print(len(image_names))

label_names = get_file_names(path)
print(len(label_names))
print(label_names == image_names)