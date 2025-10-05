import os


def find_files_with_extension(folder_path, extension):
    files_found = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                files_found.append(os.path.join(root, file))
    return files_found

