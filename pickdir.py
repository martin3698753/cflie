import os

def list_directories(path):
    """Lists directories in a given path and returns a dictionary of directory names and their corresponding numbers."""
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directory_dict = {str(i+1): d for i, d in enumerate(directories)}
    for i, (num, dir_name) in enumerate(directory_dict.items()):
        print(f"{num}. {dir_name}")
    return directory_dict

def get_user_choice(directory_dict):
    """Gets user's choice of directory from the list."""
    while True:
        choice = input("Enter the number of the directory you want to choose: ")
        if choice in directory_dict:
            return directory_dict[choice]
        else:
            print("Invalid choice. Please try again.")

def choose_directory(path):
    """Chooses a directory from a given path and returns the full path."""
    directory_dict = list_directories(path)
    chosen_directory = get_user_choice(directory_dict)
    return os.path.join(path, chosen_directory)
