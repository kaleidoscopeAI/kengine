import os

# Define the root directory of your project
root_dir = 'path/to/your/project'

# Walk through all directories and files
for subdir, _, files in os.walk(root_dir):
    for file in files:
        # Process only Python files
        if file.endswith('.py'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                content = f.read()
            # Replace the import statement
            new_content = content.replace('from types import', 'from app_types import')
            # If changes were made, overwrite the file
            if new_content != content:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                print(f'Updated imports in {file_path}')
