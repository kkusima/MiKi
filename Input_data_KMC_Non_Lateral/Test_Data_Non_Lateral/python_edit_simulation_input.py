import os, glob
files = glob.glob('Test_*/simulation_input.dat')
files.sort()

for file_to_edit_path in files:
    with open(file_to_edit_path, 'r') as file:
        # Reading the contents of the file
        lines = file.readlines()

        # Editing the desired line
    line_number_to_edit_1 = 28
    new_content_1 =  'override_array_bounds & & 200 200\n'
    lines[line_number_to_edit_1] = new_content_1

    with open(file_to_edit_path, 'w') as file:
        # Writing the modified lines back to the file
        file.writelines(lines)
