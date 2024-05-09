def get_min_line_length(file1_path, file2_path):
    # 打开文件并获取第一行的长度
    with open(file1_path, 'r') as file1:
        min_length = len(file1.readline().strip())
    with open(file2_path, 'r') as file2:
        min_length = min(min_length, len(file2.readline().strip()))
    return min_length

def trim_lines(file_path, min_length):
    trimmed_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            # 截取每一行的长度为最小长度
            trimmed_line = line.strip()[:min_length]
            trimmed_lines.append(trimmed_line)
    return trimmed_lines

def write_to_file(trimmed_lines, output_file):
    with open(output_file, 'w') as file:
        for line in trimmed_lines:
            file.write(line + '\n')

def align_txt_files(file1_path, file2_path):
    min_length = get_min_line_length(file1_path, file2_path)
    trimmed_lines_file1 = trim_lines(file1_path, min_length)
    trimmed_lines_file2 = trim_lines(file2_path, min_length)
    write_to_file(trimmed_lines_file1, './K_3/data_K_3/data_processed/raw_del.txt')
    write_to_file(trimmed_lines_file2, './K_3/data_K_3/data_processed/steg_del.txt')

# Example usage:
file1_path = './K_3/data_K_3/data_processed/raw.txt'
file2_path = './K_3/data_K_3/data_processed/steg.txt'
#output_file = 'aligned_files'
align_txt_files(file1_path, file2_path)
