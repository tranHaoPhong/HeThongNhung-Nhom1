import binascii

tflite_model_name = 'hand_model.tflite'
header_file_name = 'hand_model.h'

# Đọc tệp .tflite
with open(tflite_model_name, 'rb') as f:
    tflite_model = f.read()

# Chuyển đổi sang định dạng hex
hex_model = binascii.hexlify(tflite_model).decode('utf-8')

# Chia hex data thành các dòng phù hợp cho C/C++
hex_lines = [hex_model[i:i+80] for i in range(0, len(hex_model), 80)]

# Tạo tệp header .h từ hex data
with open(header_file_name, 'w') as f:
    f.write('#ifndef HAND_MODEL_H\n')
    f.write('#define HAND_MODEL_H\n\n')
    f.write('const char g_model[] = {\n')
    for line in hex_lines:
        f.write('    "' + line + '"\n')
    f.write('};\n\n')
    f.write('#endif // HAND_MODEL_H\n')
