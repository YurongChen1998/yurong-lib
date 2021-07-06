import os
from AE_test import test
import shutil

test_path = 'images/test'
output_path = 'test/result'
num_path = os.listdir(test_path)

for i in range(len(num_path)):
    img_path = os.path.join(test_path, num_path[i])
    #os.rmdir('test/input')
    #os.rmdir('test/output')
    os.mkdir('test/input')
    os.mkdir('test/output')
    op_path = os.path.join(output_path, num_path[i])
    #os.rmdir(op_path)
    os.mkdir(op_path)
    test(img_path)
    shutil.move('test/input', os.path.join(op_path, 'input'))
    shutil.move('test/output', os.path.join(op_path, 'output'))
