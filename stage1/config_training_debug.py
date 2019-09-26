# config = { 'train_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/Data/public/Data/',
#           'val_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/Data/public/Data/',
#           'test_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/Data/public/Data/',
config = {'train_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/data_preprocess/test_2/',
          # 'test_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/data_preprocess/test/',
          'test_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/data_preprocess/test_2/',
          'val_preprocess_result_path': '/home/linyi/Code/pytorch/pe_detection/data_preprocess/test_2/',
          'trainfilelist': ['PAT001', 'PAT002', 'PAT003'],
          'testfilelist': ['PAT004', 'PAT005'],
          # 'black_list': ['PAT006', 'PAT010', 'PAT024', 'PAT031', 'PAT032', 'PAT025', 'PAT030', 'PAT033', 'PAT004', 'PAT034'],
          'black_list': ['PAT006', 'PAT024', 'PAT031', 'PAT032',
                        'PAT017', 'PAT018', 'PAT019', 'PAT020', 'PAT033', 'PAT035'],

          'preprocessing_backend': 'python',

          }